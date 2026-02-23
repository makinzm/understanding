# Meta Information

- URL: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Qiying Yu et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. ByteDance Seed / Tsinghua University AIR / The University of Hong Kong.

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) is a fully open-source reinforcement learning system for training large language models (LLMs) on mathematical reasoning tasks. It was developed by ByteDance Seed, Tsinghua University AIR, and The University of Hong Kong, and targets researchers and engineers who need a reproducible alternative to closed-source reasoning systems like OpenAI o1 and DeepSeek-R1.

- **Input**: A base LLM (Qwen2.5-32B) and a curated dataset of math problems with rule-based verifiable answers.
- **Output**: A fine-tuned LLM capable of long-chain-of-thought (CoT) mathematical reasoning, evaluated on AIME 2024.

## Background: GRPO and Its Limitations

DAPO builds on **Group Relative Policy Optimization (GRPO)**, which extends PPO by replacing a learned value function with group-based advantage estimation. For a prompt $q$ with $G$ sampled responses $\{o_1, \ldots, o_G\}$, the group-normalized advantage for the $t$-th token of response $i$ is:

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

where $R_i \in \{0, 1\}$ is the rule-based binary reward (correct/incorrect final answer). The GRPO objective is:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

where $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ is the importance sampling ratio.

The authors identified four failure modes in vanilla GRPO when applied to long-CoT reasoning:

1. **Entropy collapse**: symmetric clipping prevents exploration of low-probability tokens.
2. **Training signal vanishing**: prompts where all $G$ responses are all-correct or all-incorrect yield $\hat{A}_{i,t} = 0$, providing zero gradient.
3. **Length bias**: sample-level loss averaging under-weights longer, more informative reasoning chains.
4. **Truncation noise**: responses cut off at max length receive the same reward penalty as fully wrong answers, introducing noisy training signal.

## DAPO Algorithm

The DAPO objective removes the KL divergence constraint and introduces **decoupled clipping bounds** $\varepsilon_{\text{low}}$ and $\varepsilon_{\text{high}}$:

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A}_{i,t}\right)\right]$$

Key changes from GRPO:
- No KL penalty term ($\beta = 0$)
- Asymmetric clip bounds ($\varepsilon_{\text{low}} \neq \varepsilon_{\text{high}}$)
- Token-level normalization (divide by total token count $\sum_i |o_i|$ instead of per-sample)

### Technique 1: Clip-Higher (Entropy Regularization)

**Problem**: Symmetric clipping at $[1-\varepsilon, 1+\varepsilon]$ limits how much probability mass can be reassigned to low-probability tokens. When those tokens are correct, the policy cannot sufficiently boost them, leading to entropy collapse as the model converges prematurely to high-confidence but narrow solutions.

**Solution**: Decouple into $\varepsilon_{\text{low}} = 0.2$ (conservative downward clip) and $\varepsilon_{\text{high}} = 0.28$ (more permissive upward clip). This asymmetry allows greater probability increases for exploratory tokens while still penalizing large probability decreases.

> [!NOTE]
> The authors observe that simply removing KL divergence (i.e., $\beta = 0$) is insufficient to prevent entropy collapse; the asymmetric clipping is critical.

### Technique 2: Dynamic Sampling (Filtering Zero-Gradient Prompts)

**Problem**: When all $G$ rollouts for a prompt have the same reward (all correct or all incorrect), the advantage $\hat{A}_{i,t} = 0$ for every token, contributing zero gradient.

**Solution**: During each training step, filter out prompts where accuracy $\text{acc}(q) = \frac{1}{G}\sum_i \mathbb{1}[R_i = 1]$ satisfies $\text{acc}(q) \in \{0, 1\}$. Replace filtered prompts by resampling from the prompt pool to maintain a constant effective batch size.

**Pseudocode:**

```
for each batch B of prompts:
    sample G rollouts per prompt
    compute acc(q) for each prompt q
    filter: B_valid = {q ∈ B : 0 < acc(q) < 1}
    while |B_valid| < target_batch_size:
        resample new prompt q' from dataset
        sample G rollouts for q'
        if 0 < acc(q') < 1: add q' to B_valid
    update policy using B_valid
```

### Technique 3: Token-Level Policy Gradient Loss

**Problem**: Sample-level loss averaging (dividing by $G$ and averaging over tokens per sample) means short samples contribute disproportionately compared to long reasoning chains.

**Solution**: Normalize by the total number of tokens across all $G$ responses: $\frac{1}{\sum_{i=1}^G |o_i|}$. This ensures each token contributes equally to the gradient regardless of which response it belongs to.

> [!IMPORTANT]
> This is especially significant in long-CoT settings where response lengths vary from hundreds to thousands of tokens. A 10,000-token response chain should not receive the same gradient weight as a 100-token trivial response.

### Technique 4: Overlong Reward Shaping

**Problem**: Responses truncated at $L_{\max}$ tokens may contain partially correct reasoning but receive the same harsh reward as completely wrong answers, introducing noisy gradients.

**Solution**: Apply a soft penalty function for responses approaching or exceeding the length limit:

$$r_{\text{shape}}(o_i) = \begin{cases} 0 & \text{if } |o_i| \leq L_{\max} \\ -\frac{|o_i| - L_{\max}}{L_{\text{cache}} - L_{\max}} & \text{if } L_{\max} < |o_i| \leq L_{\text{cache}} \\ -1 & \text{if } |o_i| > L_{\text{cache}} \end{cases}$$

where $L_{\text{cache}}$ is the KV-cache limit ($L_{\text{cache}} = 20{,}480$ tokens in experiments). Truncated responses receive a graduated penalty $\in [-1, 0]$ rather than a full $-1$ penalty.

## Dataset: DAPO-Math-17K

The authors curate a dataset of 17,000 mathematical competition problems, denoted **DAPO-Math-17K**, with the following characteristics:

- **Sources**: Art of Problem Solving (AoPS) forums and official mathematics competition archives.
- **Preprocessing**: All answers normalized to integers to enable deterministic string-matching reward computation.
- **Scope**: Competition-level mathematics including AMC, AIME, and olympiad-style problems.
- **Purpose**: Provides a distribution where problem difficulty is calibrated so that $0 < \text{acc}(q) < 1$ for a 32B model, keeping dynamic sampling effective.

> [!NOTE]
> The dataset is fully open-sourced alongside training code to address the lack of reproducibility in prior systems like DeepSeek-R1, which released weights but withheld datasets and exact training procedures.

## Experiments

- **Dataset**: DAPO-Math-17K (17,000 math problems from AoPS and competition archives)
- **Hardware**: Not explicitly specified, but training uses the `verl` distributed RL framework
- **Base Model**: Qwen2.5-32B (pre-trained base, not instruction-tuned)
- **Optimizer**: AdamW, learning rate $1 \times 10^{-6}$
- **Batch Configuration**: 512 prompts per step, $G = 16$ rollouts per prompt ($= 8{,}192$ total samples)
- **Hyperparameters**: $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$, $L_{\max} = 16{,}384$ tokens, $L_{\text{cache}} = 20{,}480$ tokens
- **Evaluation Metric**: AIME 2024, avg@32 (average accuracy over 32 independent samples)
- **Baseline**: Naive GRPO re-implemented under identical conditions

### Ablation Results (AIME 2024 avg@32)

| Configuration | AIME 2024 Score |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering (remove truncated) | 36 |
| + Clip-Higher ($\varepsilon_{\text{high}} = 0.28$) | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-Level Policy Gradient Loss | 42 |
| + Dynamic Sampling (full DAPO) | **50** |

Each technique contributes incrementally, with dynamic sampling providing the largest single gain (+8 over token-level loss).

### Comparison with Prior Work

| Method | AIME 2024 | Training Steps |
|---|---|---|
| DeepSeek-R1-Zero-Qwen-32B | 47 | ~full |
| DAPO (Qwen2.5-32B) | **50** | ~50% fewer |

DAPO outperforms DeepSeek-R1-Zero on the same base model while requiring half the training compute.

## Emergent Behaviors

The authors document that **reflection and backtracking** behaviors (where the model explicitly revisits its reasoning steps) emerge organically during DAPO training without being explicitly rewarded. These behaviors are absent in the base model's outputs before RL fine-tuning and appear progressively as training progresses, suggesting that long-CoT structure self-organizes under reward pressure.

## Comparison with Related Methods

| Method | KL Constraint | Clipping | Loss Aggregation | Zero-Gradient Filtering | Open-Source |
|---|---|---|---|---|---|
| PPO | Yes ($\beta > 0$) | Symmetric | Sample-level | No | Partial |
| GRPO | Yes ($\beta > 0$) | Symmetric | Sample-level | No | Partial |
| DeepSeek-R1-Zero | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Weights only |
| **DAPO** | **No** ($\beta = 0$) | **Asymmetric** | **Token-level** | **Yes** | **Full** |

> [!TIP]
> DAPO is implemented on top of the [verl](https://github.com/volcengine/verl) framework, which supports distributed rollout and training for large-scale LLM RL.

## Applicability

DAPO is suited for:
- **Researchers** studying LLM reasoning who need a fully reproducible RL baseline.
- **Practitioners** applying RL to verifiable tasks (math, code, formal proofs) where rule-based reward signals are available.
- **Scale**: Designed for 32B+ parameter models; smaller models may not exhibit the same entropy collapse patterns that motivate the techniques.
- **Prerequisites**: Requires a dataset where problems have deterministically verifiable answers, and where model accuracy is in the intermediate range ($0 < \text{acc} < 1$) so dynamic sampling can function effectively.
