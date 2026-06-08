# Meta Information

- URL: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

This paper presents DeepSeekMath 7B, an open-source language model specialized for mathematical reasoning. The key insight is that **data quality and training methodology** matter more than raw model size — a 7B parameter model trained on a carefully curated 120B-token math corpus with a novel RL algorithm achieves performance competitive with models 77× larger (Minerva 540B).

**Applicable context:** Practitioners building math-focused LLMs, researchers studying RL for reasoning, or anyone who needs a capable open-source math reasoning model at the 7B scale.

## 1. Introduction and Motivation

Mathematical reasoning is a benchmark for general intelligence in language models. Prior open-source work (Llemma, InternLM-Math, MetaMath) lagged behind closed systems like GPT-4 and Gemini-Ultra. DeepSeekMath addresses three gaps:

1. **Data:** Prior math corpora were small (OpenWebMath: ~13.6B tokens). DeepSeekMath constructs a 120B-token corpus.
2. **Pretraining:** Choosing the right base model and training mix matters significantly.
3. **RL alignment:** PPO requires a large critic model. GRPO eliminates it.

## 2. DeepSeekMath Corpus (Data Collection)

### 2.1 Iterative fastText Pipeline

**Input:** Common Crawl (CC) web crawl dump
**Output:** ~120B math-related tokens after 4 collection iterations

The pipeline iterates as follows:

```
Iteration 0 (seed):
  - Use OpenWebMath (13.6B tokens) as positive examples
  - Sample 500K positive + 500K negative from CC for fastText training

For each iteration i = 1, 2, 3, 4:
  1. Train fastText binary classifier on current labeled set
  2. Score all CC pages; select high-confidence math pages
  3. Manual annotation of domains where >10% pages are classified as math
     → Reclassify entire domain as math-related
  4. Decontaminate: remove pages with ≥10-gram overlap with benchmarks
     (GSM8K, MATH, CMATH, AGIEval)
  5. Add new pages to corpus; update positive examples for next round
```

**Why iterative?** Each iteration discovers new math-rich domains missed by previous classifiers. Domain-level boosting reduces false negatives (e.g., entire math education websites miscategorized on page-by-page basis).

### 2.2 Corpus Statistics

| Component | Tokens | % of corpus |
|---|---|---|
| DeepSeekMath Corpus (web) | ~120B | 56% of pretraining |
| AlgebraicStack (code) | ~4B | 4% of pretraining |
| arXiv papers | — | 10% of pretraining |
| GitHub code | — | 20% of pretraining |
| General natural language | — | 10% of pretraining |

> [!NOTE]
> "Domains where over 10% of web pages were collected classified as math-related" — this domain-level heuristic is critical for recall.

> [!IMPORTANT]
> ArXiv papers were included in the pretraining mix but **not** in the 120B-token web corpus. The 120B figure refers exclusively to web-crawled mathematical content.

## 3. Model Architecture and Pretraining

### 3.1 Base Model Selection

DeepSeekMath-Base 7B is initialized from **DeepSeek-Coder-Base-v1.5 7B** (not a general LLM). This is a deliberate choice: code-trained models show stronger mathematical reasoning, even without tool use.

**Training configuration:**
- Total tokens: 500B
- Mix: 56% DeepSeekMath Corpus + 4% AlgebraicStack + 10% arXiv + 20% GitHub + 10% general
- Learning rate: $4.2 \times 10^{-4}$
- Batch size: $10^7$ tokens (10M tokens per step)
- Context length: 4096 tokens

### 3.2 Key Pretraining Finding: Code vs. ArXiv

A controlled ablation on 8 mathematical benchmarks reveals:

| Pretraining data | Effect |
|---|---|
| Math web pages (DeepSeekMath Corpus) | Strong positive improvement |
| Code (GitHub, AlgebraicStack) | Positive improvement even without tool use |
| arXiv papers | Negligible or slightly negative |

> [!IMPORTANT]
> "arXiv papers seem ineffective in improving mathematical reasoning." This contradicts the intuition that formal mathematical writing should help. The authors attribute this to arXiv's distribution being too narrow and different from benchmark evaluation styles.

## 4. Supervised Fine-Tuning (SFT): DeepSeekMath-Instruct 7B

**Input:** Natural language math question $q \in \mathcal{Q}$
**Output:** Solution $o$ using Chain-of-Thought (CoT), Program-of-Thought (PoT), or tool-integrated reasoning (TIR)

**SFT data composition (776K total instruction examples):**

| Format | Description |
|---|---|
| CoT (Chain-of-Thought) | Step-by-step natural language reasoning |
| PoT (Program-of-Thought) | Python code generation for computation |
| TIR (Tool-Integrated Reasoning) | Interleaved natural language + code with execution results |

**Training configuration:**
- Steps: 500
- Batch size: 256
- Learning rate: $5 \times 10^{-5}$

## 5. Reinforcement Learning: GRPO

### 5.1 Unified RL Framework

The paper proposes a **unified framework** covering SFT, RFT, DPO, PPO, and GRPO under a single gradient formula:

$$\nabla_\theta \mathcal{J}_\mathcal{A}(\theta) = \mathbb{E}_{(q,o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_t \text{GC}_\mathcal{A}(q, o, t, \pi_{rf}) \cdot \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) \right]$$

where $\text{GC}_\mathcal{A}$ is an **algorithm-specific gradient coefficient** determining how strongly each token is reinforced. The three axes of variation:

| Axis | Options |
|---|---|
| **Data source** | Offline (SFT model samples) vs. Online (current policy $\pi_\theta$ samples) |
| **Reward function** | Rule-based (answer correctness) vs. Model-based (learned reward model) |
| **Gradient coefficient** | Varies per algorithm (see below) |

### 5.2 Group Relative Policy Optimization (GRPO)

**Difference from PPO:** PPO requires training a separate critic (value) network $V_\phi(s)$ to estimate baselines. GRPO eliminates the critic by using **within-group reward normalization**.

**Algorithm:**

```
Input: question q, old policy π_θ_old, reference policy π_ref
       group size G, clip range ε, KL coefficient β

For each training step:
  1. Sample G outputs {o_1, ..., o_G} ~ π_θ_old(· | q)
  2. Score each output: r_i = reward(q, o_i)  [rule-based correctness]
  3. Compute group baseline:
       μ = (1/G) Σ r_i,  σ = std({r_i})
  4. Normalized advantage for output i, token t:
       Â_{i,t} = (r_i - μ) / σ
  5. Maximize GRPO objective:
       J_GRPO(θ) = E[(1/G) Σ_i (1/|o_i|) Σ_t
                    min(ratio_{i,t} * Â_{i,t},
                        clip(ratio_{i,t}, 1-ε, 1+ε) * Â_{i,t})
                    - β * D_KL[π_θ || π_ref]]
     where ratio_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_θ_old(o_{i,t}|q,o_{i,<t})
```

**KL divergence estimator** (unbiased, avoids log computation instability):

$$D_{KL}[\pi_\theta \| \pi_{ref}] = \frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$$

> [!NOTE]
> This KL estimator is always non-negative (by Jensen's inequality) and gives a tighter estimate than the standard $\log(\pi_\theta / \pi_{ref})$ form used in most RLHF implementations.

### 5.3 GRPO vs. PPO vs. Other Methods

| Method | Data source | Reward | Critic model | Baseline |
|---|---|---|---|---|
| SFT | Offline (expert) | Teacher forcing | None | N/A |
| RFT | Offline (SFT samples) | Rule-based correctness | None | N/A |
| DPO | Offline (preference pairs) | Implicit (preference) | None | Reference policy |
| PPO | Online | Model-based (RM) | Yes (value network) | $V_\phi(s)$ |
| **GRPO** | **Online** | **Rule-based** | **No** | **Group mean reward** |

**Key tradeoff:** GRPO gives up a learned value function but eliminates its memory and compute overhead. With $G=64$ samples per question, group statistics provide a reliable baseline for mathematical problems where correctness is binary.

### 5.4 GRPO Training Configuration

- Base model: DeepSeekMath-Instruct 7B
- Training data: 144K questions from GSM8K + MATH
- Samples per question: $G = 64$
- Max output length: 1024 tokens
- Learning rate: $1 \times 10^{-6}$
- KL coefficient: $\beta = 0.04$

## 6. Experiments

### 6.1 Datasets

| Dataset | Language | Type | Description |
|---|---|---|---|
| GSM8K | English | Grade school math | 8,500 problems, multi-step arithmetic |
| MATH | English | Competition math | 12,500 problems, 5 difficulty levels, 7 subjects |
| SAT | English | Standardized test | College entrance math |
| OCW Courses | English | University courses | MIT OpenCourseWare problem sets |
| MMLU-STEM | English | Multi-subject | Science/tech/engineering/math subset |
| MGSM-zh | Chinese | Multilingual GSM | Chinese translation of GSM8K |
| CMATH | Chinese | Elementary math | 1,000 Chinese elementary/middle school problems |
| Gaokao-MathCloze | Chinese | College entrance | Chinese college entrance fill-in problems |
| Gaokao-MathQA | Chinese | College entrance | Chinese college entrance multiple choice |
| miniF2F | Formal | Theorem proving | Formalized math with Isabelle prover |
| HumanEval | English | Code generation | Python programming benchmark |
| MBPP | English | Code generation | Mostly basic Python problems |

### 6.2 Key Results

**DeepSeekMath-Base 7B** (no fine-tuning, 4-shot):
- GSM8K: 64.2% — surpasses Minerva 540B (64.1%) with 77× fewer parameters
- MATH: 36.2%

**DeepSeekMath-Instruct 7B** (SFT with CoT):
- GSM8K: 82.9%
- MATH: 46.8%

**DeepSeekMath-RL 7B** (GRPO):
- GSM8K: 88.2%
- MATH: 51.7% (top-1); 60.9% with 64-sample majority voting

> [!NOTE]
> "RL enhances Maj@K's performance but not Pass@K." This means RL improves consistency/reliability of correct outputs rather than discovering new solution strategies. Pass@K (whether any sample in K is correct) stays roughly the same; Maj@K (majority vote across K) improves because correct answers become more frequent.

### 6.3 Comparison with Similar Systems

| Model | Params | MATH (CoT) |
|---|---|---|
| Minerva | 540B | 33.6% |
| GPT-4 | ~1T (est.) | 42.5% |
| DeepSeekMath-Instruct | 7B | 46.8% |
| DeepSeekMath-RL | 7B | **51.7%** |
| Gemini-Ultra | ~1T (est.) | 53.2% |

## 7. Limitations

- **Geometry and proof tasks:** DeepSeekMath underperforms on geometry (visual reasoning not trained) and formal theorem proving (miniF2F with Isabelle remains weak).
- **Data selection bias:** The fastText pipeline may miss certain math subfields not well-represented in Common Crawl.
- **Pass@K unchanged by RL:** RL improves answer consistency but does not expand the model's problem-solving coverage.
- **Context length:** 4K token pretraining context is limiting for long multi-step proofs.

# Experiments

- Dataset: GSM8K (8.5K problems), MATH (12.5K problems), CMATH (1K problems), MGSM-zh, Gaokao-MathCloze, Gaokao-MathQA, SAT, OCW Courses, MMLU-STEM, miniF2F, HumanEval, MBPP
- Hardware: Not explicitly stated; estimated multi-GPU cluster for 500B token pretraining
- Optimizer: AdamW (standard for LLM training; learning rates vary by stage: $4.2 \times 10^{-4}$ for pretraining, $5 \times 10^{-5}$ for SFT, $1 \times 10^{-6}$ for GRPO)
- Results: DeepSeekMath-RL 7B achieves 51.7% on MATH and 88.2% on GSM8K, surpassing all open-source models up to 70B; competitive with closed-source models except GPT-4 and Gemini-Ultra
