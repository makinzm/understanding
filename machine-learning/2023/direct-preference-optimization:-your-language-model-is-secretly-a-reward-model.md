# Meta Information

- URL: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023.

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## Overview

Direct Preference Optimization (DPO) is a method for aligning large language models (LLMs) with human preferences without reinforcement learning. It eliminates the need for a separate reward model and the RL training loop (e.g., PPO) by reparameterizing the reward function in terms of the optimal policy, transforming preference learning into a supervised binary classification objective.

**Applicability:**
- Who: ML practitioners wanting to align pretrained LLMs to human preferences
- When: After supervised fine-tuning (SFT), during the alignment phase
- Where: Any task with pairwise human preference data (summarization, dialogue, instruction following)

## Background: RLHF Pipeline

Standard RLHF consists of three stages:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune a pretrained LM on high-quality demonstration data to obtain $\pi^{SFT}$.
2. **Reward Modeling**: Collect pairwise preference data $\mathcal{D} = \{(x, y_w, y_l)\}$ where $y_w$ is preferred over $y_l$. Train a reward model $r_\phi(x, y)$ using the Bradley-Terry model:

$$\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

3. **RL Fine-Tuning**: Optimize the language model policy $\pi_\theta$ by maximizing expected reward while constraining divergence from the reference policy $\pi_{ref}$ (usually $\pi^{SFT}$):

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y|x)} [r_\phi(x, y)] - \beta\, \mathbb{D}_{KL}[\pi_\theta(y|x) \| \pi_{ref}(y|x)]$$

where $\beta > 0$ controls the deviation from $\pi_{ref}$. Solving this via algorithms like PPO is computationally expensive, training-unstable, and requires sampling from the model during training.

## Bradley-Terry Preference Model

Human preferences are assumed to arise from a latent reward function $r^*(x, y)$. The probability that response $y_1$ is preferred over $y_2$ given prompt $x$ follows the Bradley-Terry model:

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))} = \sigma(r^*(x, y_1) - r^*(x, y_2))$$

where $\sigma$ is the sigmoid function. The partition function cancels when taking differences, which is the key property DPO exploits.

## DPO: Core Derivation

### Optimal Policy in Closed Form

The KL-constrained RL objective has a known analytical solution:

$$\pi_r(y \mid x) = \frac{1}{Z(x)}\, \pi_{ref}(y \mid x)\, \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_{ref}(y|x)\exp(r(x,y)/\beta)$ is the partition function (intractable in general).

### Reparameterizing Reward via Policy

Inverting the optimal policy equation gives the reward as a function of the policy:

$$r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry model:

$$p^*(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

Because preference probabilities depend only on reward differences, $Z(x)$ cancels exactly.

### DPO Loss (Final Objective)

By replacing $\pi^*$ with the parameterized policy $\pi_\theta$ and plugging into the negative log-likelihood:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]$$

- **Input**: Prompt $x$, preferred response $y_w$, dispreferred response $y_l$; all tokenized as sequences
- **Output**: Scalar loss; gradients update only $\pi_\theta$ (reference policy $\pi_{ref}$ is frozen)
- **No reward model**: The term $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ is an implicit reward embedded in the policy ratio

## Gradient Analysis

The gradient of $\mathcal{L}_{DPO}$ reveals an importance-weighted update mechanism:

$$\nabla_\theta \mathcal{L}_{DPO} = -\beta\, \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{importance weight}} \left[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\right] \right]$$

- **Upweights** examples where the current policy $\pi_\theta$ incorrectly assigns higher implicit reward to the dispreferred response $y_l$
- **Downweights** examples already correctly ranked (weight $\approx 0$ when $\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$)
- This provides an automatic curriculum: harder misranked examples receive more gradient signal

## Algorithm

```
Input: Dataset D = {(x, y_w, y_l)}, reference policy π_ref (frozen), hyperparameter β
Output: Policy π_θ aligned to human preferences

1. Initialize π_θ = π_ref (SFT checkpoint)
2. For each mini-batch B ⊂ D:
   a. Compute log π_θ(y_w|x) and log π_θ(y_l|x)   [forward pass through π_θ]
   b. Compute log π_ref(y_w|x) and log π_ref(y_l|x) [forward pass through π_ref, no grad]
   c. Compute implicit rewards:
      r̂_w = β * (log π_θ(y_w|x) - log π_ref(y_w|x))
      r̂_l = β * (log π_θ(y_l|x) - log π_ref(y_l|x))
   d. Compute loss: L = -mean(log σ(r̂_w - r̂_l))
   e. Backpropagate and update θ
3. Return π_θ
```

> [!NOTE]
> No sampling from the policy during training. No critic network. No reward model inference at training time. $\pi_{ref}$ is evaluated forward-only (no gradient tracked), so memory overhead is one additional forward pass per batch.

## Theoretical Results

**Theorem 1 (Expressive Power)**: The reparameterization $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$ can represent all reward functions consistent with the Plackett-Luce preference model without loss of generality. This means DPO's policy class is as expressive as the reward model class used in RLHF.

**Lemma 1**: Reward functions in the same equivalence class (differing only by a function of $x$) induce identical preference distributions.

**Lemma 2**: Equivalent reward functions yield the same optimal KL-constrained policy, confirming that the DPO objective optimizes the correct underlying preference structure.

## Comparison with Related Methods

| Method | Reward Model | RL Loop | Sampling During Training | Memory |
|---|---|---|---|---|
| PPO-RLHF | Explicit (separate) | Yes | Yes (from policy) | High |
| DPO | Implicit (in policy) | No | No | Moderate (2x forward) |
| SFT (Preferred-FT) | None | No | No | Low |
| Unlikelihood | None | No | No | Low |
| Best-of-N | Explicit (separate) | No | Yes (at inference) | High (inference) |

**vs. PPO**: PPO requires training a reward model, sampling from the current policy at each step, and a critic network. DPO replaces all of this with a static dataset and a single supervised loss.

**vs. Unlikelihood training**: Unlikelihood directly penalizes dispreferred response tokens but has no theoretical grounding in preference model optimality.

**vs. Best-of-N**: Best-of-N requires a reward model at inference and scales poorly; DPO achieves comparable or better quality without inference-time search.

> [!IMPORTANT]
> DPO implicitly learns a reward model inside the policy. The implicit reward $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ can be extracted and used to rank responses at inference time, making it interpretable.

## Experiments

### Datasets

| Task | Dataset | Details |
|---|---|---|
| Sentiment Control | IMDb | Movie review prefixes; sentiment classifier (positive/negative) used as evaluator |
| Summarization | Reddit TL;DR (Stiennon et al., 2020) | Reddit forum posts with human preference annotations; ~125k training pairs |
| Dialogue | Anthropic HH | 170k dialogue transcripts with binary human preference labels; evaluates single-turn helpfulness |

### Baselines

- **SFT**: Fine-tuned on preferred completions only
- **Preferred-FT** (= SFT): Supervised learning on $y_w$ only
- **Unlikelihood**: Maximizes $\log \pi(y_w|x)$ while minimizing $\log \pi(y_l|x)$
- **PPO-GT** (sentiment only): PPO with access to ground-truth sentiment classifier as reward
- **PPO**: Standard RLHF with learned reward model
- **Best-of-N**: Sample $N$ completions at inference, select highest-reward one

### Key Results

- **Sentiment Control**: DPO achieves better reward at equivalent KL divergence than PPO, even PPO-GT (ground truth reward). Demonstrates DPO can avoid reward over-optimization more effectively.
- **Summarization (TL;DR)**: DPO wins ~61% of pairwise comparisons against the reference SFT policy (evaluated by GPT-4 as proxy); PPO wins ~57%. DPO is also more robust to temperature variations.
- **Dialogue (HH)**: DPO is the only computationally efficient method that consistently improves over the preferred completions baseline; PPO fails to improve meaningfully in this setting.

### Hardware

Not explicitly specified, but experiments use GPT-2 (124M) for sentiment, and a 6B-parameter LM (Pythia-6B or similar) for summarization and dialogue.

### Human Evaluation Validation

The authors validated GPT-4 as an evaluation proxy by conducting human studies showing that human agreement with GPT-4 judgments is comparable to or higher than inter-human annotator agreement—justifying automated evaluation with GPT-4.

## Limitations

- Tested only up to 6B parameter models; scalability to larger models (70B+) not evaluated at publication time
- Reward over-optimization behavior under DPO's implicit reward formulation is not fully characterized
- Out-of-distribution generalization requires further study
- Assumes the preference data distribution matches the deployment distribution

> [!CAUTION]
> DPO's gradient signal depends on the quality of the static preference dataset. If $\pi_{ref}$'s generations are very different from those that human annotators compared, the implicit reward may not transfer well. This is why RLHF with online sampling can sometimes generalize better despite its complexity.
