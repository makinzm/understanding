# Meta Information

- URL: [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593.

# Introduction

This paper presents a practical framework for applying Reinforcement Learning from Human Feedback (RLHF) to language model fine-tuning when task success cannot be measured programmatically. Rather than defining a reward function by hand, the method trains a reward model from human pairwise comparisons and uses it to guide policy optimization via Proximal Policy Optimization (PPO). The framework is evaluated on stylistic text continuation (sentiment, descriptiveness) and abstractive summarization (CNN/Daily Mail, TL;DR Reddit).

## Applicability

- **Who**: NLP researchers and practitioners wanting to align language model outputs with human judgment rather than fixed heuristics.
- **When**: Tasks where "good" output is subjective, costly to specify programmatically, or where quality is best measured by direct human evaluation.
- **Where**: Any autoregressive language model that can be fine-tuned; the paper uses GPT-2 774M as the base model.

> [!NOTE]
> The paper explicitly targets settings where "the human is the most natural way to specify the task." This distinguishes RLHF from conventional supervised fine-tuning, which requires a large labeled dataset of (input, desired output) pairs.

# Method Overview

The pipeline has three stages: supervised pre-fine-tuning (optional), reward model training from human comparisons, and RL fine-tuning of the policy.

## Stage 1: Supervised Pre-Fine-Tuning (SFT)

For summarization, the policy $\pi_\theta$ is first fine-tuned on a supervised dataset of (article, summary) pairs using standard language modeling cross-entropy loss. This gives the model a reasonable starting distribution before RL begins.

For stylistic continuation tasks, the base pretrained GPT-2 model is used directly without a supervised stage.

## Stage 2: Reward Model Training

### Human Comparison Procedure

Human labelers are shown a context $x$ (a book passage or article) and four model-generated continuations $\{y_1, y_2, y_3, y_4\}$. They select the best continuation $y_b$ according to the task criterion (e.g., most positive sentiment, most descriptive).

### Training Objective

The reward model $r_\phi : (x, y) \to \mathbb{R}$ is trained with a softmax cross-entropy loss that treats the four-way comparison as a classification problem:

```math
\begin{align}
  \mathcal{L}_\text{reward} = -\mathbb{E}_{(x, y_1, y_2, y_3, y_4, b) \sim \mathcal{D}} \left[ \log \frac{e^{r_\phi(x, y_b)}}{\sum_{i=1}^{4} e^{r_\phi(x, y_i)}} \right]
\end{align}
```

where $b \in \{1,2,3,4\}$ is the human-chosen best option and $\mathcal{D}$ is the collected comparison dataset.

### Architecture

The reward model initializes from the same pretrained GPT-2 774M weights as the policy. A randomly initialized linear projection maps the final token's hidden state $h \in \mathbb{R}^{1280}$ to a scalar reward $r \in \mathbb{R}$.

> [!IMPORTANT]
> The reward model and policy are kept as **separate copies** throughout training. Experiments with a shared backbone (motivated by auxiliary task benefits) failed because of the severe data imbalance: ~60K reward comparison samples vs. ~2M RL policy episodes caused the shared parameters to overfit on the smaller reward objective.

## Stage 3: RL Policy Fine-Tuning

### Objective

The policy $\pi_\theta$ is fine-tuned to maximize the expected reward while staying close to the original model $\rho$ (the SFT model or base GPT-2). The effective per-sample reward is:

```math
\begin{align}
  R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\rho(y \mid x)}
\end{align}
```

where:
- $r_\phi(x, y)$ is the learned scalar reward,
- $\beta \log \frac{\pi_\theta(y \mid x)}{\rho(y \mid x)}$ is a per-token KL penalty that penalizes divergence from the reference model $\rho$,
- $\beta$ is a coefficient dynamically adjusted to keep KL divergence in a target range (6–10 nats) using a proportional controller with gain $K_\beta = 0.1$.

### Algorithm

The policy is trained with PPO-2 (clipped surrogate objective):

**Input**: pretrained reference policy $\rho$, reward model $r_\phi$, dataset of contexts $\mathcal{X}$

1. Sample a batch of contexts $\{x_i\}$ from $\mathcal{X}$.
2. For each $x_i$, sample a continuation $y_i \sim \pi_\theta(\cdot \mid x_i)$.
3. Compute token-level log-probability ratio $\log \pi_\theta(y_i \mid x_i) - \log \rho(y_i \mid x_i)$.
4. Compute effective reward $R(x_i, y_i) = r_\phi(x_i, y_i) - \beta \cdot \text{KL}(x_i, y_i)$.
5. Compute PPO clipped surrogate loss and update $\theta$.
6. Update $\beta$ using proportional controller toward target KL.

**Output**: fine-tuned policy $\pi_\theta$ whose samples are preferred by the reward model while remaining near $\rho$.

### Input / Output Dimensions

| Component | Input | Output |
|-----------|-------|--------|
| Policy $\pi_\theta$ | Context $x \in \mathbb{N}^{L_x}$ (token IDs) | Continuation $y \in \mathbb{N}^{L_y}$, per-token log-probs $\in \mathbb{R}^{L_y \times V}$ |
| Reward model $r_\phi$ | $(x, y)$ concatenated token sequence | Scalar $r \in \mathbb{R}$ |
| KL penalty | Per-token log-probs from $\pi_\theta$ and $\rho$ | Scalar $\sum_t \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\rho(y_t \mid x, y_{<t})}$ |

where $V$ is vocabulary size, $L_x$ and $L_y$ are context and continuation lengths respectively.

## Online vs. Offline Data Collection

| Mode | Description | Trade-off |
|------|-------------|-----------|
| **Offline** | Collect all comparisons using the base model; reward model fixed during RL | Simpler to implement; comparisons may not cover the improving policy's distribution |
| **Online** | Continuously query the current policy for new comparisons during RL training | Better distribution coverage (+3 ROUGE on summarization at 60K labels); introduces software complexity, debugging difficulty, and data quality maintenance overhead |
| **Batched (recommended)** | Pause RL, collect a batch of comparisons on the current policy, retrain reward model, resume | Middle ground balancing coverage and implementation simplicity |

The paper recommends batched collection for future work given the challenges encountered with fully online training.

# Model Architecture

The backbone for both the policy and the reward model is **GPT-2 774M**:

| Property | Value |
|----------|-------|
| Parameters | 774M |
| Transformer layers | 36 |
| Attention heads | 20 |
| Embedding dimension | 1280 |
| Feed-forward hidden dim | 5120 |

The reward model adds a linear projection $W \in \mathbb{R}^{1280 \times 1}$ initialized randomly on top of the final-layer hidden state of the last token.

# Tasks

## Stylistic Text Continuation

**Context**: 32–64 token excerpts from BookCorpus.  
**Continuation**: 24 tokens generated by the policy.  
**Task variants**:
- **Sentiment**: Continue the passage in a positive, happy tone.
- **Descriptiveness**: Continue the passage with vivid physical detail.

Labelers compare four continuations and select the best according to the stylistic criterion.

## Summarization

**Context**: News article or Reddit post, truncated to 500 tokens.  
**Summary**: Up to 75 tokens generated by the policy.  
**Datasets**:
- **CNN/Daily Mail**: News articles with human-written highlights; 13,368 test articles.
- **TL;DR (Reddit)**: Long Reddit posts with user-written "too long; didn't read" summaries; 30,000 held-out validation posts excluded from training.

Labelers compare two summaries and select the one they prefer overall.

# Experiments

## Datasets

| Dataset | Task | Size Used |
|---------|------|-----------|
| BookCorpus | Stylistic continuation (context source) | Not specified |
| CNN/Daily Mail | Summarization | 13,368 test articles |
| TL;DR (Reddit) | Summarization | 30,000 validation posts |

## Hardware and Optimizer

| Setting | Value |
|---------|-------|
| Reward model optimizer | Adam |
| Reward model batch size | 8–32 |
| Reward model learning rate | $\approx 1.77 \times 10^{-5}$ |
| Reward model training epochs | 1 |
| RL optimizer | PPO-2 |
| RL episodes | 2M |
| RL batch size | 512–1024 |
| RL learning rate | $7.07 \times 10^{-6}$ to $1.41 \times 10^{-5}$ |
| KL target range | 6–10 nats |

## Key Results

### Stylistic Continuation

- With ~5,000 human comparisons, the RL-fine-tuned model was **preferred 86% of the time** over the zero-shot GPT-2 baseline.
- The RL model trained on real human feedback was **preferred 77% of the time** over a model trained using a mock reward (a sentiment classifier), demonstrating that human feedback captures nuances beyond simple classifier rewards.
- Inter-labeler agreement on the sentiment task was 38–46% (vs. 25% chance for 4-way); authors themselves agreed with labelers 60–62% of the time, confirming that stylistic tasks are inherently subjective.

### Summarization

- Pure RL-fine-tuned models developed **extractive "smart copying" behavior**: on TL;DR they reproduced 71% of original sentences verbatim; on CNN/Daily Mail, 98%.
- Despite low ROUGE scores relative to supervised baselines, human evaluators **preferred RL-fine-tuned summaries over reference summaries** 96% of the time on TL;DR and 84% on CNN/Daily Mail.
- A **hybrid approach** (supervised pre-fine-tuning + RL) achieved the best ROUGE scores.
- Factual accuracy of extractive RL models: 90–95%; abstractive supervised models: ~70%, indicating a quality-abstraction trade-off.

> [!IMPORTANT]
> The 96% human preference rate for RL summaries over reference summaries likely reflects that labelers rewarded extractive accuracy (easily verifiable by scanning) rather than abstractive quality. This is a task design issue rather than a true measure of summarization quality.

# Challenges and Failure Modes

## Extractive Behavior

The RL policy learned to copy source sentences rather than paraphrase. This emerged because:
1. Extraction guarantees factual accuracy, which human raters rewarded.
2. Raters could quickly detect inaccuracy in original text but not in paraphrases.
3. The training objective penalized inaccuracy without explicitly penalizing extraction.

## Reward Hacking via Sign Flip

A bug that **negated the reward** while preserving the KL penalty caused the policy to optimize for the worst possible content (explicit sexual material in some experiments). This highlighted the need for automated monitoring to detect degenerate reward-maximizing solutions.

## Parameter Sharing Failure

Sharing parameters between the reward model and policy was tested as an auxiliary task strategy. It failed because the reward comparison data (~60K samples) was orders of magnitude smaller than the RL rollout data (~2M episodes), causing the shared parameters to overfit on the reward objective.

## Labeling Ambiguity

Open-ended quality judgments (e.g., "which summary is better overall?") produced noisier labels than targeted factual questions. The authors recommend tasks where labelers are asked to identify specific problems rather than make holistic quality comparisons.

# Comparison with Related Methods

| Method | Reward Source | Fine-tuning Algorithm | Key Difference |
|--------|--------------|----------------------|----------------|
| **This paper (RLHF)** | Human pairwise comparisons → learned reward model | PPO with KL penalty | Human preferences define the reward; reward model generalizes to new inputs |
| Supervised Fine-tuning (SFT) | Labeled (input, output) pairs | Cross-entropy MLE | No RL; optimizes likelihood of references, not human preference |
| Reward from classifier | Fixed pretrained classifier (e.g., sentiment model) | PPO | Reward is not learned from direct human feedback; misses human nuance |
| InstructGPT / RLHF (2022+) | Human comparisons with instruction following | PPO | Larger scale (175B); adds instruction following SFT before RL; broader task coverage |

> [!TIP]
> This paper is a direct predecessor to InstructGPT (Ouyang et al., 2022) and the broader RLHF paradigm used in ChatGPT. The three-stage pipeline (SFT → reward model → PPO) introduced here is essentially the same pipeline used in those later systems at much larger scale.

# Experiments Summary

- **Metric for stylistic tasks**: Human preference rate over baseline (pairwise evaluation).
- **Metric for summarization**: ROUGE-{1,2,L,L-sum} plus human preference rate over reference summaries.
- **Label budget studied**: 5K–60K human comparisons; online collection showed 3-point ROUGE improvement at 60K vs. offline.
- **Code released**: Reward modeling and offline fine-tuning code for GPT-2 124M (smaller version only).
