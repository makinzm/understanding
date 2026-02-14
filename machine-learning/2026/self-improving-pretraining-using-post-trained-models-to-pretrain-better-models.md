# Meta Information

- URL: [Self-Improving Pretraining: using post-trained models to pretrain better models](https://arxiv.org/abs/2601.21343)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Tan, E. X., Dhuliawala, S., Xu, J., Yu, P., Sukhbaatar, S., Weston, J., & Golovneva, O. (2026). Self-Improving Pretraining: using post-trained models to pretrain better models. FAIR at Meta.

# Self-Improving Pretraining

## Overview

Self-Improving Pretraining (SIP) is a method that augments standard language model pretraining with reinforcement learning, using a strong post-trained model as a judge and rewriter. Rather than training only on static next-token prediction, SIP treats each prefix–suffix pair as a decision problem: the policy model generates candidate suffixes, the judge scores them, and the highest-scoring candidate supervises the update. This enables the pretraining phase itself to optimize for quality, safety, and factuality — properties that standard pretraining on raw corpora cannot directly target.

**Applicability:** SIP is most valuable when (1) high-quality post-trained models are available to act as judges, (2) the raw pretraining corpus contains unsafe or low-quality text that degrades model behavior, and (3) compute budget is sufficient to support multiple model roles (judge, rewriter, policy).

## 1. Introduction

Standard LLM pretraining minimizes cross-entropy loss over token sequences from raw web data. This approach learns distributional properties of text but does not directly optimize for downstream quality, safety, or factual accuracy — qualities typically introduced only during fine-tuning or RLHF. SIP proposes to bring these RL-based improvements into the pretraining stage, using the insight that post-trained models encode human-preference knowledge that can be distilled back into pretraining.

> [!IMPORTANT]
> SIP is designed specifically for large-scale pretraining or continual pretraining scenarios where changing the data distribution from unsafe/low-quality to high-quality outputs is the primary goal.

## 2. Self-Improving Pretraining

### 2.1 The Sequence Pretraining Task: Prefix-Conditioned Suffix Generation

Standard next-token prediction is reformulated as **prefix-conditioned suffix generation**. Given a document tokenized into positions $1, \ldots, T$, training is divided into steps indexed by $j$. At each step:

- **Prefix**: $x_{1}, x_{2}, \ldots, x_{j-1}$ (all tokens before position $j$)
- **Suffix**: $x_j$ — a contiguous block of $N = 128$ tokens starting at position $j$

The policy model $\pi$ generates a suffix candidate:
$$\bar{x}_j \sim \pi(\cdot \mid x_{1}, \ldots, x_{j-1})$$

where $\bar{x}_j \in \mathbb{R}^{N}$ is a sequence of $N$ discrete token ids. This framing is equivalent to standard autoregressive LM training but makes explicit the "generation unit" that is evaluated and rewarded.

### 2.2 Self-Improving Pretraining Using Post-Trained Models

Three components are combined to implement SIP:

#### Suffix Rewriter

A post-trained model rewrites the original suffix $x_j$ into an improved version $\tilde{x}_j$ that maintains coherence with the prefix while improving quality, safety, or factuality. The rewriter is trained on supervised pairs (original suffix → improved suffix) and is frozen during policy training.

#### Suffix Judge

The judge $J$ is a fine-tuned or prompted model that evaluates any candidate suffix for:
- **Quality** $J_{\text{qual}}(\bar{x}_j, x_j \mid x_{1,\ldots,j-1})$: pairwise comparison between candidate and original
- **Safety** $J_{\text{safe}}(\bar{x}_j)$: binary safety classification
- **Factuality** $J_{\text{fact}}(\bar{x}_j \mid x_{1,\ldots,j-1})$: factual correctness score

The reward for a rewritten suffix from a safe context is:
$$R_{\text{safe}} = \begin{cases} 1.0 & \text{if } \bar{x}_j = x_j \text{ (original kept)} \\ 0.0 & \text{otherwise} \end{cases}$$

For rewrites of unsafe suffixes:
$$R_{\text{unsafe}} = \frac{1}{2}\bigl(J_{\text{qual}}(\bar{x}_j, x_j \mid x_{1,\ldots,j-1}) + J_{\text{safe}}(\bar{x}_j)\bigr)$$

#### Policy Training

The policy model $\pi$ is trained using **online DPO** (Direct Preference Optimization) with the judge's scores as rewards. At each training step, $K$ rollouts are sampled from $\pi$ alongside the original and rewritten suffixes. All candidates are scored by the judge, and the highest-scoring candidate is used as the positive example.

**Training curriculum:**
- Early training: prefer original and rewritten suffixes over rollouts (rollouts have high variance early in training)
- Late training: reward high-quality rollouts when they surpass the rewrite quality

The DPO loss is applied over pairs $(y^+, y^-)$ where $y^+$ is the chosen (highest-reward) candidate and $y^-$ is the rejected candidate:
$$\mathcal{L}_{\text{DPO}}(\pi; \pi_{\text{ref}}) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \beta \log \frac{\pi(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)}\right)\right]$$

where $\pi_{\text{ref}}$ is the reference model (periodically updated EMA of the policy) and $\beta$ is the KL-divergence coefficient.

> [!NOTE]
> An alternative training objective **RF-NLL** (Reinforcement learning with Negative Log-Likelihood) supervises the policy model only on the highest-scoring suffix via standard NLL loss, without a reference model or pairwise comparison. This is simpler but less stable.

**Comparison with related methods:**

| Method | Training Stage | Reward Signal | Policy Update |
|---|---|---|---|
| Standard Pretraining | Pretrain | None (MLE) | NLL on all tokens |
| RLHF / PPO | Post-train | Human preference | PPO gradient |
| DPO | Post-train | Human preference | Pairwise NLL |
| **SIP (this work)** | Pretrain | Judge model | Online DPO / RF-NLL |
| KTO / ORPO | Post-train | Binary preference | Modified NLL |

## 3. Experiments

### 3.1 Models and Data

- **Policy model**: Llama 1.4B (trained from scratch or continued from Llama base)
- **Judge model**: Fine-tuned Llama3.1-8B (trained via GRPO) or prompted GPT-OSS-120B
- **Rewriter model**: Post-trained Llama 3.x (fine-tuned on (prefix, original, rewrite) triples)

**Judge training hyperparameters:**
- Algorithm: GRPO (Group Relative Policy Optimization)
- Global batch size: 256, 16 generations per prompt
- Temperature: 0.6, top-p: 0.6
- Training steps: 500 on 64 GPUs
- Learning rate: $2.0 \times 10^{-7}$

**Policy training hyperparameters (continual):**
- Batch size: 256, 16 rollouts per prompt
- Temperature: 1.0, top-p: 1.0
- Training steps: 2000 on 64 GPUs
- Learning rate: $5.0 \times 10^{-6}$ with cosine schedule

### Datasets

| Dataset | Role | Size |
|---|---|---|
| SlimPajama (SP) | Main pretraining corpus | 983,520 training samples |
| RedPajama (RP) | Safety experiment corpus | 257,154 unsafe-filtered samples |
| Judge training (quality) | Fine-tune judge | 75,432 train / 4,096 val |
| Judge training (safety) | Fine-tune judge | 3,192 train / 512 val |
| Rewriter training | Supervised rewrite pairs | 73,080 samples |
| Evaluation (SP) | Safe prefix test set | 1,000 samples |
| Evaluation (RP) | Unsafe prefix test set | 1,000 samples |

> [!NOTE]
> SlimPajama is a higher-quality derivative of RedPajama with aggressive deduplication and quality filtering. RedPajama is used specifically for safety experiments because its noisier data contains a meaningful fraction of unsafe prefixes.

### 3.2 Experimental Setup

**Continual pretraining**: The Llama 1.4B base model is continued on SP/RP data using SIP, with different optimization targets (quality, safety, factuality). This tests whether SIP can shift a pretrained model's distribution toward higher-quality outputs.

**From-scratch training**: A new Llama 1.4B is trained from scratch on SP data with SIP applied throughout pretraining.

### 3.3 Evaluations

- **Generation quality win rate**: GPT-OSS-120B pairwise judge comparing SIP outputs vs. baseline on held-out prefixes
- **Factuality**: FactScore-style retrieval-augmented factuality metric; baseline 42.3
- **Safety**: Binary classifier score over completions on unsafe prefixes; baseline 76.9
- **Standard benchmarks**: BoolQA, PIQA, HellaSwag, ARC-easy, ARC-challenge, OpenBookQA, SIQA, MMLU

### 3.4 Results

**Continual pretraining (quality optimization):**
- Generation quality win rate: **86.3%** (vs. 50% for the base Llama)
- Coherence win rate: **87.9%**
- Standard eval average: **50.1**

**Continual pretraining (factuality optimization):**
- Generation quality win rate: **84.0%**
- Factuality score: **57.6** (vs. 42.3 baseline, +36.2% relative)

**Continual pretraining (safety optimization):**
- Generation quality win rate on unsafe prefixes: **77.7%**
- Safety score: **91.1** (vs. 76.9 baseline, +18.5% relative)

**From-scratch training:**
- Generation quality win rate: **32.4%** (vs. 1.3% for standard pretraining baseline)
- Safety score: **97.5** (vs. 85.2 baseline)

> [!IMPORTANT]
> The from-scratch win rate of 32.4% being much lower than continual pretraining (86.3%) reflects the difficulty of applying RL from random initialization — early in training, rollouts are too noisy for the judge to provide stable reward signals.

### 3.5 Analysis and Ablations

**Training objective comparison (Table 6):**

| Objective | Std-prefix quality | Unsafe-prefix quality | Safety score |
|---|---|---|---|
| SFT on rewrite | 52.7 | 50.6 | — |
| Online DPO (rewrite vs 1 rollout) | 60.2 | 87.2 | — |
| Online DPO (suffix vs 16 rollouts) | 73.6 | 77.7 | 91.1 |

Online DPO with 16 rollouts consistently outperforms simpler objectives, confirming the importance of diverse candidate generation.

**Judge model comparison (Table 7):**

| Judge | Quality win rate |
|---|---|
| Fine-tuned Llama3.1-8B (GRPO) | 72.1 |
| Prompted GPT-OSS-120B | 84.3 |

A stronger judge leads to better policy training outcomes.

## 4. Related Work

- **Safety in pretraining**: Korbak et al. (2023) use control tokens during pretraining to separate safe/unsafe distributions; Min et al. (2023) rewrite with special tokens. SIP replaces control tokens with explicit RL rewards.
- **Factuality**: Tian et al. (2023), Lin et al. (2024), Zhang et al. (2024b) apply supervised fine-tuning or offline RL for factuality — SIP applies these ideas online during pretraining.
- **Reasoning and RL in pretraining**: DeepSeek-AI (2025) applies RLVR during post-training for reasoning; Wang et al. (2025) and Dong et al. (2025) integrate RL into mid-training. SIP extends this to the pretraining phase.
- **Foundational LM work**: Bengio et al. (2003) introduced neural language models; Brown et al. (2020) demonstrated scaling properties of GPT-style models.

## 5. Conclusion and Discussion

SIP demonstrates that reinforcement learning can be integrated into pretraining to produce models that are safer, more factual, and higher quality — without additional post-training. The method generalizes naturally to multiple simultaneous objectives by summing reward signals from multiple judges.

**Limitations and trade-offs:**
- Requires maintaining three separate models (judge, rewriter, policy) simultaneously, increasing compute cost.
- From-scratch RL-based pretraining is unstable early in training when rollouts are too noisy.
- Fine-grained safety control via reward tokens may be preferable to removing capabilities entirely from the pretraining distribution.
- As internet-scale data becomes saturated, incentive-based training on synthetic or rewritten data may become increasingly necessary.

# Experiments

- Dataset: SlimPajama (983,520 training samples), RedPajama (257,154 safety-filtered samples), judge training data (75,432 quality / 3,192 safety samples), rewriter training data (73,080 samples)
- Hardware: 64 GPUs (judge training and policy training)
- Optimizer: Online DPO (policy); GRPO (judge)
- Results: 86.3% quality win rate, 36.2% relative factuality improvement (57.6 vs. 42.3), 18.5% relative safety improvement (91.1 vs. 76.9) in continual pretraining; 32.4% quality win rate and 97.5 safety score in from-scratch training
