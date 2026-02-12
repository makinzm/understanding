# Meta Information

- URL: [LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](https://arxiv.org/abs/2602.09924)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lugoloobi, W., Foster, T., Bankes, W., & Russell, C. (2026). LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations. arXiv:2602.09924.

# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

## Overview

This paper investigates whether large language models internally encode information about the likelihood of success on a given problem *before* generating any output. The central hypothesis is that the residual stream activations at the end of the instruction prompt carry a latent signal about problem difficulty that is distinct from surface-level features (e.g., question length or TF-IDF similarity).

The authors demonstrate that:

1. **Linear probes trained on pre-generation hidden states** can predict task success far better than surface-level baselines, with AUROC reaching 0.78–0.91 depending on the model and domain.
2. **Human-perceived difficulty and model-specific difficulty are distinct signals**: probes achieve Spearman ρ = 0.83–0.87 for predicting human IRT difficulty, but only ρ = 0.40–0.64 for predicting model-specific expected success rate — confirming these are not the same quantity.
3. **Extended reasoning (chain-of-thought) degrades probe quality** even as it improves task accuracy: AUROC drops from 0.78 to 0.64 for GPT-OSS-20B under budget-forcing reasoning, suggesting the pre-generation representation becomes less informative once the model relies on explicit reasoning.
4. **Probe-based routing** across a pool of five models matches the best single model while reducing inference cost by up to 70% on MATH benchmarks.

This work is applicable to practitioners deploying multiple LLMs who want to dynamically route queries to the cheapest capable model without sacrificing accuracy.

## Problem Formulation

Given a question $x$ and a language model $M$ with hidden dimension $D$, let $h(x) \in \mathbb{R}^D$ be the residual stream activation at the final token position of the instruction prompt (i.e., after the chat template, before the model generates). The goal is to learn a probe $f: \mathbb{R}^D \to \mathbb{R}$ such that $f(h(x))$ predicts one of three targets:

| Target | Type | Description |
|--------|------|-------------|
| Human IRT difficulty | Continuous | Item response theory difficulty parameter estimated from human performance data |
| Expected success rate | Continuous | $\mathbb{E}[\text{correct}(x)]$ estimated from $K=50$ rollouts at temperature $T=1$ |
| Binary success | Binary | Whether greedy decoding or Maj@K produces the correct answer |

## Probe Architecture and Training

### Activation Extraction

Pre-generation activations are extracted from the **residual stream** (before layer normalization) at the position of the **last instruction token**, identified by parsing the chat template. For a model with $L$ layers, this produces $L$ candidate vectors, each of shape $D$. The authors probe each layer independently and select the best via validation performance.

> [!NOTE]
> "We extract representations at post-instruction template positions, capturing what the model 'knows' before it begins to write."

Input to probe: $h_\ell(x) \in \mathbb{R}^D$ for layer $\ell \in \{1, \ldots, L\}$

### Probe Models

**Continuous target (success rate / IRT difficulty)**:

$$\hat{y} = w^\top h_\ell(x), \quad w \in \mathbb{R}^D$$

Trained with Ridge regression (MSE loss with $\ell_2$ regularization):

$$\mathcal{L} = \sum_i (y_i - w^\top h_\ell(x_i))^2 + \alpha \|w\|^2$$

**Binary target (success/failure)**:

$$\hat{p} = \sigma(w^\top h_\ell(x)), \quad w \in \mathbb{R}^D$$

Trained with $\ell_2$-regularized logistic regression (BCE loss). No bias term is used in either case.

**Regularization** strength $\alpha \in \{10^{-3}, 10^{-2}, \ldots, 10^4\}$ selected via grid search on the validation split.

> [!IMPORTANT]
> The absence of a bias term and use of a single linear layer makes these probes interpretable as linear classifiers in the activation space — a deliberate design choice to test whether success information is *linearly* decodable rather than relying on nonlinear transformations.

### Calibration

After training, Platt scaling is applied to binary probe outputs using validation data to convert raw logits to calibrated probability estimates. Expected calibration error (ECE) is reported before and after scaling.

## Datasets

| Dataset | Domain | Size | Notes |
|---------|--------|------|-------|
| E2H-AMC | Math (AMC problems) | 4,000 problems | Contains human IRT labels + model rollout data (K=50 rollouts per problem per model); used as primary training source |
| MATH | Math (competition) | Standard splits | Used for routing evaluation |
| GSM8K | Math (grade school) | Standard splits | Generalization test |
| AIME 1983–2024 | Math (Olympiad) | ~400 problems | Held-out evaluation |
| AIME 2025 | Math (Olympiad) | Recent competition problems | Routing evaluation |
| LiveCodeBench | Coding | Standard splits | Domain transfer test (math probes → code) |

Data split: 80% train, 20% validation for hyperparameter selection; final test evaluation on held-out sets.

> [!NOTE]
> E2H-AMC is a novel dataset constructed by the authors containing parallel human psychometric data and model performance data on identical AMC problems, enabling direct comparison of human vs. model difficulty perceptions.

## Key Experiments and Results

### Experiment 1: Predicting Human IRT Difficulty

Probes trained to predict human IRT difficulty scores achieve:
- Spearman rank correlation ρ = 0.83–0.87 across models
- This substantially outperforms TF-IDF and question-length baselines

This shows that model activations encode a signal correlated with human difficulty perception — but the authors flag this as an *indirect* measure: the probe is measuring what the model thinks, not what humans experience.

### Experiment 2: Predicting Model-Specific Success Rate

Probes predicting each model's own expected success rate (averaged over K=50 rollouts) achieve:
- Spearman ρ = 0.40–0.64 across models

The gap between ρ~0.85 for human IRT and ρ~0.52 for model-specific difficulty confirms that **a model's internal difficulty representation diverges from human difficulty judgments**, particularly for problems where reasoning effort changes the outcome.

### Experiment 3: Binary Success Prediction (AUROC)

| Model | Domain | Probe AUROC | TF-IDF baseline |
|-------|--------|-------------|-----------------|
| GPT-OSS-20B | MATH | 0.78 | ~0.65 |
| Qwen2.5-Coder | LiveCodeBench | 0.81–0.91 | lower |
| DeepSeek-R1 | LiveCodeBench | 0.85–0.91 | lower |

### Experiment 4: Effect of Reasoning (Chain-of-Thought)

For GPT-OSS-20B under budget-forcing (extended reasoning):
- AUROC drops from **0.78 → 0.64** as reasoning budget increases
- Yet task accuracy *improves* with more reasoning

This anti-correlation suggests that when models engage in extended reasoning, the pre-generation activation becomes a less reliable predictor of outcome — the model's fate is increasingly determined by the quality of its reasoning process rather than its initial latent state.

> [!CAUTION]
> This result implies a potential limitation: probe-based routing may be less reliable for reasoning-heavy models (e.g., o1, DeepSeek-R1 in chain-of-thought mode). The authors acknowledge this but note probes still outperform surface baselines even in this setting.

## Routing Algorithms

### Cascade Routing (Two-Model)

A threshold $\tau$ on the predicted success probability $\hat{p}_s(x)$ routes between a small cheap model $M_s$ and a large expensive model $M_l$:

$$\hat{M}(x) = \begin{cases} M_s & \text{if } \hat{p}_s(x) \geq \tau \\ M_l & \text{if } \hat{p}_s(x) < \tau \end{cases}$$

By sweeping $\tau \in [0, 1]$, a Pareto frontier of accuracy vs. cost tradeoffs is traced. At $\tau = 0.6$ on MATH, accuracy = 91.2% with 17% cost reduction vs. always using $M_l$.

### Utility-Based Routing (Multi-Model Pool)

For a pool of $N$ models, the selected model maximizes expected utility:

$$\hat{M}(x) = \arg\max_{i \in \{1,\ldots,N\}} \left(\hat{p}_i(x) - \lambda \hat{c}_i\right)$$

where $\hat{p}_i(x)$ is the probe's predicted success probability for model $i$, $\hat{c}_i$ is the normalized deployment cost (tokens × price), and $\lambda$ controls the accuracy-cost tradeoff.

With 5 models on MATH: accuracy = 92%, cost reduction = **70%** vs. always using the highest-performing model.

> [!TIP]
> The utility function is closely related to the classical explore-exploit tradeoff. See [Madaan et al., 2023 (FrugalGPT)](https://arxiv.org/abs/2305.05176) for related work on LLM routing.

## Comparison with Related Methods

| Method | Feature Type | Requires Rollouts | Routing Granularity |
|--------|-------------|-------------------|---------------------|
| **This work** | Internal activations (linear probe) | Yes (for training) | Per-query |
| FrugalGPT | Surface features + prior accuracy | No | Per-query |
| LLM-Blender | Pairwise ranking (post-generation) | No | Per-query |
| FLAP | Confidence scores (post-generation) | No | Per-query |
| Human IRT proxies | Question text features | No | Per-question-set |

Key differentiator: this work operates **entirely pre-generation**, requiring no partial output or generation-time signals. Probes are model-specific (trained on each model's rollout data) rather than model-agnostic.

## Limitations

- Probes are trained and evaluated on a single token position (end of instruction); multi-position probing is not explored.
- The approach requires $K=50$ rollouts per training problem to estimate success rates — substantial upfront compute.
- Routing policies use simple threshold/utility functions; learned routing policies (e.g., meta-learners) are not compared.
- Cross-dataset transfer (e.g., probes trained on E2H-AMC, tested zero-shot on MATH) is not evaluated.
- Extended reasoning (CoT) degrades probe quality, limiting applicability to frontier reasoning models.

# Experiments

- Dataset: E2H-AMC (4,000 AMC problems with human IRT labels), MATH, GSM8K, AIME 1983–2024, AIME 2025, LiveCodeBench
- Hardware: Not specified
- Optimizer: Ridge regression and $\ell_2$-regularized logistic regression for probes; $\alpha \in \{10^{-3}, \ldots, 10^4\}$ via grid search
- Results: Linear probes achieve AUROC 0.78–0.91 for binary success prediction; utility-based routing over 5 models achieves 92% accuracy with 70% cost reduction on MATH vs. always using the best model
