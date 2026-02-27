# Meta Information

- URL: [LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](https://arxiv.org/abs/2602.09924)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / [arXiv.org Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lugoloobi, W., Foster, T., Bankes, W., & Russell, C. (2026). LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations. arXiv:2602.09924.

---

# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

## Overview

This paper investigates whether large language models (LLMs) encode signals about their own likelihood of success within their internal representations **before** generating any output. The central claim is that pre-generation activations—residual-stream states immediately after processing the input prompt—contain predictive information about whether the model will correctly answer a math or coding problem.

**Applicability:** This work is relevant to any ML practitioner deploying LLMs in high-volume or cost-sensitive settings (e.g., math tutoring systems, code assistants), where routing queries to the cheapest sufficient model or flagging uncertain responses for review can significantly reduce inference costs.

---

## Problem Statement

Running LLMs with extended chain-of-thought (CoT) reasoning on every query is expensive. Determining which inputs genuinely require additional compute is non-trivial.

Existing heuristics such as question length or TF-IDF keyword features are surface-level and do not reflect the model's internal assessment of problem difficulty. This paper proposes using **linear probes** trained on pre-generation activations to directly predict per-instance success, enabling informed routing before any tokens are generated.

---

## Key Distinction: Human Difficulty vs. Model Difficulty

Using the **E2H-AMC** dataset (4,000 American Mathematics Competition problems, each annotated with both IRT-derived human difficulty scores and empirical LLM solve rates), the authors demonstrate that two distinct signals coexist in the pre-generation activations:

| Signal | Spearman ρ | Description |
|---|---|---|
| Human-perceived difficulty | 0.83–0.87 | How hard humans find the problem |
| Model-specific difficulty | 0.40–0.64 | How likely the model is to fail |

> [!IMPORTANT]
> Model difficulty is a model-specific concept: a problem that humans find easy may be hard for a particular LLM, and vice versa. As model capability scales, this divergence intensifies—human difficulty labels increasingly mischaracterize what the model finds challenging.

---

## Method: Linear Probes on Pre-Generation Activations

### Input / Output

- **Input:** Residual-stream activations at the post-instruction token position, $h \in \mathbb{R}^{d}$, where $d$ is the hidden dimension of the LLM.
- **Output (continuous):** Predicted success rate $\hat{p} \in [0, 1]$ (how often the model solves the problem across samples).
- **Output (binary):** Predicted success label $\hat{y} \in \{0, 1\}$ for a fixed decoding policy (e.g., majority voting over $k$ samples).

### Probe Architecture

Linear probes are intentionally simple to isolate what is linearly decodable from the model's internal state:

```
Algorithm: Linear Probe Training
Input: dataset D = {(h_i, y_i)} where h_i ∈ ℝ^d and y_i ∈ {0,1} or ℝ

For continuous targets (success rate):
  - Fit Ridge regression: min_w ||Hw - y||² + λ||w||²
  - Tune λ via grid search on validation set

For binary targets (per-instance success):
  - Fit Logistic regression: min_w Σ log(1 + exp(-y_i * w^T h_i)) + λ||w||²
  - Tune λ via grid search on validation set

Inference:
  - Given new prompt, extract h from post-instruction residual stream
  - Compute ŷ = w^T h (continuous) or σ(w^T h) (binary)
  - Use ŷ as routing/flagging signal (no generation required)
```

> [!NOTE]
> Probes are trained on activations extracted **before** any output token is generated. This means the routing decision incurs only a single forward pass through the prompt, not a full generation.

### Baseline Comparisons

Text-only baselines (TF-IDF features, question character length) substantially underperform activation-based probes, with the largest gap on model-difficulty prediction. This confirms that the predictive signal is genuinely encoded in the model's internal state, not derivable from surface-level question features.

---

## Key Finding: Extended Reasoning Degrades Linear Probe Quality

A counterintuitive result: as reasoning budgets increase (more thinking tokens), binary probe AUROC paradoxically **decreases** despite the model achieving higher accuracy.

| Reasoning Budget | Model Accuracy | Probe AUROC (Spearman ρ) |
|---|---|---|
| Low reasoning (GPT-OSS-20B) | Lower | ρ = 0.58 |
| High reasoning (GPT-OSS-20B) | Higher | ρ = 0.40 |

**Explanation:** Extended CoT reasoning causes the model's pre-generation activations to correlate more strongly with human-perceived difficulty rather than its own solve rate. The model "thinks harder" about problems humans find hard, even if it will ultimately solve them—making the activation signal noisier as a predictor of the model's own success.

> [!NOTE]
> "Probe performance drops from ρ=0.58 (low reasoning) to ρ=0.40 (high reasoning), despite the model achieving higher accuracy."

---

## Routing Applications

### Cascade Strategy (Two-Model)

Route each query to a cheaper base model or a more capable (expensive) model based on the probe's predicted success probability.

```
Algorithm: Threshold Cascade Routing
Input: query q, probe score p̂ = Pr(base model succeeds | h_q), threshold τ

if p̂ ≥ τ:
    use base_model(q)      # cheap
else:
    use strong_model(q)    # expensive
```

Result: Matching the strong model's accuracy at **17% cost reduction** on MATH.

### Utility-Based Routing (Multi-Model Pool)

For a heterogeneous pool of $M$ models with costs $c_m$ and probe-estimated success probabilities $\hat{p}_m$, route to the model maximizing expected utility:

```math
\begin{align}
  m^* = \arg\max_{m \in \mathcal{M}} \; \hat{p}_m - \lambda \cdot c_m
\end{align}
```

where $\lambda$ is a cost-sensitivity hyperparameter.

Result: **70% cost reduction** compared to always using `GPT-OSS-20B-high` exclusively on the MATH dataset, at equivalent accuracy.

### Benchmark-Adaptive Behavior

The routing strategy adapts to dataset characteristics:

- **AIME** (hard, high accuracy variance): Prefers stronger models when base model confidence is low.
- **GSM8K** (easy, accuracy plateaus at 94.5%): Selects the cheapest model since probe accuracy is uniformly high.

---

## Experiments

- **Datasets:**
  - **E2H-AMC**: 4,000 AMC problems with human IRT difficulty scores and LLM solve rates; used to separate human vs. model difficulty.
  - **MATH**: Standard math reasoning benchmark; used for routing evaluation.
  - **GSM8K**: Grade-school math; 94.5% accuracy plateau observed.
  - **AIME (1983–2024)**: Competition math; high difficulty variance.
  - **LiveCodeBench**: Code generation tasks; validates generalization beyond math.
- **Models tested:** GPT-OSS-20B (low and high reasoning budgets), plus additional models for multi-model routing.
- **Probe evaluation:** Spearman ρ for continuous targets; AUROC for binary classification (AUROC > 0.7 in most settings, several > 0.8).

---

## Differences from Similar Approaches

| Approach | When Routing Decision is Made | Signal Used | Cost |
|---|---|---|---|
| **This paper (pre-gen probes)** | Before any generation | Internal activations | Single forward pass |
| Confidence-based (token probs) | After generation | Output token probabilities | Full generation required |
| Self-consistency / majority voting | After multiple generations | Agreement across samples | $k$ × generation cost |
| Length heuristics | Before generation | Question character count | Negligible, but weak |
| TF-IDF features | Before generation | Bag-of-words similarity | Lightweight, but weaker than probes |

The key differentiator is that pre-generation probes require **no output tokens** and outperform surface-level features while achieving competitive routing performance versus methods that require full generation.

---

## Limitations

- Probe quality degrades under extended reasoning, suggesting that pre-generation activations become less cleanly separable as the model's reasoning becomes more complex.
- Alternative probing positions (e.g., mid-sequence, final layers only) and non-linear probes are not explored.
- Cross-domain transfer (e.g., probe trained on MATH, applied to LiveCodeBench) is not evaluated.
- Routing policies use fixed-$k$ majority voting rather than learned adaptive strategies.

> [!CAUTION]
> The authors identify that "probe reliability—not routing sophistication—bottlenecks performance," implying that better routing algorithms would not help if the underlying difficulty signal is noisy.
