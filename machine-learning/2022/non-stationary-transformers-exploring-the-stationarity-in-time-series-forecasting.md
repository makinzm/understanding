# Meta Information

- URL: [Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://arxiv.org/abs/2205.14415)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yong Liu, Haixu Wu, Jianmin Wang, Mingsheng Long (2022). Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting. arXiv:2205.14415.

# Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting

## Overview

This paper addresses a fundamental tension in Transformer-based time series forecasting: **stationarizing** the input improves predictability by stabilizing distributional statistics, but excessive stationarization ("over-stationarization") causes the model to lose discriminative power across series with different temporal dynamics. The authors propose a framework—**Non-stationary Transformers**—that (1) applies lightweight normalization to improve training stability and (2) recovers the discarded non-stationary information through a novel attention mechanism called **De-stationary Attention**.

Applicable to any Transformer-based forecasting model (Transformer, Informer, Reformer, Autoformer, FEDformer, Pyraformer, etc.) as a plug-in improvement without significant computational overhead. Useful for practitioners working on real-world multivariate time series where non-stationarity is common (finance, traffic, epidemiology, weather).

## Problem: Over-stationarization

Standard practice applies z-score normalization to the input time series before feeding it to a Transformer. While this stabilizes training, it creates a problem: stationarized series from different temporal regimes produce nearly identical attention maps, making the model insensitive to the structural differences in the raw data. The paper quantifies non-stationarity using the **Augmented Dickey-Fuller (ADF) test statistic**: smaller (more negative) ADF values indicate higher stationarity. Datasets with high stationarity after normalization (e.g., Exchange, ILI) see the largest performance drops under standard transformers.

## Method

### Series Stationarization

**Input:** Raw multivariate time series $\mathbf{x} \in \mathbb{R}^{S \times C}$, where $S$ is the lookback window length and $C$ is the number of channels (variates).

**Normalization (instance-wise z-score):**

$$\mu_x = \frac{1}{S} \sum_{i=1}^{S} x_i \in \mathbb{R}^{C}$$

$$\sigma_x^2 = \frac{1}{S} \sum_{i=1}^{S} (x_i - \mu_x)^2 \in \mathbb{R}^{C}$$

$$x_i' = \frac{1}{\sigma_x} \odot (x_i - \mu_x)$$

where $\odot$ denotes element-wise multiplication. The normalized input $\mathbf{x}' \in \mathbb{R}^{S \times C}$ is then fed to the base Transformer model.

**De-normalization (output restoration):**

After the Transformer produces predictions $\mathbf{y}' \in \mathbb{R}^{T \times C}$ on the normalized input, the raw-scale output is recovered:

$$\hat{y}_i = \sigma_x \odot y_i' + \mu_x$$

The statistics $\mu_x$ and $\sigma_x$ are computed from the encoder input only and applied to both encoder and decoder.

### De-stationary Attention

**Motivation:** The attention scores computed on the stationarized series $\mathbf{x}'$ differ from those that would be computed on the original $\mathbf{x}$. To recover the non-stationary temporal dependencies, the authors derive the mathematical relationship between the two attention distributions.

**Theoretical derivation:** Let $Q, K, V$ be query/key/value matrices from unstationarized data, and $Q', K', V'$ from stationarized data. The exact softmax attention from raw data satisfies:

$$\text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) = \text{Softmax}\!\left(\frac{\sigma_x^2 Q'K'^\top + \mathbf{1}\Delta^\top}{\sqrt{d_k}}\right)$$

where $\Delta \in \mathbb{R}^{S}$ is a shift term that arises from the mean $\mu_x$, and $\sigma_x^2$ acts as a scale factor. This identity motivates two learned **de-stationary factors**:

- $\tau \in \mathbb{R}^{+}$: learned scale factor (approximates $\sigma_x^2$)
- $\Delta \in \mathbb{R}^{S}$: learned shift vector (approximates the mean-induced term)

**Implementation:**

$$\log \tau = \text{MLP}(\sigma_x,\, \mathbf{x})$$

$$\Delta = \text{MLP}(\mu_x,\, \mathbf{x})$$

$$\text{Attn}(Q', K', V', \tau, \Delta) = \text{Softmax}\!\left(\frac{\tau Q'K'^\top + \mathbf{1}\Delta^\top}{\sqrt{d_k}}\right) V'$$

Both MLPs take the stationarization statistics (scalar $\sigma_x$, vector $\mu_x \in \mathbb{R}^C$) plus the original series $\mathbf{x}$ as input, and have 2 hidden layers with dimensions in $\{64, 128, 256\}$.

### Complete Algorithm (Pseudocode)

```
Input: x ∈ ℝ^(S×C)   (encoder input, lookback window)
Output: ŷ ∈ ℝ^(T×C) (forecast horizon T)

// 1. Series Stationarization
μ_x  = mean(x, dim=0)        # ℝ^C
σ_x  = std(x, dim=0)         # ℝ^C
x'   = (x - μ_x) / σ_x      # ℝ^(S×C), stationarized input

// 2. De-stationary factors
τ    = exp(MLP_τ(σ_x, x))   # ℝ^+, scaling
Δ    = MLP_Δ(μ_x, x)        # ℝ^S, shifting

// 3. Transformer forward pass with De-stationary Attention
//    For each attention layer:
Q', K', V' = Linear(x')     # projected queries, keys, values
A = Softmax((τ · Q'K'^T + 1·Δ^T) / √d_k) · V'

// 4. De-normalization of output
y'   = TransformerDecoder(A, ...)   # ℝ^(T×C)
ŷ    = σ_x ⊙ y' + μ_x             # ℝ^(T×C), restored to original scale
```

## Relationship to Similar Methods

| Method | Normalization | Attention Type | Non-stationary Recovery |
|---|---|---|---|
| Standard Transformer | None | Vanilla softmax | No |
| RevIN (Kim et al., 2022) | Instance z-score + learnable affine | Unchanged | Partial (via affine) |
| Autoformer | None | Auto-Correlation | No |
| **Non-stationary Transformer** | Instance z-score | De-stationary Attention | Yes (learned τ, Δ) |

> [!NOTE]
> RevIN (Reversible Instance Normalization) also applies z-score normalization and restores the output. However, it does not modify the attention mechanism to recover the lost non-stationary information—only the final output scale is corrected. De-stationary Attention reconstructs the full attention distribution that would have been computed on the raw series.

> [!IMPORTANT]
> The De-stationary Attention is a **plug-in module**: it replaces the standard `Softmax(QK^T / √d_k)` computation in any existing Transformer, requiring only two small MLPs per attention head as additional parameters.

## Experiments

- **Datasets:** ETTh1, ETTh2, ETTm1, ETTm2 (electricity transformer temperature, hourly/minutely), Exchange (daily exchange rates, 8 countries), ILI (weekly US influenza-like illness data), Traffic (road occupancy, 862 sensors), Weather (21 meteorological indicators)
- **Prediction horizons:** {96, 192, 336, 720} time steps (multivariate), {24, 36, 48, 60} (ILI)
- **Baselines:** Autoformer, FEDformer, Pyraformer, Informer, Reformer, Transformer
- **Hardware:** NVIDIA TITAN V GPU (12 GB)
- **Optimizer:** Adam, learning rate $10^{-4}$, batch size 32
- **Loss:** MSE (L2)
- **Results:**
  - Applying Non-stationary Transformer framework to vanilla Transformer reduces MSE by **49.43%** on average across all benchmarks
  - Applying to Informer: **47.34%** MSE reduction; Reformer: **46.89%** reduction
  - Largest gains on highly non-stationary datasets: Exchange, ILI
  - Non-stationary Transformer achieves state-of-the-art on most ETT variants and ILI benchmarks compared to FEDformer and Autoformer

> [!TIP]
> Official implementation: [https://github.com/thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers)
