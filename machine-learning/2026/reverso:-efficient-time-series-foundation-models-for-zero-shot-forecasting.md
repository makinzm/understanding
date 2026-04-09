# Meta Information

- URL: [Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting](https://arxiv.org/abs/2602.17634)
- LICENSE: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Fu, X., Li, Y., Papaioannou, G., & Kim, Y. (2026). Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting. arXiv:2602.17634.

# Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting

## Overview

Reverso is a family of compact foundation models (0.2M–2.6M parameters) for zero-shot time series forecasting. The core claim is that small **hybrid models** combining long convolution layers and linear RNN layers (DeltaNet) can match the forecasting accuracy of transformer-based foundation models that are more than 100× larger.

**Who should use this:** Practitioners needing zero-shot time series forecasting at low computational cost; researchers studying efficient architectures as alternatives to large transformers for sequential data.

**When:** Zero-shot or few-shot forecasting across diverse domains (energy, weather, retail, finance) where training data for the target series is scarce or unavailable.

## Problem Setting

### Input / Output

- **Input:** A univariate time series context window $x \in \mathbb{R}^{L}$, where $L = 2048$ is the fixed context length.
  - Shorter series are padded via backfill; longer series are downsampled using FFT-based frequency detection.
  - Min-max normalized to $[0, 1]$ before feeding into the model.
- **Output:** A forecast of length $p = 48$ steps ahead $\hat{y} \in \mathbb{R}^{p}$.
  - For horizons $H > p$, the model performs autoregressive rollout: predict $p$ steps, append to context, predict next $p$ steps, and so on until $H$ steps are generated.

### Difference from Transformer-based Models

| Aspect | Transformer (e.g., Chronos, Sundial) | Reverso |
|--------|--------------------------------------|---------|
| Parameters | ~710M – billions | 0.2M – 2.6M |
| Sequence mixing | Full attention $O(L^2)$ | Conv + Linear RNN $O(L)$ |
| Training GPU hours | Hundreds to thousands | 10–40 H100 GPU-hours |
| Prediction style | Direct multi-step or tokenized | Autoregressive (stride $p=48$) |
| Normalization | z-score | min-max to $[0,1]$ |

## Architecture

### High-level Structure

Reverso stacks $N$ blocks, each consisting of:
1. **Sequence mixing layer** (alternating between long convolution and DeltaNet)
2. **Channel mixing layer** (MLP with 4× expansion)

Followed by an **attention-based decoder head** that produces the final forecast.

```
Input x ∈ ℝ^L
  → Token embedding: ℝ^L → ℝ^{L × d}
  → Block 1: [Sequence Mixer → LayerNorm → Channel MLP → LayerNorm]
  → Block 2: ...
  → Block N
  → Attention Decoder: ℝ^{L × d} → ℝ^p
```

Model configurations:

| Model | Params | Layers $N$ | Hidden dim $d$ |
|-------|--------|------------|----------------|
| Reverso-Nano | 200K | 2 | 32 |
| Reverso-Small | 550K | 4 | 64 |
| Reverso | 2.6M | 8 | 128 |

### Long Convolution Layer

Each long convolution block applies a **depthwise separable convolution** with kernel size equal to the full context length $L$:

$$h_t = \sum_{\tau=0}^{L-1} w_\tau \cdot x_{t-\tau}$$

where $w \in \mathbb{R}^L$ are learned per-channel weights. This efficiently captures long-range temporal dependencies without quadratic attention cost.

### DeltaNet Layer (Linear RNN)

The DeltaNet layer implements a **linear recurrence** with a rank-1 state update rule. Given input token $x_i$, it produces keys $k_i \in \mathbb{R}^d$, values $v_i \in \mathbb{R}^d$, queries $q_i \in \mathbb{R}^d$, and a scalar gate $\beta_i \in (0,1)$.

**State update (recurrence):**

$$S_i = S_{i-1} \left(I - \beta_i k_i k_i^\top\right) + \beta_i v_i k_i^\top$$

where $S_i \in \mathbb{R}^{d \times d}$ is the hidden state matrix. This is a rank-1 correction to the previous state, subtracting the old association for key $k_i$ and writing the new one $v_i k_i^\top$.

**Output:**

$$o_i = S_i q_i$$

The recurrence can be parallelized via chunk-wise computation for training efficiency (analogous to how state space models are parallelized).

> [!NOTE]
> DeltaNet is inspired by the "delta rule" from Hebbian learning: selectively overwrite memory at a specific key rather than blending all updates uniformly.

### Channel Mixing (MLP)

Standard two-layer MLP with expansion ratio 4 and ReLU activation:

$$\text{MLP}(h) = W_2 \cdot \text{ReLU}(W_1 h + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$.

### Attention-based Decoder Head

Rather than a simple linear projection $\hat{y} = Wh_L$ from the last hidden state, Reverso uses a cross-attention decoder:

- **Query:** $p$ learned query vectors $Q \in \mathbb{R}^{p \times d}$ (one per forecast step)
- **Key/Value:** the full hidden state sequence $H \in \mathbb{R}^{L \times d}$
- **Output:** $\hat{y} \in \mathbb{R}^{p}$ after a linear projection

$$\hat{y} = \text{softmax}\left(\frac{Q H^\top}{\sqrt{d}}\right) H W_o$$

Ablation shows the attention decoder provides ~2% MASE improvement over a bilinear projection head.

## Training Methodology

### Pretraining Data

- **GiftEval pretraining dataset:** 4.5 million time series totalling 230 billion time points across diverse domains (retail, finance, energy, weather, transport).
- To handle dataset imbalance, each constituent dataset is capped at **100,000 samples per epoch**, preventing large datasets from dominating training.
- **Synthetic data:** 1 million additional time series generated via Gaussian processes with randomly composed kernels, plus synthetic spike and trapezoidal patterns. Maximum length 4096. Ablation confirms synthetic data contributes 0.048 MASE improvement.

### Data Augmentation Pipeline

Applied sequentially to each training sample:

1. **Downsampling** – randomly reduce temporal resolution
2. **Amplitude modulation** – scale values by random factor
3. **Temporal/vertical flipping** – reverse time order or negate values
4. **Censoring** – randomly mask portions of the context
5. **Mixup** – interpolate between two random series

Each augmentation individually provides ~0.1% MASE improvement; all together yield a total gain of 0.017 MASE.

### Optimization

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | $5 \times 10^{-4}$ |
| LR schedule | WSD (warmup–stable–decay) |
| Weight decay | 0.1 |
| Batch size | 512 |
| Normalization | Min-max to $[0, 1]$ |
| Context length | $L = 2048$ |
| Prediction length | $p = 48$ |

### Training Cost

| Model | GPU Hours (H100) |
|-------|-----------------|
| Reverso-Nano | ~10 |
| Reverso-Small | ~20 |
| Reverso | ~40 |

## Inference Strategies

### Flip Equivariance

Time series forecasting is expected to be approximately **equivariant to temporal flipping** in many cases (e.g., symmetric trend patterns). Reverso exploits this by averaging two predictions:

$$\hat{y}_{\text{final}} = \frac{f(x) - f(-x)}{2}$$

where $f(-x)$ denotes running the model on the negated (vertically flipped) input. This requires two forward passes but consistently reduces forecasting error, especially on short sequences.

> [!NOTE]
> The negative sign on $f(-x)$ ensures the averaged output is compatible with the original scale (flipping negates values, so output of flipped model must be negated back).

### FFT-based Downsampling

When a time series is longer than context length $L = 2048$ or has seasonality exceeding $L$, Reverso uses FFT to identify dominant frequency components:

1. Compute FFT of the input series.
2. Identify the top-$k$ dominant frequencies.
3. Select a downsampling factor $s$ such that the primary seasonality period fits within $L$.
4. Downsample the series by factor $s$ before feeding into the model; upsample predictions back.

This is critical for medium- and long-term forecasting horizons.

## Experiments

- **Datasets:**
  - **GiftEval benchmark** (held-out evaluation): diverse domains including M4, M5, Tourism, ETT variants, Electricity, Weather, and others.
  - **TSLib/LTSF benchmarks:** ETTm1, ETTm2, ETTh1, ETTh2, Electricity, Weather — standard long-term forecasting benchmarks with horizons $H \in \{96, 192, 336, 720\}$.
- **Hardware:** H100 GPUs (10–40 GPU-hours for training)
- **Optimizer:** AdamW, LR $5 \times 10^{-4}$, WSD schedule

### Key Results

**GiftEval Benchmark (MASE, lower is better):**

| Model | Params | MASE |
|-------|--------|------|
| Reverso | 2.6M | **0.711** |
| Reverso-Small | 550K | 0.726 |
| Reverso-Nano | 200K | 0.741 |
| Chronos-2 (comparable large model) | ~710M | ~0.700 |

Reverso (2.6M) achieves MASE competitive with transformer models 100× larger, establishing a new **performance-efficiency Pareto frontier**.

**TSLib/LTSF (average MAE across ETT, Electricity, Weather):**

Reverso achieves average MAE ≈ 0.322 across prediction horizons {96, 192, 336, 720}, competitive with Chronos-2 and Sundial at a fraction of model size.

> [!IMPORTANT]
> Reverso uses autoregressive rollout with stride $p=48$ for long horizons, while some baselines directly predict $H=720$ steps. Despite this, Reverso remains competitive on medium/long horizons, especially after downsampling augmentation.

## Comparison with Related Methods

| Method | Architecture | Params | Zero-shot | Univariate |
|--------|-------------|--------|-----------|------------|
| Chronos-2 | Transformer (T5) | 710M | Yes | Yes |
| Sundial | Transformer | Large | Yes | Yes |
| TimeMixer | MLP-Mixer | Moderate | No | Yes |
| **Reverso** | Conv + DeltaNet | 0.2M–2.6M | **Yes** | Yes |

Reverso's key differentiator is its **sub-million parameter count with zero-shot capability**, achieved by replacing attention with linear RNN (DeltaNet) and long convolution. Existing efficient models (e.g., TimeMixer) are not pretrained for zero-shot use.

## Limitations

- **Univariate only:** Multivariate correlations between channels are not exploited; each channel is forecasted independently.
- **Short-sequence gap:** On very short input sequences, Reverso underperforms larger models.
- **Point forecasts only:** The model does not produce uncertainty estimates or distributional forecasts.
- **Autoregressive rollout overhead:** Long-horizon predictions require repeated inference passes (stride $p=48$), adding latency for very long horizons.

## Code

- Implementation: [https://github.com/shinfxh/reverso](https://github.com/shinfxh/reverso)
