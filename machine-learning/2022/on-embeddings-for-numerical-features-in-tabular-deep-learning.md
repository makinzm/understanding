# Meta Information

- URL: [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Gorishniy, Y., Rubachev, I., & Babenko, A. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. arXiv:2203.05556.

> [!CAUTION]
> NOTE comments are my personal understanding and may contain errors.

# 1. Introduction

Tabular data—structured tables of rows and columns—has traditionally been a domain where gradient boosted decision trees (GBDT) like XGBoost and CatBoost outperform deep learning methods. This paper identifies **how scalar numerical features are initially represented (embedded) before entering deep networks** as a critical and underexplored design axis.

The standard approach is trivial: a scalar value $x \in \mathbb{R}$ is passed as-is to the first linear layer. The authors argue this is unnecessarily restrictive and that mapping scalars into higher-dimensional vectors before the main network can substantially improve downstream accuracy—across multiple backbone architectures.

The two main embedding strategies proposed are:
1. **Piecewise Linear Encoding (PLE)**: encodes a scalar via a piecewise linear transformation over learned or quantile-derived bins.
2. **Periodic Activations (P)**: encodes a scalar via trainable sine-cosine functions.

# 2. Preliminaries

## 2.1 Tabular Data Setting

A tabular dataset consists of $n$ objects $\{x_i, y_i\}_{i=1}^n$ where each object $x_i$ has $k$ features: $x_i = (x_i^{(1)}, \ldots, x_i^{(k)})$. Features can be numerical ($\mathbb{R}$) or categorical. The label $y_i$ may be a real value (regression), binary (binary classification), or a class index (multiclass classification).

The paper focuses on the embedding of individual numerical features $x^{(j)} \in \mathbb{R}$ into a vector $e^{(j)} \in \mathbb{R}^d$ before passing to the main backbone.

## 2.2 Backbone Architectures

Three backbone architectures are evaluated:

| Backbone | Description |
|----------|-------------|
| **MLP** | Standard feedforward network with ReLU activations and batch normalization |
| **ResNet** | MLP with residual connections; stronger baseline than vanilla MLP |
| **Transformer** | Self-attention architecture adapted from FT-Transformer for tabular data |

Each backbone receives concatenated feature embeddings $[e^{(1)}, \ldots, e^{(k)}]$ and outputs a prediction.

# 3. Embedding Methods

## 3.1 Linear (L) and Linear-ReLU (LR)

The simplest embedding maps a scalar $x \in \mathbb{R}$ to a $d$-dimensional vector:

$$e = W x + b, \quad W \in \mathbb{R}^d, \; b \in \mathbb{R}^d$$

Applied **per-feature** (each feature has its own $W^{(j)}, b^{(j)}$). Adding a ReLU gives the **LR** variant:

$$e = \mathrm{ReLU}(W x + b)$$

These are used as simple baselines. The linear mapping is a "quite restrictive parametric mapping" that limits the complexity of the scalar-to-vector transformation.

> [!NOTE]
> Dimensions: Input $x \in \mathbb{R}$ (scalar), Output $e \in \mathbb{R}^d$. Each feature $j$ has its own weight vector $W^{(j)} \in \mathbb{R}^d$ and bias $b^{(j)} \in \mathbb{R}^d$.

## 3.2 Piecewise Linear Encoding (PLE)

PLE maps a scalar $x$ to a $T$-dimensional vector by dividing the feature's value range into $T$ intervals and computing a soft membership value for each interval:

$$\mathrm{PLE}(x) = [e_1, \ldots, e_T] \in \mathbb{R}^T$$

where each component $e_t$ (for bin $t$ with boundaries $b_{t-1}$ and $b_t$) is:

$$e_t = \begin{cases}
0 & \text{if } x < b_{t-1} \text{ and } t > 1 \\
1 & \text{if } x \geq b_t \text{ and } t < T \\
\dfrac{x - b_{t-1}}{b_t - b_{t-1}} & \text{otherwise}
\end{cases}$$

This can be interpreted as: bins to the left of $x$ receive 1, the bin containing $x$ receives a fractional value proportional to how far into the bin $x$ is, and bins to the right receive 0.

> [!NOTE]
> Input: $x \in \mathbb{R}$ (scalar). Output: $\mathrm{PLE}(x) \in [0,1]^T$ (piecewise linear encoding with $T$ bins). Bin boundaries $b_0 < b_1 < \ldots < b_T$ are precomputed from training data.

### Bin Boundary Computation

Two strategies are proposed:

| Strategy | Name | Method |
|----------|------|--------|
| Quantile-based | PLE_q | $b_t = Q_{t/T}(\{x_i^{(j)}\}_{i \in \text{train}})$, the $t/T$ quantile of feature $j$ on the training set |
| Target-aware | PLE_t | Splits from a shallow decision tree trained on feature $j$ alone to predict $y$ |

The quantile strategy is hyperparameter-free given $T$; the target-aware strategy can capture label-correlated breakpoints.

### Full PLE-based Embedding (Q-LR, T-LR)

A PLE encoding is followed by a linear layer and ReLU:

$$e^{(j)} = \mathrm{ReLU}(W^{(j)} \cdot \mathrm{PLE}(x^{(j)}) + b^{(j)}), \quad W^{(j)} \in \mathbb{R}^{d \times T}$$

So the pipeline is: $x^{(j)} \xrightarrow{\mathrm{PLE}} \mathbb{R}^T \xrightarrow{\mathrm{Linear}} \mathbb{R}^d \xrightarrow{\mathrm{ReLU}} \mathbb{R}^d$.

## 3.3 Periodic Activations (P, PLR)

Inspired by positional encodings and random Fourier features, numerical features are encoded via trainable sinusoidal functions:

$$v^{(j)} = [2\pi c_1^{(j)} x^{(j)}, \; 2\pi c_2^{(j)} x^{(j)}, \; \ldots, \; 2\pi c_k^{(j)} x^{(j)}] \in \mathbb{R}^k$$

$$\mathrm{Periodic}(x^{(j)}) = [\sin(v^{(j)}), \; \cos(v^{(j)})] \in \mathbb{R}^{2k}$$

where $c_l^{(j)} \sim \mathcal{N}(0, \sigma)$ are **trainable parameters** initialized from a Gaussian. The hyperparameters $k$ (number of frequency components) and $\sigma$ (initialization scale) are tuned per dataset.

> [!NOTE]
> Input: $x^{(j)} \in \mathbb{R}$. Output: $\mathrm{Periodic}(x^{(j)}) \in \mathbb{R}^{2k}$. Each feature has $k$ trainable coefficients $c_l^{(j)}$.

This differs from Fourier features (Tancik et al., 2020) in that: (1) coefficients are **learned** rather than fixed, (2) they are applied per-feature rather than mixed across features.

The **PLR** variant adds a linear layer and ReLU after the periodic encoding:

$$e^{(j)} = \mathrm{ReLU}(W^{(j)} \cdot \mathrm{Periodic}(x^{(j)}) + b^{(j)}), \quad W^{(j)} \in \mathbb{R}^{d \times 2k}$$

## 3.4 Summary of Embedding Variants

| Name | Pipeline | Key Property |
|------|----------|--------------|
| L | $x \to \text{Linear} \to e$ | Simplest non-trivial embedding |
| LR | $x \to \text{Linear} \to \text{ReLU} \to e$ | Non-linear, no extra structure |
| Q-LR | $x \to \text{PLE}_q \to \text{Linear} \to \text{ReLU} \to e$ | Quantile-based piecewise linear |
| T-LR | $x \to \text{PLE}_t \to \text{Linear} \to \text{ReLU} \to e$ | Target-aware piecewise linear |
| P | $x \to \text{Periodic} \to e$ | Trainable sinusoidal, no Linear |
| PLR | $x \to \text{Periodic} \to \text{Linear} \to \text{ReLU} \to e$ | Trainable sinusoidal + projection |

# 4. Algorithm: Training with Embeddings

The overall computation for a single object $x = (x^{(1)}, \ldots, x^{(k)})$:

```
Input: x ∈ ℝ^k (numerical features), backbone θ
Output: prediction ŷ

1. For each feature j = 1, ..., k:
   a. Embed: e^(j) = Embed(x^(j))       # e.g., PLR, Q-LR, etc.
      → e^(j) ∈ ℝ^d

2. Concatenate: E = [e^(1), e^(2), ..., e^(k)]   # E ∈ ℝ^(k·d)

   (For Transformer backbone: treat each e^(j) as a token → E ∈ ℝ^(k × d))

3. Forward through backbone:
   ŷ = Backbone(E; θ)

4. Compute loss L(ŷ, y)

5. Backpropagate through backbone and embedding parameters jointly
```

Embedding parameters (e.g., $c_l^{(j)}$ for Periodic, $W^{(j)}$ for Linear) are trained end-to-end with the backbone via standard gradient descent. For PLE, the bin boundaries are **fixed** after precomputation (not trained).

# 5. Experiments

## 5.1 Datasets

Eleven publicly available tabular datasets spanning regression, binary classification, and multiclass classification:

| ID | Dataset | Objects | Numerical | Categorical | Task |
|----|---------|---------|-----------|-------------|------|
| GE | Gesture Phase | 9,873 | 32 | 0 | Multiclass |
| CH | Churn Modelling | 10,000 | 10 | 1 | Binclass |
| CA | California Housing | 20,640 | 8 | 0 | Regression |
| HO | House 16H | 22,784 | 16 | 0 | Regression |
| AD | Adult | 48,842 | 6 | 8 | Binclass |
| OT | Otto Group | 61,878 | 93 | 0 | Multiclass |
| HI | Higgs Small | 98,049 | 28 | 0 | Binclass |
| FB | Facebook Comments | 197,080 | 50 | 1 | Regression |
| SA | Santander Transactions | 200,000 | 200 | 0 | Binclass |
| CO | Covertype | 581,012 | 54 | 0 | Multiclass |
| MI | MSLR-WEB10K | 1,200,192 | 136 | 0 | Regression |

Each dataset is split into train/validation/test. The evaluation metric is RMSE for regression, accuracy for classification.

## 5.2 Baselines

- **GBDT**: CatBoost, XGBoost (with hyperparameter tuning)
- **Backbone without embedding**: MLP, ResNet, Transformer (vanilla)
- **Backbone with embedding**: MLP-L, MLP-LR, MLP-Q-LR, MLP-T-LR, MLP-P, MLP-PLR; same for ResNet and Transformer

## 5.3 Key Results

**Embedding improves all backbones** (single model, average rank over 11 datasets):

| Model | Avg Rank |
|-------|----------|
| MLP (vanilla) | 8.5 |
| MLP-LR | 5.5 |
| MLP-PLR | 3.0 ± 2.4 |
| CatBoost | 3.6 ± 2.9 |
| Transformer-PLR | ~3.5 |

- **MLP-PLR** outperforms vanilla Transformer on average and is competitive with CatBoost.
- **California Housing and Adult datasets**: DL models (with embeddings) match GBDT for the first time, closing a long-standing gap.
- **PLE reduces preprocessing sensitivity**: Models with PLE encoding are significantly less sensitive to whether features are standardized, quantile-transformed, or left raw.

## 5.4 Parameter Overhead

Numerical embeddings substantially increase the number of parameters, primarily because each feature has its own embedding weights. For example, MLP-PLR on the Churn dataset has ×250 more parameters than vanilla MLP. Despite this, training time increase is moderate (e.g., ×1.5), since the embedding operations are simple.

# 6. Comparison to Similar Methods

| Method | Approach | Key Difference |
|--------|----------|----------------|
| **This work (PLE)** | Piecewise linear per-feature encoding | Bin boundaries from quantiles or decision trees; soft membership |
| **This work (PLR)** | Trainable sinusoidal encoding | Coefficients are learned, applied per-feature |
| **Standard MLP** | Raw scalar input | No transformation; relies on the first linear layer to learn structure |
| **Fourier Features** (Tancik et al., 2020) | Fixed random frequencies, feature-mixing | Applied across all features jointly; fixed (not learned) |
| **GBDT (XGBoost, CatBoost)** | Tree splits on numerical features | Non-differentiable; naturally handles arbitrary value ranges |
| **FT-Transformer** | Per-feature linear + Transformer | Uses linear embedding only; this paper replaces linear with richer embeddings |

> [!IMPORTANT]
> The key insight is that PLE and Periodic embeddings are **feature-wise** (each feature has its own embedding parameters), which is why they are applicable to tabular data without assuming relationships between features.

# 7. Applicability

- **Who**: Practitioners training deep learning models on structured tabular data with numerical features.
- **When**: When vanilla MLP or Transformer underperforms GBDT on a tabular benchmark; when features span varying scales or have non-uniform distributions.
- **Where**: Any tabular dataset, especially with many numerical features. PLE_q requires no labels beyond training data; PLE_t uses label information to find informative splits.
- **Limitations**: Large parameter overhead per feature (especially PLR); bin boundaries for PLE must be recomputed if the training data distribution shifts significantly.

# Experiments

- Datasets: Gesture Phase, Churn Modelling, California Housing, House 16H, Adult, Otto Group, Higgs Small, Facebook Comments, Santander Transactions, Covertype, MSLR-WEB10K (11 total)
- Hardware: Not specified
- Optimizer: Adam (standard for DL baselines); Optuna used for hyperparameter search
- Results: MLP-PLR achieves average rank 3.0 across 11 datasets, competitive with CatBoost (rank 3.6) and outperforming vanilla Transformer. DL models with embeddings match GBDT for the first time on California Housing and Adult datasets.
