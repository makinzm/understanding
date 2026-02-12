# Meta Information

- URL: [Deep Delta Learning](https://arxiv.org/abs/2601.00417)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, Y., Liu, Y., Wang, M., & Gu, Q. (2026). Deep Delta Learning. arXiv:2601.00417.

# Deep Delta Learning

## Overview

Deep Delta Learning (DDL) generalizes the shortcut connections in residual networks from fixed identity mappings $X_{l+1} = X_l + F(X_l)$ to learnable, state-dependent linear operators. Rather than simply adding a function's output to the current hidden state, DDL applies a rank-1 perturbation of the identity — called the **Delta Operator** — to the residual path, enabling the network to learn when and how much to deviate from identity, projection, or reflection at each layer.

The paper targets Transformer-based language model pretraining, where improving depth-wise feature transformations can reduce validation loss and improve downstream task accuracy without changing the overall architecture or parameter count significantly.

## 2. Deep Delta Learning

### 2.1 Preliminaries: Householder Transformation

For a nonzero vector $k \in \mathbb{R}^d$, the **Householder matrix** is:

$$H_k = I - 2\frac{kk^\top}{\|k\|_2^2}$$

Geometrically, $H_k$ reflects any vector across the hyperplane perpendicular to $k$. This is an orthogonal matrix ($H_k^\top H_k = I$) that serves as the theoretical upper bound of the Delta Operator.

### 2.2 Formulation of the Delta Operator

**Definition 2.2 (Delta Operator):**

$$A(X) = I - \beta(X)k(X)k(X)^\top$$

where:
- $k(X) \in \mathbb{R}^d$ is a **unit-norm direction** learned from $X$: $k(X) = \tilde{k}(X) / \|\tilde{k}(X)\|_2$
- $\beta(X) \in (0, 2)$ is a **scalar gate** parameterized as $\beta(X) = 2 \cdot \sigma(\text{Linear}(G(X)))$, with $\sigma$ being the sigmoid function and $G(\cdot)$ a pooling or convolution operation

**Delta Residual Block Output:**

$$X_{l+1} = A(X_l)X_l + \beta(X_l)k(X_l)v(X_l)^\top$$

where $X_l \in \mathbb{R}^{d \times d_v}$ is the hidden state matrix at layer $l$ and $v(X_l) \in \mathbb{R}^{d_v}$ is a learned value vector.

Combining the two terms yields a single **rank-1 delta write**:

$$X_{l+1} = X_l + \beta(X_l)k(X_l)\bigl(v(X_l)^\top - k(X_l)^\top X_l\bigr)$$

This can be interpreted as: erase the $k$-direction component of $X_l$ and simultaneously inject a new value $v$, scaled by gate $\beta$.

**Projected dynamics along $k$:**

$$k_l^\top X_{l+1} = (1-\beta_l)k_l^\top X_l + \beta_l v_l^\top$$

Features in the complement subspace $k^\perp$ are left unchanged.

## 3. Analysis of DDL

### 3.1 Spectral Decomposition

**Theorem 3.1:** For $A = I - \beta k k^\top$ with unit vector $k$ ($k^\top k = 1$):

- Eigenvalue $(1-\beta)$ with multiplicity 1, eigenvector $k$
- Eigenvalue $1$ with multiplicity $(d-1)$, eigenspace $k^\perp$

As a consequence, the determinant of the spatial operator is $\det(A(X)) = 1 - \beta(X)$.

This shows that DDL acts as a selective scaling: the chosen direction $k$ is scaled by $(1-\beta)$, while all orthogonal directions are preserved exactly.

### 3.2 Unified Geometric Interpretation

The gate $\beta$ continuously interpolates between three canonical linear transformations:

| $\beta$ value | Operator behavior | Geometric meaning |
|---|---|---|
| $\beta \to 0$ | $A \to I$ | Identity mapping (standard residual) |
| $\beta = 1$ | $A = I - kk^\top$ | Orthogonal projection onto $k^\perp$ |
| $\beta = 2$ | $A = I - 2kk^\top = H_k$ | Householder reflection across $k^\perp$ |

This unification allows a single block to dynamically choose between "copy forward," "erase a direction," or "reflect across a hyperplane" depending on the input.

### 3.3 Connection to the Delta Rule

The update rule $k_l^\top X_{l+1} = (1-\beta_l)k_l^\top X_l + \beta_l v_l^\top$ matches the classical **Widrow-Hoff delta rule** from neural network learning theory. The component along $k$ decays toward the target $v^\top$ at rate $\beta_l$, while the complement is fixed. This makes DDL an instance of associative memory writing via rank-1 updates — but applied along the depth axis rather than the sequence axis (as in DeltaNet).

### 3.6 Relationship to DeltaNet

**Appendix A** establishes a structural isomorphism: DDL applies rank-1 delta writes **depth-wise** (layer-to-layer), while DeltaNet applies the same algebraic structure **sequence-wise** (token-to-token in a recurrent update). Formally, under a transpose convention change, the DDL update rule and DeltaNet's recurrence are algebraically equivalent. This reveals that techniques developed for DeltaNet (e.g., hardware-efficient parallel scans) may transfer to DDL.

> [!NOTE]
> DeltaNet: "a linear RNN which uses delta rule for in-context learning" (Schlag et al., 2021; Yang et al., 2024).

## 4. DDL Transformer

### 4.1 Scalar Regime ($d_v = 1$)

When the hidden state is a vector $x_l \in \mathbb{R}^d$ (standard Transformer), the block update simplifies to:

$$x_{l+1} = x_l + \beta_l(v_l - k_l^\top x_l)k_l$$

Two parameterization variants are explored:

- **k-Map**: the Transformer backbone output determines $k$; a separate linear projection produces $v$
- **v-Map**: the backbone output determines $v$; an auxiliary branch produces $k$

Gate $\beta_l$ is computed via $\beta = 2\sigma(\text{Linear}(\text{RMSNorm}(x_l)))$.

### 4.2 Expanded State Regime ($d_v > 1$)

When the hidden state is a matrix $X_l \in \mathbb{R}^{d \times d_v}$ with $d_v > 1$ (e.g., $d_v = 4$), the **Compress-Process-Expand** protocol is used:

**Algorithm: Expanded-State DDL Forward Pass**
```
Input: token embeddings E ∈ ℝ^{T × d}

1. Initialize expanded state:
   X₀ = E ⊗ 1_{d_v}^T  // replicate embedding d_v times

2. For each layer l:
   a. Compress: apply ShortConv (causal depthwise conv) to X_l
      C_l = ShortConv(X_l) ∈ ℝ^{d × 1}  // weighted pooling across d_v

   b. Process: run standard Transformer sublayer on C_l
      P_l = Transformer(C_l) ∈ ℝ^{d × 1}

   c. Expand: generate rank-1 update from P_l
      k_l = normalize(W_k P_l) ∈ ℝ^d
      v_l = W_v P_l ∈ ℝ^{d_v}
      β_l = 2σ(Linear(RMSNorm(P_l))) ∈ ℝ

   d. Apply Delta update:
      X_{l+1} = X_l + β_l k_l (v_l^T - k_l^T X_l)

3. Output: X_L  (collapsed to d-dim for logits)
```

Variants include **DDL-CC** (channel convolution along $d_v$), **DDL-EC** (embedding expansion convolution), and **DDL-CC-EC** (combined).

> [!IMPORTANT]
> The $d_v > 1$ regime increases memory by a factor of $d_v$ for the hidden state, but computation scales modestly since the Transformer sublayer operates on the compressed $d$-dim representation.

## 5. Experiments

### Setup

- **Dataset**: FineWeb-Edu 100B (100B training tokens, 0.1B validation tokens, sequence length 1024)
- **Hardware**: 4 × NVIDIA H200 GPUs
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1, gradient clip 1.0)
- **Learning rate**: $10^{-3}$ with cosine schedule (2,000 warmup steps), 100,000 total steps
- **Backbone**: Pre-norm RMSNorm, RoPE embeddings, SwiGLU activations, QK normalization, $\mu$P initialization
- **Evaluation**: lm-evaluation-harness with 1-shot and 0-shot settings

### Model Scales

| Scale | Parameters | Layers | Heads | Hidden size |
|---|---|---|---|---|
| Small | 124M | 12 | 6 | 768 |
| Medium | 353M | 24 | 8 | 1024 |

### Baseline

Standard residual addition (nanoGPT-based backbone), identical in parameter count and architecture except for the shortcut mechanism.

### Key Results

**Validation Loss and Perplexity (final step, lower is better):**

| Model | Small Val Loss | Small PPL | Medium Val Loss | Medium PPL |
|---|---|---|---|---|
| Baseline | 2.85426 | 17.3616 | 2.60532 | 13.5356 |
| DDL ($d_v=1$) | 2.84817 | 17.2562 | 2.60388 | 13.5161 |
| DDL ($d_v=4$) | **2.83545** | **17.0381** | **2.59267** | **13.3654** |

**Downstream 1-shot accuracy, Small model (averaged over 8 tasks):**

| Model | Avg Accuracy |
|---|---|
| Baseline | 48.56 |
| DDL ($d_v=1$) | 48.73 |
| DDL ($d_v=4$) | 48.91 |
| DDL-CC | 49.13 |
| DDL-EC | **49.47** |
| DDL-CC-EC | 49.29 |

The expanded-state variants consistently outperform the standard residual baseline, with larger gains at medium scale. Gains are modest but consistent across all eight downstream benchmarks (ARC-Challenge, ARC-Easy, HellaSwag, OpenBookQA, PIQA, SciQ, Social IQA, WinoGrande).

## 6. Related Work

DDL connects to several lines of prior work:

| Research direction | Connection |
|---|---|
| **Gated residual networks** (Highway Networks, GRU, LSTM) | DDL adds geometric interpretation to gating via rank-1 structure |
| **Orthogonal/unitary networks** | Householder parameterization is a special case ($\beta=2$) |
| **Neural ODEs** | DDL interprets layer depth as discrete time in a dynamical system |
| **Memory-augmented architectures** | Expanded-state regime ($d_v>1$) acts as learnable working memory |
| **Depth-adaptive mechanisms** | Gate $\beta \to 0$ allows layer skipping without explicit skip connections |
| **DeltaNet** | Algebraically equivalent but applied along depth vs. sequence axis |

> [!TIP]
> Project code: https://github.com/yifanzhang-pro/deep-delta-learning

## Differences from Standard Residual Networks

| Property | Standard ResNet/Transformer | Deep Delta Learning |
|---|---|---|
| Shortcut operator | Fixed identity $I$ | Learnable $A(X) = I - \beta kk^\top$ |
| Update mechanism | Additive: $X + F(X)$ | Rank-1 write: erase + inject along $k$ |
| Direction selectivity | None | Selective: only $k$-direction is modified |
| Geometric regime | Fixed | Dynamically interpolates identity/projection/reflection |
| State dimensionality | $\mathbb{R}^d$ | Optionally $\mathbb{R}^{d \times d_v}$ (matrix state) |
| Connection to sequence models | None | Equivalent to DeltaNet applied depth-wise |

> [!CAUTION]
> The gains in validation loss and downstream accuracy are real but modest (e.g., ~0.015 improvement in validation loss for the 353M model). DDL adds architectural complexity; whether the benefit justifies this in production-scale training remains to be evaluated at larger scales (e.g., 1B+ parameters).
