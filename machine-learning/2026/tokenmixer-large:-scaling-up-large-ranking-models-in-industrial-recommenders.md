# Meta Information

- URL: [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Jiang, Y., Zhu, J., Han, X., Lu, H., Bai, K., Yang, M., Wu, S., Zhang, R., Zhao, W., Bai, S., Zhou, S., Yang, H., Liu, T., Liu, W., Gong, Z., Ding, H., Chai, Z., Xie, D., Chen, Z., Zheng, Y., & Xu, P. (2026). TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders. arXiv:2602.06563.

---

# TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders

## Overview

TokenMixer-Large is an evolved large-scale ranking architecture developed by ByteDance that addresses critical limitations in the original TokenMixer (also called RankMixer) design. The core problem is that naively scaling recommendation models beyond ~500M parameters causes gradient degradation and residual design instability, preventing effective training. This paper introduces three main fixes — a mixing-and-reverting operation, inter-layer residuals, and an auxiliary loss — that collectively enable stable scaling to 7B+ parameters in production.

**Who uses this / when / where**: Industrial recommendation systems requiring large models (billion-scale parameters) to rank hundreds of candidates in real time, deployed at companies like ByteDance for e-commerce, advertising, and live-streaming platforms serving hundreds of millions of users.

## Background: Limitations of TokenMixer (RankMixer)

The original TokenMixer works by grouping feature embeddings into tokens and applying cross-token MLP mixing. However, it has several problems at larger scale:

1. **Sub-optimal residual design**: The residual path in the original mixing block adds the *mixing output* to the *original tokens*, but the mixing operation transposes dimensions ($T \times D \rightarrow H \times (T \cdot D/H)$), so the input and output have incompatible shapes. This forces the residual to use the un-transposed input, which loses positional semantics.

2. **Vanishing gradients in deep models**: Without special intervention, stacking many TokenMixer blocks causes gradients to decay before reaching early layers, preventing effective training of models deeper than ~10 blocks.

3. **Fragmented operator overhead**: Small, isolated operators such as custom activations and normalizations reduce GPU utilization due to kernel launch overhead and memory bandwidth waste.

## Architecture: TokenMixer-Large

### Tokenization (Input Representation)

Features are grouped and projected into $T$ tokens:

- Sparse feature embedding: $e_i = \text{Embedding}(F_i, d_i) \in \mathbb{R}^{d_i}$
- Group-wise MLP projection for group $G_i$: $X_i = \text{MLP}_i(\text{concat}[e_l, \ldots, e_m])$ for all $e_l, \ldots, e_m \in G_i$
- Global aggregation token: $X_G = \text{MLP}_g(\text{concat}[G_1, \ldots, G_{T-1}])$
- Final token matrix: $X = \text{concat}[X_G, X_0, \ldots, X_{T-1}] \in \mathbb{R}^{T \times D}$

where $T$ is the number of tokens and $D$ is the hidden dimension per token.

### Mixing-and-Reverting Operation

The key architectural fix that resolves residual shape mismatch:

**Step 1 — Mix (token → head layout)**:

$$
[x_t^{(0)}, \ldots, x_t^{(H)}] = \text{split}(X), \quad x_t^{(h)} \in \mathbb{R}^{D/H}
$$

$$
H_h = \text{concat}[x_1^{(h)}, \ldots, x_T^{(h)}] \in \mathbb{R}^{T \cdot D/H}, \quad H = \text{concat}[H_1, \ldots, H_H] \in \mathbb{R}^{H \times (T \cdot D/H)}
$$

$$
H^{\text{next}} = \text{Norm}(\text{pSwiGLU}(H) + H) \in \mathbb{R}^{H \times (T \cdot D/H)}
$$

**Step 2 — Revert (head → token layout)**:

The reverting step reconstructs the original token ordering from head layout, producing:

$$
X^{\text{revert}} \in \mathbb{R}^{T \times D}
$$

**Step 3 — Residual with compatible shapes**:

$$
X^{\text{next}} = \text{Norm}(\text{pSwiGLU}(X^{\text{revert}}) + X) \in \mathbb{R}^{T \times D}
$$

> [!IMPORTANT]
> The key insight is that reverting restores the token layout before adding the residual, so $X^{\text{revert}}$ and $X$ have the same shape $\mathbb{R}^{T \times D}$. This fixes the semantic mismatch in the original RankMixer residual.

### Per-token SwiGLU (pSwiGLU)

Each token gets its own set of projection weights, enabling the model to treat heterogeneous features differently:

$$
\text{pSwiGLU}(\cdot) = \text{FC}_{\text{down}}(\text{Swish}(\text{FC}_{\text{gate}}(\cdot)) \odot \text{FC}_{\text{up}}(\cdot))
$$

$$
\text{FC}_i(x_t) = W_i^t x_t + b_i^t, \quad i \in \{\text{up}, \text{gate}, \text{down}\}
$$

$$
W_{\text{up}}^t, W_{\text{gate}}^t \in \mathbb{R}^{D \times nD}, \quad W_{\text{down}}^t \in \mathbb{R}^{nD \times D}
$$

where $n$ is a hidden dimension scaling factor and each token $t \in [T]$ has its own weight matrices.

### Sparse Per-token MoE (S-P MoE)

To scale parameters efficiently, pSwiGLU is replaced with a Sparse Per-token Mixture of Experts:

$$
\text{S-P MoE}(\cdot) = \alpha \cdot \sum_{j=1}^{k-1} g_j(\cdot) \cdot \text{Expert}_j(\cdot) + \text{SharedExpert}(\cdot)
$$

where:
- $k$ is the number of top-k routed experts (plus 1 shared expert)
- $g_j(\cdot)$ is the learned gating weight for expert $j$
- $\alpha$ is a gate scaling factor set equal to the sparsity ratio (e.g., $\alpha = 0.5$ for 1:2 sparsity)
- Each expert: $\text{Expert}_j(\cdot) = \text{FC}_{\text{down},j}(\text{Swish}(\text{FC}_{\text{gate},j}(\cdot)) \odot \text{FC}_{\text{up},j}(\cdot))$
- Expert weight dimensions: $W_{\text{up}}^t, W_{\text{gate}}^t \in \mathbb{R}^{D \times nD/E}$, $W_{\text{down}}^t \in \mathbb{R}^{nD/E \times D}$, where $E$ is the number of experts per token

> [!NOTE]
> The paper calls this "sparse training, sparse serving" to distinguish it from previous "dense training, sparse serving" approaches. By using true sparse training (activating only 50% of parameters at 1:2 sparsity), the model matches dense performance while reducing compute.

### Training Stabilization: Inter-layer Residuals and Auxiliary Loss

Two mechanisms address vanishing gradients in deep models:

1. **Inter-layer residuals**: Every 2–3 layers, a direct skip connection bypasses multiple blocks, providing a short gradient path to early layers. The final layer is excluded from this connection to preserve task-specific representations.

2. **Auxiliary loss**: Intermediate layer outputs are also supervised with the same task loss, summed with the main output loss. This ensures all layers receive direct gradient signal from the target.

3. **Down-matrix small initialization**: The $W_{\text{down}}$ matrix is initialized with standard deviation 0.01 (versus the typical default of 1.0), preventing initial large-magnitude outputs that could destabilize training.

### Normalization

RMSNorm replaces LayerNorm throughout the model, reducing computation while maintaining training stability:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}} \cdot \gamma
$$

Pre-Norm design (normalization applied before each sublayer) is used instead of Post-Norm.

## Parallelism: Token Parallel Strategy

For multi-GPU training, a Token Parallel strategy is introduced where the $T$ tokens are distributed across $N$ GPUs:

```
Input:  X_prev [N*B, T, D]  (sharded by batch)
Step 1: All-to-All exchange  → [N*B, T/N, D]   (sharded by token)
Step 2: Mixing operation     → [H, N*B, T*D/H]  (sharded by head)
Step 3: Per-token SwiGLU/MoE
Step 4: All-to-All exchange  → [N*B, T, D]      (sharded by batch)
```

This reduces total communication steps from $4L$ (naïve pipeline) to $2L + 1$ for $L$ layers by merging consecutive All-to-All collectives at layer boundaries.

**Throughput**: This achieves a 96.6% throughput improvement with computation-communication overlap compared to no parallelism, and 1.7× inference speedup when combined with FP8 quantization.

## Comparison with Similar Architectures

| Feature | TokenMixer (RankMixer) | TokenMixer-Large | Transformer-based |
|---|---|---|---|
| Parameter sharing across tokens | Yes (shared MLP) | No (per-token weights) | No (shared attention) |
| Residual correctness | Mismatched shapes | Matching shapes via revert | Matching shapes |
| Scaling beyond 1B | Unstable gradients | Stable via inter-residuals + aux loss | High compute cost |
| Expert routing | Dense | Sparse Per-token MoE | Not applicable |
| Communication in parallel | 4L all2all ops | 2L+1 all2all ops | 4L+ all2all ops |
| Quadratic complexity w.r.t. sequence | No | No | Yes |

> [!TIP]
> The core difference from standard MoE (e.g., Switch Transformer) is that experts here are applied *per token*, meaning each of the $T$ tokens has its own set of routing decisions and per-token weight matrices, rather than a single routing decision per sequence position.

## Scaling Laws

The paper identifies empirical scaling laws for their architecture:

- **Balanced dimension expansion**: Beyond 1B parameters, expanding hidden dimension $D$, number of tokens $T$, and depth $L$ proportionally yields better AUC gains than expanding a single dimension.
- **Data requirement grows with model size**: Scaling from 500M → 2B parameters requires 60 days of training samples versus 14 days for 30M → 90M, indicating a roughly linear relationship between parameter count and required data volume.

# Experiments

- **Dataset (E-commerce)**: ~400 million records per day over 2 years; 500+ distinct features; task = CTR and CVR prediction
- **Dataset (Douyin Ads)**: ~300 million records per day; advertising click/conversion task
- **Dataset (Douyin Live Streaming)**: ~17 billion records per day; revenue/viewing duration task
- **Hardware**: 64 GPUs for e-commerce baseline; 256 GPUs for ads and live streaming large-scale experiments
- **Optimizer**: Adagrad for both dense (lr = 0.01) and sparse (lr = 0.05) parameters
- **Model scales tested**: 7B online (e-commerce), 4B online (ads), 2B online (live streaming); up to 15B offline
- **Sparsity**: 1:2 ratio for Sparse Per-token MoE (50% parameters active)
- **Results (offline vs SOTA baseline ~500M params)**:
  - TokenMixer-Large 7B: +1.20% AUC improvement
  - AUC gain is substantially higher than attention-based models at same FLOP budget
- **Results (online A/B tests vs production baseline)**:
  - E-commerce: +1.66% orders, +2.98% per-capita preview payment GMV
  - Advertising (ADSS): +2.0%
  - Live streaming revenue: +1.4%
