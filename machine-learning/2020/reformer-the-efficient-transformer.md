# Meta Information

- URL: [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. International Conference on Learning Representations (ICLR 2020).

# Reformer: The Efficient Transformer

## Overview

Standard Transformers require $O(L^2)$ memory and computation for attention (where $L$ is sequence length) and store $N$ copies of activations for backpropagation (where $N$ is the number of layers). This becomes prohibitive for long sequences: a 64K-token sequence at batch size 1 requires 16GB just to store the attention matrix $QK^T \in \mathbb{R}^{L \times L}$.

Reformer introduces two techniques to address these bottlenecks:

1. **Locality-Sensitive Hashing (LSH) Attention**: Reduces attention complexity from $O(L^2)$ to $O(L \log L)$ by grouping similar queries/keys and computing attention only within the same hash bucket.
2. **Reversible Residual Layers**: Eliminates the need to store $N$ copies of activations by recomputing them during backpropagation, reducing memory cost from $O(N \cdot L \cdot d_\text{model})$ to $O(L \cdot d_\text{model})$.

**Applicability**: Reformer is designed for practitioners who need to process long sequences (e.g., documents with tens of thousands of tokens, high-resolution images, or long audio sequences) on limited hardware. It is most beneficial when $L$ is large enough that $O(L^2)$ attention is a bottleneck.

---

## 1. Locality-Sensitive Hashing (LSH) Attention

### 1.1 Standard Attention Recap

In standard dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{L \times d_k}$. The $QK^T$ matrix has shape $\mathbb{R}^{L \times L}$, resulting in $O(L^2)$ time and memory.

### 1.2 LSH Attention Mechanism

**Key Insight**: In the softmax, only the largest elements of $QK^T$ contribute significantly. Thus, for each query $q_i$, it suffices to find only the $k$ nearest keys.

**Shared Query-Key Space**: Reformer sets $K = Q$ (i.e., $k_j = q_j / \|q_j\|$). This halves the number of projections and is theoretically well-motivated: the attention weight $q_i \cdot k_j$ is large when $q_i$ and $k_j$ point in the same direction, which is equivalent to $q_i$ and $q_j$ being similar (since $k_j \propto q_j$).

**LSH Hash Function**: For a hash of size $b/2$, a random matrix $R \in \mathbb{R}^{d_k \times b/2}$ is drawn, and:

$$h(x) = \arg\max([xR; -xR])$$

This produces a bucket assignment in $\{1, \ldots, b\}$ such that similar vectors (measured by cosine similarity) are likely assigned the same bucket.

**Multi-Round Hashing**: A single hash has non-zero probability of separating similar vectors. To reduce this risk, $n_r$ different hash functions $\{h^{(1)}, \ldots, h^{(n_r)}\}$ are applied independently, and attention is computed in each round separately.

### 1.3 LSH Attention Algorithm

**Input**: Queries $Q \in \mathbb{R}^{L \times d_k}$, values $V \in \mathbb{R}^{L \times d_v}$

**Algorithm**:
```
1. Apply hash: assign each position i to bucket s_i = h(q_i)
2. Sort positions by (bucket s_i, original position i) → get permutation P
3. Chunk P into segments of length m = 2L/n_buckets
4. For chunk c, let S_i = {chunk c} ∪ {chunk c-1}  (attend to current + previous chunk)
5. Compute attention within each S_i:
   o_i = ∑_{j ∈ S_i, s_j = s_i or (j is neighbor)} softmax(q_i · k_j / √d_k) · v_j
6. Undo sorting permutation to restore original order
```

**Causal Masking Adjustment**: To prevent position $i$ from attending to itself (which would be trivially the most similar position), self-attention is masked out unless no other positions are in the same bucket.

**Complexity**:

| Component | Memory | Time |
|---|---|---|
| Dot-Product Attention | $O(b n_h L^2)$ | $O(b n_h L^2)$ |
| LSH Attention (per round) | $O(b n_h L n_r (4L/n_c)^2 / n_r)$ | Same |
| LSH Attention (total, $n_r$ rounds) | $O(b n_h L n_r c^2)$ where $c = 4L/n_c$ | Same |

With $n_c$ chunks proportional to $L$, the per-chunk attention is $O(1)$ per chunk, giving $O(L \log L)$ overall.

### 1.4 Comparison with Standard Attention

| Property | Standard Attention | LSH Attention |
|---|---|---|
| Complexity | $O(L^2)$ | $O(L \log L)$ |
| Exact computation | Yes | No (approximate) |
| Separate $Q$, $K$ matrices | Yes | No (shared, $K = Q/\|Q\|$) |
| Multiple hash rounds | N/A | Yes (reduces approximation error) |
| Speed on long sequences | Slow (quadratic) | Fast (log-linear) |

> [!NOTE]
> "LSH attention has, to our knowledge, not been directly applied to Transformer attention layers before."

> [!TIP]
> Related work: Sparse Transformers (Child et al., 2019) use fixed sparse attention patterns; Product-Key Attention (Lample et al., 2019) reduces feed-forward memory. Reformer's LSH approach is data-driven: which positions attend to which depends on input content.

---

## 2. Reversible Transformer

### 2.1 Motivation

Standard Transformers with $N$ layers require storing $N$ activation tensors of shape $(b, L, d_\text{model})$ for backpropagation, giving $O(N \cdot b \cdot L \cdot d_\text{model})$ memory. For a 12-layer model with long sequences, this is a significant bottleneck.

### 2.2 Reversible Residual Layers

Reversible residual networks (RevNets; Gomez et al., 2017) allow activation recomputation from the outputs, eliminating the need to store intermediate activations.

**Standard residual block**:
$$y = x + F(x)$$

(irreversible: cannot recover $x$ from $y$ alone)

**Reversible block** (two-stream):

Forward pass:
$$y_1 = x_1 + \text{Attention}(x_2)$$
$$y_2 = x_2 + \text{FeedForward}(y_1)$$

Backward recovery:
$$x_2 = y_2 - \text{FeedForward}(y_1)$$
$$x_1 = y_1 - \text{Attention}(x_2)$$

**Input/Output**: $x_1, x_2 \in \mathbb{R}^{b \times L \times d_\text{model}}$; $y_1, y_2 \in \mathbb{R}^{b \times L \times d_\text{model}}$

> [!IMPORTANT]
> The reversible formulation means activations for all $N$ layers can be discarded after the forward pass; only the final layer outputs $y_1, y_2$ need to be kept. During backpropagation, intermediate activations are recomputed on-the-fly. This reduces activation memory from $O(N)$ to $O(1)$ with respect to layer count.

### 2.3 Chunked Feed-Forward Layers

Feed-forward layers process each position independently ($\text{FF}(x) = W_2 \max(0, W_1 x + b_1) + b_2$), so computation can be chunked across positions.

For chunk size $c$, the feed-forward layer is computed as:

$$Y_2 = \bigl[X_2^{(1)} + \text{FF}(Y_1^{(1)}); \;\ldots;\; X_2^{(c)} + \text{FF}(Y_1^{(c)})\bigr]$$

This reduces the peak memory for intermediate vectors from $O(b \cdot L \cdot d_{ff})$ to $O(b \cdot L/c \cdot d_{ff})$, storing only the current chunk's $d_{ff}$-dimensional vectors instead of all $L$ at once.

---

## 3. Full Model Complexity

The combined Reformer model achieves:

| Model | Memory (activations) | Time |
|---|---|---|
| Standard Transformer | $O(b L d_{ff} N)$ | $O(b n_h L^2 N)$ |
| Reformer | $O(b L d_\text{model} + b n_h L n_r c^2)$ | $O(b n_h L n_r c^2 N)$ |

where $b$ = batch size, $n_h$ = attention heads, $L$ = sequence length, $N$ = layers, $n_r$ = hash rounds, $c$ = chunk size.

The critical elimination: Reformer's activation memory does **not** grow with $N$ (due to reversibility), whereas the standard Transformer scales linearly with $N$.

---

# Experiments

## Datasets

| Dataset | Task | Sequence Length | Details |
|---|---|---|---|
| Synthetic (duplication) | Copy second half of sequence | 1024 | 511 symbols per half, alphabet size 128 |
| enwik8 | Language modeling (bits/dim) | 64,000 | Wikipedia character-level data |
| imagenet64 | Image generation | 12,288 | 64×64 images (192 tokens/row × 64 rows) |
| WMT 2014 En-De | Machine translation | <128 | Standard translation benchmark |

## Hardware and Setup

- All experiments run on 8 GPUs or TPU v3 cores
- LSH attention trained with $n_r = 4$ hash rounds by default
- Chunk size $m = 2L / n_\text{buckets}$, with bucket count proportional to $L/2$

## Key Results

**Synthetic Duplication Task** (Table 2):
- Full attention: 100% accuracy
- LSH with 1 hash (train), 8 hashes (eval): near-perfect accuracy
- LSH with 4 hashes (train and eval): 100% accuracy
- Demonstrates that multi-round hashing at eval time recovers accuracy lost from fewer training hashes.

**enwik8 Language Modeling** (64K sequence length):
- 12-layer Reformer: **1.19 bits/dim**
- 12-layer Reformer (tuned): **1.05 bits/dim**

**WMT 2014 En-De Translation** (Reversible Transformer, Table 4):
- Base model (100K steps): 27.6 BLEU
- Base model (500K steps): 28.0 BLEU
- Big model (300K steps): 29.1 BLEU
- Comparable to standard Transformer baselines, confirming reversibility does not degrade quality.

**Speed Comparison** (Figure 5):
- LSH attention speed is roughly flat as $L$ grows (log-linear)
- Standard attention degrades quadratically; at $L = 64{,}000$, Reformer is substantially faster.

## Ablation: Shared QK Space

Testing on enwik8: shared-QK attention does **not** hurt performance compared to separate $Q$, $K$ matrices. In fact, it trains slightly faster on enwik8.

## Ablation: Reversible Layers

Comparing reversible Reformer to standard Transformer with equal parameter count: nearly identical learning curves on WMT translation (Figure 3, right), confirming zero accuracy cost from reversibility.

---

# Summary of Differences from Standard Transformer

| Feature | Standard Transformer | Reformer |
|---|---|---|
| Attention type | Dot-product, full $O(L^2)$ | LSH-based, $O(L \log L)$ |
| $Q$ and $K$ | Separate projections | Shared ($K = Q / \|Q\|$) |
| Activation storage | $O(N)$ copies | $O(1)$ (reversible layers) |
| Feed-forward memory | Full $d_{ff} \times L$ | Chunked, $d_{ff} \times L/c$ |
| Approximation | Exact | Approximate (controllable via $n_r$) |
| Practical limit | ~8K tokens on single GPU | 64K+ tokens feasible |

> [!CAUTION]
> LSH attention introduces approximation error. With a small number of hash rounds, some relevant key-query pairs may fall into different buckets and be missed. The paper shows 4 rounds suffices empirically, but this may vary with sequence structure.

**Code**: https://github.com/google/trax/tree/master/trax/models/reformer
