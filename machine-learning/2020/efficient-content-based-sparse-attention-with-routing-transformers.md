# Meta Information

- URL: [Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/abs/2003.05997)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Roy, A., Saffar, M., Vaswani, A., & Grangier, D. (2021). Efficient Content-Based Sparse Attention with Routing Transformers. *Transactions of the Association for Computational Linguistics*, 9, 53–68.

# Efficient Content-Based Sparse Attention with Routing Transformers

## Problem

Standard Transformer self-attention has $O(n^2 d)$ time and memory complexity for sequence length $n$ and hidden dimension $d$. This quadratic scaling makes it prohibitive to process long sequences (e.g., $n > 4096$) in language modeling, image generation, or summarization tasks.

Prior work addressed this with either:
- **Fixed sparse patterns** (Sparse Transformer, Child et al. 2019): strided/local patterns that ignore content
- **LSH-based routing** (Reformer, Kitaev et al. 2020): hash-based grouping that can be inaccurate
- **Recurrence** (Transformer-XL, Dai et al. 2019): limited to fixed-size memory segments

Routing Transformer addresses the fundamental limitation by making sparsity content-dependent: each token dynamically attends to the most semantically relevant tokens, not just nearby ones.

## Method: Routing Attention

### Core Idea

Routing Attention clusters queries $Q \in \mathbb{R}^{n \times d}$ and keys $K \in \mathbb{R}^{n \times d}$ into $k$ groups using shared learnable centroids $\mu \in \mathbb{R}^{k \times d}$. A token $i$ only attends to tokens $j$ assigned to the same cluster as $i$:

$$X'_i = \sum_{j: K_j \in \mu(Q_i),\, j < i} A_{ij} V_j$$

where $\mu(Q_i)$ denotes the centroid closest to query $Q_i$, and $A_{ij}$ is the standard softmax attention weight.

### Connection to MIPS

The method connects to Maximum Inner Product Search (MIPS). After normalizing $Q$ and $K$ to unit vectors:

$$\|Q_i - K_j\|^2 = 2 - 2(Q_i^\top K_j)$$

If both $Q_i$ and $K_j$ are within $\varepsilon$-distance of centroid $\mu$, then:

$$Q_i^\top K_j > 1 - 2\varepsilon^2$$

This guarantees high dot-product (large attention weight) between tokens in the same cluster, making cluster membership a principled approximation to finding high-attention pairs.

### Complexity

With $k$ clusters and uniform cluster sizes $n/k$, each cluster has $(n/k)^2$ pairs, and computing all clusters requires $O(n \cdot (n/k) \cdot d) = O(n^2 d / k)$ operations. Additionally, computing cluster assignments requires $O(nkd)$. Total:

$$O\!\left(nkd + \frac{n^2 d}{k}\right)$$

Setting $k = \sqrt{n}$ minimizes this to $O(n^{1.5} d)$, a significant improvement over $O(n^2 d)$.

### Algorithm: Routing Attention

**Input:** $Q, K, V \in \mathbb{R}^{n \times d}$, centroids $\mu \in \mathbb{R}^{k \times d}$, cluster size $S = n/k$

**Output:** Attended values $X' \in \mathbb{R}^{n \times d}$, updated centroids $\mu'$

```
1. Normalize Q, K to unit ball via LayerNorm (no scale/bias):
     Q̂ = LayerNorm(Q),  K̂ = LayerNorm(K)

2. Compute assignment scores:
     Q_scores = μ · Q̂ᵀ  ∈ ℝ^{k×n}
     K_scores = μ · K̂ᵀ  ∈ ℝ^{k×n}

3. Assign top-S tokens per centroid (balanced clustering):
     Q_idx[c] = top-S indices from Q_scores[c, :]   for c in 1..k
     K_idx[c] = top-S indices from K_scores[c, :]   for c in 1..k

4. Sort Q_idx, K_idx within each cluster (preserves causal order)

5. Gather clustered representations:
     Q'[c] = Q[Q_idx[c], :] ∈ ℝ^{S×d}
     K'[c] = K[K_idx[c], :] ∈ ℝ^{S×d}
     V'[c] = V[K_idx[c], :] ∈ ℝ^{S×d}

6. Compute attention within each cluster:
     A[c] = softmax(Q'[c] · K'[c]ᵀ / √d) ∈ ℝ^{S×S}
     (Apply lower-triangular causal mask if autoregressive)
     X'[c] = A[c] · V'[c] ∈ ℝ^{S×d}

7. Scatter X'[c] back to original positions (average if multi-assignment)

8. Update centroids via exponential moving average (λ = 0.999):
     μ_new[c] ← λ·μ[c] + ((1-λ)/2)·mean(Q̂[Q_idx[c]])
                         + ((1-λ)/2)·mean(K̂[K_idx[c]])
     μ_new[c] ← μ_new[c] / ‖μ_new[c]‖  (re-normalize)
```

> [!NOTE]
> For causal (autoregressive) routing attention, queries and keys share the same sequence of vectors. This avoids explicit future masking complexity: a query $Q_i$ and key $K_j$ with $j > i$ both appear in the same sorted cluster, and the lower-triangular mask on position handles causality correctly.

### Hybrid Architecture

In practice, Routing Transformer uses a **mixed attention** strategy: half of the attention heads use local (window-based) attention, and the other half use routing attention. Local attention handles short-range coherence (syntax, nearby co-reference), while routing attention handles long-range semantics (discourse, distant co-reference). This hybrid design consistently outperforms either mechanism alone.

**Local attention** computes self-attention within a sliding window of size $w$ (e.g., $w = 256$ tokens), with $O(nwd)$ complexity.

## Architecture

| Component | Specification |
|-----------|--------------|
| Embedding | Learned token embeddings + positional encoding (relative positions, Shaw et al. 2018) |
| Self-attention | Half heads: local window attention; Half heads: routing attention |
| Feed-forward | Standard 2-layer MLP with ReLU |
| Normalization | Pre-LN (LayerNorm before attention/FFN sublayers) |
| Routing | Shared centroids across layers; top-$S$ balanced assignment |

## Experiments

### Datasets

| Dataset | Task | Sequence Length | Train Tokens | Notes |
|---------|------|----------------|-------------|-------|
| CIFAR-10 | Image generation | 3,072 (32×32×3) | 50,000 images | Pixel-level autoregressive |
| Wikitext-103 | Language modeling | 3,584 | ~103M tokens | English Wikipedia |
| enwik-8 | Character-level LM | 8,192 | ~90M chars | Wikipedia dump |
| ImageNet-64 | Image generation | 12,288 (64×64×3) | 1.2M images | Pixel-level autoregressive |
| PG-19 | Language modeling | 8,192 | 1.9B tokens | 28,000 Project Gutenberg books |

### Hyperparameters

| Dataset | Layers | Heads | $k$ (clusters) | Window | Optimizer |
|---------|--------|-------|---------------|--------|-----------|
| CIFAR-10 | 12 | 8 | 6 | 512–1024 | Adam (lr=2e-4) |
| Wikitext-103 | 10 | 16 | 16 | 256 | Adam |
| enwik-8 | 24 | 8 | 32 | 256 | Adam |
| ImageNet-64 | 24 | 16 | 8 | 2,048 | Adam |
| PG-19 | 22 | 8 | varied | — | Adafactor (lr=0.01) |

### Key Quantitative Results

| Dataset | Model | Result | Comparison |
|---------|-------|--------|------------|
| Wikitext-103 | Routing Transformer (10L, 16H) | 15.8 ppl | vs. 18.3 (Transformer-XL 18L) |
| enwik-8 | Routing Transformer (12L) | 0.99 bpc | vs. 0.99 (Sparse Transformer 30L) with fewer layers |
| ImageNet-64 | Routing Transformer (24L) | 3.43 bits/dim | vs. 3.44 (Sparse Transformer 48L) with half the layers |
| PG-19 | Routing Transformer (22L) | 33.2 ppl | vs. 33.6 (Compressive Transformer 36L) |

> [!IMPORTANT]
> On every benchmark, Routing Transformer achieves better or equal results with significantly fewer layers than fixed-pattern baselines, demonstrating that content-based routing provides more efficient use of model capacity.

## Analysis

### Local Attention as a Strong Baseline

The paper shows that local attention alone is surprisingly competitive, often outperforming random sparse attention. This validates that **most information relevant to a token is nearby**, but the routing mechanism captures the additional long-range signal that local attention misses.

### Diversity of Attention Patterns

Jensen-Shannon divergence between two local attention heads is very small ($0.0038 \pm 0.0018$ at layer 0), while divergence between a local head and a routing head is near maximum (0.47–0.65 across layers). This confirms that routing and local attention capture fundamentally different information, justifying the hybrid design.

### Wall-Clock Performance

On TPUs, local attention is ~1.7× faster than routing attention due to limited hardware support for sparse operations. The routing mechanism's computational advantage (vs. full attention) is offset by the current TPU kernel implementation, but on CPUs/GPUs with sparse operation support, the theoretical $O(n^{1.5})$ speedup is more realizable.

## Comparison with Similar Approaches

| Method | Attention Pattern | Complexity | Content-Aware |
|--------|------------------|-----------|--------------|
| Full Transformer | All pairs | $O(n^2 d)$ | Yes |
| Sparse Transformer (Child+2019) | Fixed strided + local | $O(n\sqrt{n}d)$ | No |
| Reformer (Kitaev+2020) | LSH bucketing | $O(n \log n \cdot d)$ | Approximate |
| Longformer (Beltagy+2020) | Local + global tokens | $O(n d)$ | Partial |
| **Routing Transformer** | Learned clustering | $O(n^{1.5} d)$ | Yes (exact within cluster) |

> [!TIP]
> For a broader survey of efficient Transformer variants, see [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) (Tay et al., 2020).

## Applicability

- **Who**: Researchers and practitioners working on long-sequence modeling tasks where full attention is computationally infeasible.
- **When**: Sequence length $n > 2048$, especially for language modeling, text generation, image generation, and document-level NLP.
- **Where**: Applicable to any autoregressive (causal) or bidirectional Transformer. Works best when important tokens are not all local (otherwise local attention suffices). Hardware with sparse operation support (GPU/CPU) will see greater practical speedup than TPU.
- **Limitation**: The exponential moving average centroid update is non-differentiable; centroids are maintained as running statistics rather than optimized end-to-end by backpropagation.
