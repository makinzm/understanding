# Meta Information

- URL: [Fast Transformers with Clustered Attention](https://arxiv.org/abs/2007.04825)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Vyas, A., Katharopoulos, A., & Fleuret, F. (2020). Fast Transformers with Clustered Attention. NeurIPS 2020.

# Fast Transformers with Clustered Attention

## Overview

Standard transformer self-attention has $O(N^2)$ complexity in sequence length $N$, which is prohibitive for long sequences. This paper proposes **Clustered Attention**, which groups queries into $C$ clusters and computes attention using cluster centroids instead of individual queries. This reduces complexity to $O(NC)$ for $C \ll N$.

**Who benefits:** Researchers and practitioners training transformers on long sequences (e.g., ASR with 500–800 token utterances, long document NLU). Works as a drop-in replacement for standard attention layers in both training and inference.

## Problem Setup

**Input:**
- Queries $Q \in \mathbb{R}^{N \times d_k}$
- Keys $K \in \mathbb{R}^{N \times d_k}$
- Values $V \in \mathbb{R}^{N \times d_v}$

**Output:**
- Attended values $\hat{V} \in \mathbb{R}^{N \times d_v}$

**Vanilla Attention** (baseline):
$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \quad \hat{V} = AV$$
Complexity: $O(N^2 d_k + N^2 d_v)$.

## Clustered Attention

### Core Algorithm

1. **Cluster** the $N$ queries into $C$ clusters. Represent cluster assignment as $S \in \{0,1\}^{N \times C}$, where $S_{ij} = 1$ if query $i$ belongs to cluster $j$.
2. **Compute centroids**:
$$Q^c_j = \frac{\sum_i S_{ij} Q_i}{\sum_i S_{ij}} \in \mathbb{R}^{d_k}$$
3. **Compute centroid attention**:
$$A^c = \text{softmax}\!\left(\frac{Q^c K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{C \times N}$$
4. **Compute centroid values**:
$$\hat{V}^c_j = \sum_l A^c_{jl} V_l \in \mathbb{R}^{d_v}$$
5. **Broadcast** to cluster members:
$$\hat{V}_i = \hat{V}^c_j \quad \text{for all } i \text{ assigned to cluster } j$$

**Resulting complexity:** $O(NC d_k + CN d_v)$, linear in $N$ for fixed $C$.

### Improved Clustered Attention (Top-$k$ Keys)

The basic variant assigns identical outputs to all members of a cluster, which may lose per-query specificity. The improved variant corrects this by recomputing exact dot products for the top-$k$ highest-attention keys per cluster.

**Algorithm:**

1. From $A^c$, identify the top-$k$ key indices for cluster $j$:
   $T_j = \text{top-}k\text{ indices of } A^c_{j,\cdot}$, encoded as $T \in \{0,1\}^{C \times N}$.
2. For each query $i$ in cluster $j$, compute the corrected attention:
$$A^t_{il} = \begin{cases}
\hat{m}_j \cdot \dfrac{\exp(Q_i K_l^\top / \sqrt{d_k})}{\sum_{r: T_{jr}=1} \exp(Q_i K_r^\top / \sqrt{d_k})} & \text{if } T_{jl} = 1 \\[6pt]
A^c_{jl} & \text{otherwise}
\end{cases}$$
where $\hat{m}_j = \sum_i T_{ji} A^c_{ji}$ is the probability mass on the top-$k$ keys.

**Additional cost:** $O(Nk \cdot \max(d_k, d_v))$, making total complexity $O(NC + Nk)$.

> [!IMPORTANT]
> Proposition 2 proves that the improved clustered attention always achieves smaller or equal $L_1$ distance from the true attention than the basic clustered attention:
> $$\|A^t_i - A_i\|_1 \leq \|A^c_j - A_i\|_1$$
> This monotonic improvement guarantee ensures no degradation from adding the top-$k$ correction.

### Approximation Quality (Proposition 1)

If two queries are close in Euclidean distance ($\|Q_i - Q_j\|_2 \leq \varepsilon$), their attention distributions are also close:
$$\|\text{softmax}(Q_i K^\top) - \text{softmax}(Q_j K^\top)\|_2 \leq \varepsilon \|K\|_2$$

This justifies using centroid attention to approximate member queries: the closer the cluster members to the centroid, the better the approximation.

## Clustering Procedure

Clustering is performed using **Locality-Sensitive Hashing (LSH)** followed by K-Means:

1. **LSH via random projections:** Project queries onto $B$ random binary hash bits to get Hamming codes.
   Complexity: $O(N d_k B)$
2. **K-Means in Hamming space:** Run $L$ Lloyd iterations to refine clusters.
   Complexity: $O(NCL + CBL)$
3. **Total clustering overhead:** $O(N d_k B + NCL + CBL)$

The number of clusters $C$ is treated as a hyperparameter controlling the speed-accuracy trade-off.

## Complexity Comparison

| Method | Time | Space |
|--------|------|-------|
| Vanilla Attention | $O(N^2)$ | $O(N^2)$ |
| Clustered Attention | $O(NC)$ | $O(NC)$ |
| Improved Clustered (top-$k$) | $O(NC + Nk)$ | $O(NC)$ |
| Reformer (LSH) | $O(N \log N)$ | $O(N \log N)$ |
| Linformer | $O(Nk)$ | $O(Nk)$ |

For fixed $C$ and $k$ (independent of $N$), both variants are **linear** in $N$.

## Comparison with Related Algorithms

| Algorithm | Clustering Strategy | Extra Parameters | Heterogeneous Q/K |
|-----------|--------------------|--------------------|-------------------|
| Clustered Attention (this) | Data-driven K-Means | None | Yes |
| Reformer | Random LSH hash | None | No (Q = K required) |
| Set Transformer | Learnable inducing points | Yes (inducing points) | Yes |
| Sparse Transformer | Fixed factorization patterns | None | Yes |
| Linformer | Linear projection of keys | Yes (projection matrix) | Yes |

> [!NOTE]
> Unlike Reformer, clustered attention does **not** require queries and keys to be identical (i.e., it supports encoder–decoder cross-attention). It uses learned clustering rather than random hash collisions, producing higher-quality cluster groups.

# Experiments

- **Datasets:**
  - Wall Street Journal (WSJ) ASR: ~780 tokens average sequence length
  - Switchboard (SWB) ASR: ~534 tokens average sequence length
  - GLUE benchmark: 10 NLU classification tasks, max 128 tokens
  - SQuAD: Question answering, max 384 tokens
- **Hardware:** NVIDIA GTX 1080 Ti
- **Comparison:** Vanilla attention, Reformer, Sinkhorn attention, Linformer

**Key Results:**
- Improved clustered attention achieves lower phone/word error rates than all baselines on WSJ and Switchboard at equal wall-clock computational budgets (2× faster per epoch than vanilla transformer).
- Approximating RoBERTa on GLUE and SQuAD with only $C = 25$ clusters (10–20% of sequence length) produces no measurable degradation on most tasks, confirming that a small number of clusters captures the dominant attention patterns in pre-trained models.
- Memory scales linearly with $N$ for fixed $C$, outperforming vanilla and Reformer in per-element memory at long sequence lengths.
