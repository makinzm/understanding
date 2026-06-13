# Meta Information

- URL: [Flash-KMeans: Fast and Memory-Efficient Exact K-Means](https://arxiv.org/abs/2603.09229)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yang, S., Xi, H., Zhao, Y., Li, M., Fan, X., Zhang, J., Cai, H., Lin, Y., Li, X., Keutzer, K., Han, S., Xu, C., & Stoica, I. (2026). Flash-KMeans: Fast and Memory-Efficient Exact K-Means. arXiv:2603.09229.

# Flash-KMeans: Fast and Memory-Efficient Exact K-Means

## Overview

K-means clustering—based on Lloyd's iterative algorithm—has historically been used as an offline processing step, but modern AI pipelines increasingly invoke it online and at scale: e.g., KV cache compression in LLMs, sparse attention token routing, and semantic deduplication at web scale. This paper identifies two GPU-level bottlenecks that prevent existing implementations from meeting these demands, and proposes kernel-level innovations to resolve each. The resulting system, **flash-kmeans**, achieves mathematically exact output (no approximation) with up to **17.9× end-to-end speedup** over the best prior GPU baselines.

**Who benefits:** Researchers and engineers building LLM serving systems, video generation pipelines, or embedding retrieval infrastructure where k-means is invoked as a high-frequency operator (not a one-off offline step).

> [!TIP]
> The analogy to FlashAttention is central: just as FlashAttention avoids materializing the $N \times N$ attention matrix in HBM, FlashAssign avoids materializing the $N \times K$ distance matrix.

## Background: Lloyd's Algorithm

Standard k-means solves:

```math
\begin{align}
  \min_{a \in \{1,\ldots,K\}^N,\, C} \sum_{i=1}^N \|x_i - c_{a_i}\|_2^2
\end{align}
```

where $x_i \in \mathbb{R}^d$ are data points and $c_k \in \mathbb{R}^d$ are cluster centroids. Each iteration has two stages:

**Assignment stage** — compute distance matrix $D \in \mathbb{R}^{N \times K}$ and find nearest centroid:

```math
\begin{align}
  D_{ik} &= \|x_i - c_k\|_2^2 = \|x_i\|_2^2 + \|c_k\|_2^2 - 2x_i^\top c_k \\
  a_i &= \arg\min_k D_{ik}
\end{align}
```

**Centroid update stage** — aggregate assigned points:

```math
\begin{align}
  n_k &= \sum_{i=1}^N \mathbb{I}[a_i = k] \\
  s_k &= \sum_{i=1}^N \mathbb{I}[a_i = k]\, x_i \\
  c_k &\leftarrow s_k / n_k
\end{align}
```

### Two Core Bottlenecks

**Bottleneck 1 — Distance matrix materialization (IO-bound assignment):**
The intermediate matrix $D \in \mathbb{R}^{N \times K}$ must be written to and read from HBM (High Bandwidth Memory). Under representative settings ($N=65536$, $K=1024$, $d=128$, batch size $B=32$), the distance computation itself takes ~2.6 ms but the HBM round-trips for $D$ take ~23 ms. Useful input/output is only $\Theta(Nd + Kd)$, but $D$ imposes $2 \cdot \Theta(NK)$ additional HBM traffic.

**Bottleneck 2 — Atomic write contention in centroid update:**
Standard aggregation performs one atomic add per data point per dimension: $O(Nd)$ scatter-style atomics with irregular, non-coalesced access patterns. On an NVIDIA H200, this achieves only 50 GB/s effective bandwidth, far below the hardware's reduction bandwidth ceiling.

## Method

### FlashAssign: Materialization-Free Assignment

**Core idea:** Fuse distance computation and row-wise argmin into a single streaming kernel so the full $N \times K$ matrix $D$ is never written to HBM.

For each data point $x_i$, maintain running state in on-chip registers:
- $m_i \in \mathbb{R}$: running minimum distance, initialized to $+\infty$
- $a_i \in \mathbb{Z}$: running best centroid index, initialized to $-1$

Scan centroids in tiles of size $B_K$; compute local tile distances on-chip; compare against running state; update registers. Use 2D tiling over both points ($B_N$) and centroids ($B_K$), plus double-buffered asynchronous prefetch to overlap HBM loads with computation.

**Algorithm (FlashAssign):**

```
Input:  X ∈ ℝ^{N×d}, C ∈ ℝ^{K×d}, tile sizes B_N, B_K
Output: a ∈ {1,...,K}^N

1. Precompute norms ||x_i||^2 for all i.
2. For each point tile X_tile of size B_N (in parallel):
   a. Initialize running state: m ← +∞, a ← -1
   b. Prefetch first centroid tile C_tile^(0) from HBM into on-chip buffer
   c. For t = 0 to ceil(K/B_K) - 1:
      i.   If t+1 < ceil(K/B_K), prefetch C_tile^(t+1) into alternate buffer
      ii.  Compute local distances for (X_tile, C_tile^(t)) on-chip
      iii. Compute tile-local minima (m_tilde, a_tilde)
      iv.  Update running state: m ← min(m, m_tilde); update a with corresponding index
      v.   Swap double buffers
   d. Write final assignments a for X_tile to HBM
```

**IO complexity improvement:**

| Stage | Standard kernel | FlashAssign |
|---|---|---|
| HBM reads | $\Theta(Nd + Kd + NK)$ | $\Theta(Nd + Kd)$ |
| HBM writes | $\Theta(NK + N)$ | $\Theta(N)$ |
| Net HBM traffic | $O(NK)$ dominant | $O(Nd + Kd)$ dominant |

### Sort-Inverse Update: Low-Contention Centroid Aggregation

**Core idea:** Sort the 1D assignment vector $a \in \{1,\ldots,K\}^N$ via argsort to obtain a permutation `sorted_idx` and sorted sequence $a^{\text{sorted}}$. Points assigned to the same cluster become contiguous segments. Gather features into contiguous segments for on-chip partial reduction, then perform a single atomic merge per segment boundary—rather than one atomic per point.

> [!NOTE]
> The heavy point matrix $X \in \mathbb{R}^{N \times d}$ is **not** physically permuted. Only the lightweight index vector `sorted_idx` (size $N$) is sorted. Features are gathered on-the-fly using the index vector.

**Algorithm (Sort-Inverse Update):**

```
Input:  X ∈ ℝ^{N×d}, a ∈ {1,...,K}^N, chunk size B_N
Output: Updated centroids C ∈ ℝ^{K×d}

1. sorted_idx ← argsort(a)
2. Construct a_sorted[j] ← a[sorted_idx[j]]
3. Initialize s ← 0_{K×d}, n ← 0_K
4. For l = 0 to N-1 step B_N:
   a. r ← min(l + B_N, N)
   b. Load a_sorted[l:r] and sorted_idx[l:r]
   c. Identify contiguous segments of identical cluster ids in a_sorted[l:r]
   d. For each segment (u, v, k):
      i.  Gather features: {X[sorted_idx[j]] : u <= j < v}
      ii. Compute partial sum delta_s_k and count delta_n_k on-chip
      iii. atomic_add(s[k], delta_s_k);  atomic_add(n[k], delta_n_k)
5. For k = 1 to K: c[k] ← s[k] / n[k]
```

**Atomic operation reduction:**

| Method | Atomic operations |
|---|---|
| Standard scatter | $O(Nd)$ |
| Sort-Inverse Update | $O\!\left(\left(K + \lceil N/B_N \rceil\right) d\right)$ |

For large $N$ and moderate $B_N$, this reduces atomics by orders of magnitude, raising effective HBM bandwidth from ~50 GB/s toward hardware limits.

### Algorithm-System Co-design

**Out-of-core chunked streaming:** When the full dataset exceeds GPU VRAM, data is partitioned into host-memory chunks. CUDA streams with a double-buffer pattern overlap PCIe host-to-device transfers with GPU k-means computation, bounding peak GPU memory footprint and hiding transfer latency.

**Cache-aware compilation heuristic:** Triton/CUDA kernels must be compiled for specific tile sizes. Exhaustive auto-tuning searches hundreds of configurations and scales poorly with input shape. The proposed heuristic analytically derives near-optimal tile sizes from hardware L1/L2 cache sizes and problem shape ($N$, $K$, $d$), achieving compilation in under 2.5 seconds with less than 0.3% runtime degradation versus exhaustive oracle tuning.

## Comparison with Related Methods

| System | Assignment approach | Update approach | Memory footprint | Scalability |
|---|---|---|---|---|
| Standard PyTorch / fast_pytorch_kmeans | Materialized $N \times K$ matrix | Atomic scatter | $O(NK)$ HBM | OOM at large $N, K$ |
| FAISS | Tiled but still materializes sub-matrices | BLAS-based | $O(NK)$ HBM | 200× slower than flash-kmeans |
| cuML | GPU BLAS with materialization | Atomic | $O(NK)$ HBM | 33× slower |
| fastkmeans | Partial IO optimization | Atomic | Reduced | 17.9× slower at best |
| **Flash-KMeans** | **FlashAssign ($O(Nd+Kd)$ IO)** | **Sort-Inverse Update** | **No $N \times K$ buffer** | **1B points supported** |

> [!IMPORTANT]
> Flash-KMeans preserves **mathematically exact** k-means output—no approximation, sampling, or early termination. The speedups come entirely from IO and compute efficiency, not algorithmic shortcuts.

## Experiments

- **Dataset:** Synthetic benchmarks with varied $N$ (up to $10^9$), $K$ (up to 65536), $d$ (up to 512), batch size $B$ (up to 32). Workload shapes are representative of LLM and video generation model use cases.
- **Hardware:** NVIDIA H200 GPU, CUDA 12.8
- **Baselines:** fast_pytorch_kmeans, fastkmeans (Clavié and Warner, 2025), NVIDIA cuML, FAISS

**Key results:**

| Setting | Speedup | Notes |
|---|---|---|
| $N=8\text{M}, K=1024$ | **17.9×** over fast_pytorch_kmeans | 94.4% latency reduction |
| $N=1\text{M}, K=64\text{K}, d=512$ | **>5.4×** over fastkmeans | Standard PyTorch OOM |
| $B=32$ small batches | **15.3×** | High-batch LLM inference scenario |
| $N=10^9, K=32768, d=128$ | **6.3×** vs fastkmeans out-of-core | PyTorch OOM |
| $N=400\text{M}, K=16384$ | **10.5×** | 41.4 s vs 261.8 s per iteration |
| vs cuML | **33×** | — |
| vs FAISS | **200×** | — |

**Kernel-level breakdown:**

| Kernel | Configuration | Speedup |
|---|---|---|
| FlashAssign | $N=1\text{M}, K=8192$ | **21.2×** (122.5 ms → 5.8 ms) |
| Sort-Inverse Update | $B=1, N=33\text{M}, K=4096$ | **6.3×** |

**Time-to-first-run with dynamic shapes:** Cache-aware heuristic completes in <2.5 s vs >325 s for exhaustive tuning — a **175× reduction** in compilation overhead with <0.3% runtime cost.
