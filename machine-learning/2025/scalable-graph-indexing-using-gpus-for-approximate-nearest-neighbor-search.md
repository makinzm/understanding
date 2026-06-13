# Meta Information

- URL: [Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2508.08744)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, Z., Ke, X., Zhu, Y., Yu, B., Zheng, B., & Gao, Y. (2025). Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search. SIGMOD 2026.

# Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search

## Overview

**Tagore** is a GPU-accelerated library for constructing graph-based indexes for approximate nearest neighbor search (ANNS). It targets database engineers, ML platform teams, and vector search practitioners who need to (re)build large-scale graph indexes — especially when index freshness requires frequent rebuilds (e.g., nightly). On million-scale datasets it outperforms the state-of-the-art GPU method CAGRA by 1.32×–6.39× and CPU methods by 36×–112×; on billion-scale datasets it reduces construction time from ~14 hours (DiskANN) to under 3 hours.

**Applicability**:
- Systems requiring nightly index rebuilds on datasets of 1M–1B vectors
- Multi-GPU deployments seeking near-linear scaling
- Workloads with high-dimensional vectors (Color: 282-d, Gist: 960-d) where distance computation is the bottleneck

## Background: Refinement-Based Graph Indexing

Refinement-based ANNS graph construction follows two phases:

1. **Initialization**: Build a k-nearest-neighbor (k-NN) graph where each node stores its $k$ closest neighbors. Input: $N$ vectors $X \in \mathbb{R}^{N \times d}$. Output: graph $G$ with adjacency list of size $k$ per node.
2. **Pruning**: Refine the k-NN graph by removing redundant edges and reconnecting to improve navigability. Output: final index graph with controlled out-degree.

The bottleneck on GPUs is that existing initialization methods (e.g., NN-Descent ported from CPU) rely on node-specific sampling that creates irregular memory access patterns — killing GPU warp efficiency. The existing pruning strategies (NSG, Vamana, NSSG, DPG, CAGRA) each have serial data dependencies or severe workload imbalance that prevent naive GPU parallelization.

## GNN-Descent: Two-Phase GPU-Optimized k-NN Initialization

GNN-Descent replaces the standard NN-Descent algorithm with a two-phase strategy tuned for GPU execution.

**Phase 1** (iterations $0$ to $it_1$, coarse convergence):

All nodes in a warp share the same random sample batch. This lets the initialization matrix multiply over a shared block, executing on Tensor Cores with high utilization:

```
for each iteration t in [0, it_1):
    sample shared batch B ← random subset of N nodes (size = block_size)
    for each node u in warp:
        compute distances d(u, b) for all b ∈ B via Tensor Core matmul
        update neighbor list N(u) with any closer neighbors found
```

**Phase 2** (iterations $it_1$ to $it_1 + it_2$, fine-grained refinement):

Each node samples independently from its own top-$m$ neighbors' lists, with non-revisitation enforcement to avoid redundant distance calculations:

```
for each iteration t in [it_1, it_1+it_2):
    for each node u:
        sample S_u ← top-m neighbors of u (node-specific)
        compute distances between S_u members
        update N(u) with improvements found
        mark processed pairs as visited (no re-computation)
```

**Lock-free Merge**: When updating neighbor lists, instead of using atomic locks, each thread computes its insertion rank via binary search over sorted distances in shared memory. This allows multiple threads to write simultaneously without synchronization barriers.

> [!IMPORTANT]
> Phase 1 exploits rapid early-stage convergence where shared batches cover the most common close neighbors. Phase 2 switches to node-specific sampling only when global batches stop improving the outlier nodes.

## CFS Pruning: Collect-Filter-Store Framework

CFS abstracts five distinct graph pruning strategies into a single three-stage GPU pipeline, enabling code reuse and GPU-specific kernel optimization.

### Stage 1: Collect (Candidate Gathering)

Each strategy uses a different expansion mode to gather candidate neighbors:

| Strategy | Collection Mode | Description |
|----------|----------------|-------------|
| NSG | Path | Run greedy search from navigation node; collect all nodes on the search path |
| Vamana | Path | Same as NSG with relaxation parameter $\alpha$ |
| DPG | 1-hop | Directly use current 1-hop neighbor set |
| NSSG | 2-hop | Expand to 2-hop neighbors for broader candidates |
| CAGRA | 2-hop | Select detourable paths from 2-hop expansion |

### Stage 2: Filter (Edge Selection)

Each strategy applies a different geometric criterion to select which candidates to keep as neighbors. Let $p$ be the query node, $C$ the candidate set, and $R$ the set of already-retained neighbors:

| Strategy | Filter Condition to keep $p'$ | Criterion Type |
|----------|-------------------------------|---------------|
| NSG | $\text{dis}(p, p') < \min_{p^* \in R} \text{dis}(p^*, p')$ | Distance-based |
| Vamana | $\text{dis}(p, p') < \alpha \cdot \text{dis}(p^*, p')$ for any $p^* \in R$ | Relaxed distance |
| NSSG | $\delta(\overrightarrow{pp^*}, \overrightarrow{pp'}) > \gamma$ for all $p^* \in R$ | Angle-based |
| DPG | Maximize $\sum_{i < j} \delta(\overrightarrow{pp_i}, \overrightarrow{pp_j})$ | Angular dispersion |
| CAGRA | $\max(\text{dis}(p_i, p_k), \text{dis}(p_k, p_j)) < \text{dis}(p_i, p_j)$ | Rank-based (detourability) |

where $\delta(\cdot, \cdot)$ denotes the angle between two vectors and $\gamma = 60°$ is the NSSG threshold.

### Stage 3: Store

Selected neighbors are written from GPU shared memory to global GPU memory, then persisted to CPU RAM or disk.

### GPU Kernel Specialization

Two custom kernels handle the two parallelism paradigms arising from Filter:

**Parallel Increment Kernel** (for NSG, Vamana, NSSG, DPG — serial dependency):

These strategies validate candidates sequentially against an accumulating retained set $R$. A direct GPU port would serialize across candidates. The kernel instead uses wavefront-parallel computing:

```
for each wavefront batch W of candidates:
    all warps in W independently check their candidate against current R
    if condition met: tentatively add to R
    synchronize wavefront: resolve conflicts by distance ranking
    finalize R updates, advance to next wavefront
```

**Parallel Balance Kernel** (for CAGRA — workload imbalance):

CAGRA's detourability check tests whether a "shortcut" node $p_k$ exists such that both $\text{dis}(p_i, p_k) < \text{dis}(p_i, p_j)$ and $\text{dis}(p_k, p_j) < \text{dis}(p_i, p_j)$. The number of candidates to check varies widely per node. The kernel pairs candidates $(p_i, p_j)$ where $\text{pos}_i + \text{pos}_j = k + 2$, distributing two warps per pair so aggregate warp work is balanced.

## GPU-CPU-Disk Asynchronous Framework for Billion-Scale Data

When the dataset exceeds GPU memory (RTX 4090: 24 GB), a three-tier pipeline manages data flow:

```
Dataset (Disk)
    │ k-means partitioning → 400 clusters
    ↓
GPU (24 GB)
    │ Build local index per cluster (GNN-Descent + CFS)
    ↓ async transfer
CPU RAM (251 GB)
    │ Merge partial neighbor lists from overlapping clusters
    ↓ async flush
Disk
    │ Store finalized index segments
    ↓
Final merged index
```

**k-means Partitioning**: The dataset is split into 400 clusters (vs. DiskANN's 40) to fit each cluster within GPU memory. Each cluster contains ~5M vectors for the 1B-vector datasets. This finer partitioning slightly degrades cross-cluster neighbor relationships.

### Cluster-Aware Caching

Merging requires loading a node's neighbors from multiple clusters. Naive sequential cluster processing causes repeated cache misses. Tagore models inter-cluster relationships as a weighted graph $CG(C, E)$ where:

```math
\begin{align}
  W(C_i, C_j) = |\text{nodes shared between cluster } C_i \text{ and } C_j|
\end{align}
```

The dispatching algorithm greedily selects which cluster to process next to maximize cumulative edge weight within the current RAM buffer (window of $n$ clusters):

```
Initialize buffer buf with one cluster
While unprocessed clusters remain:
    if |buf| < capacity:
        C_next ← argmax_{C ∈ C \ buf} Σ_{C_j ∈ buf} W(C, C_j)
        add C_next to buf
    else:
        (C*, C') ← argmax_{C∈C\buf, C'∈buf} Σ_{C_k ∈ buf\{C'}} W(C*, C_k)
        evict C' from buf, load C*
    process next cluster in buf
```

> [!NOTE]
> Cache hit rates: 61.6% on BIGANN and 67.2% on Deep-1B, versus 24.9% and 19.1% for the Gorder-based reordering baseline. This translates to 1.79×–2.01× reduction in merge I/O overhead.

## Comparison with Similar Methods

| Method | Platform | Initialization | Pruning | Billion-Scale |
|--------|----------|---------------|---------|---------------|
| NN-Descent | CPU | CPU-native sampling | N/A | No |
| CAGRA | GPU | Random + NN-Descent | CAGRA only | No |
| NSG | CPU | Knng-based | Distance (serial) | No |
| Vamana (DiskANN) | CPU+Disk | Random | Relaxed distance | Yes (I/O bound) |
| NSSG | CPU | NSG-based | Angle (serial) | No |
| DPG | CPU | Knng-based | Angular dispersion | No |
| **Tagore** | **GPU+CPU+Disk** | **GNN-Descent (2-phase)** | **CFS (all above)** | **Yes (cache-aware)** |

**Key differences from CAGRA**: CAGRA focuses on pruning efficiency but uses standard NN-Descent for initialization, which stalls in late iterations. GNN-Descent's phase-switch eliminates this by shifting from shared to node-specific sampling. On CAGRA-pruned indexes, Tagore's initialization alone yields 4.65× average speedup.

**Key differences from DiskANN**: DiskANN uses 40 coarse clusters and performs sequential I/O during merging. Tagore uses 400 clusters (GPU-memory-fit) and the cluster-aware cache reduces merge I/O by ~1.9×.

# Experiments

- **Datasets (million-scale)**:
  - SIFT: 1M vectors, 128-d, 0.48 GB
  - Deep-1M: 1M vectors, 96-d, 0.36 GB
  - UKBench: 1M vectors, 128-d, 0.48 GB
  - Color: 1M vectors, 282-d, 1.05 GB
  - Gist: 1M vectors, 960-d, 3.58 GB
- **Datasets (billion-scale)**:
  - BIGANN: 1B vectors, 128-d, 119.21 GB
  - Deep-1B: 1B vectors, 96-d, 357.63 GB
- **Hardware**: NVIDIA GeForce RTX 4090 (24 GB), Intel Xeon Silver 4316 @2.30 GHz, 251 GB RAM, Samsung 980 SSD 1 TB; CUDA 12.2, Ubuntu 22.04
- **Optimizer**: Not applicable (index construction, no gradient-based training)
- **Graph parameters**: $k = 32$ neighbors; NSSG $\gamma = 60°$; Vamana $\alpha = 1.2$
- **Baselines**: CAGRA, GANNS, SymQC (GPU/CPU), NSG, NSSG, DPG, Vamana (CPU)
- **Results**:
  - Million-scale: 1.32×–6.39× faster than CAGRA; 36.23×–112.79× faster than CPU methods; no accuracy degradation
  - Billion-scale: 6.68× (BIGANN) and 6.00× (Deep-1B) faster than DiskANN; construction under 3 hours
  - Pruning-only speedup: CAGRA 16.57×–19.43×; NSG/Vamana 22.15×–27.94×; NSSG 74.36×–127.95×; DPG 179.04×–342.48×
  - Multi-GPU scaling: ~2× speedup with 2 GPUs, ~4× with 4 GPUs (near-linear)
  - Minor quality gap at billion-scale: 0.5%–1.4% lower recall than DiskANN due to finer cluster partitioning
