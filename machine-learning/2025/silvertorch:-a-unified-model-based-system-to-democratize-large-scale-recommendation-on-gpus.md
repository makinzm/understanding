# Meta Information

- URL: [SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs](https://arxiv.org/abs/2511.14881)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Xue, B., Wu, H., Chen, L., Yang, C., et al. (2025). SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs. arXiv:2511.14881.

# SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs

SilverTorch is a GPU-native recommendation serving system developed by Meta Platforms and Fireworks AI that replaces the traditional multi-service CPU architecture (separate ANN indexing, feature filtering, and ranking services) with a single unified model executing entirely on GPU. The system targets two core production stages: **retrieval** (finding thousands of relevant items from a pool of millions) and **Early-Stage Ranking (ESR)** (re-ranking those thousands of items). It serves hundreds of models supporting billions of daily active users at Meta.

## Background and Motivation

Traditional large-scale recommendation pipelines decompose retrieval into independently deployed services:
- A **user tower** GPU service to compute user embeddings.
- A CPU-based **ANN (Approximate Nearest Neighbor) search** service (e.g., Faiss-CPU IVF).
- A CPU-based **feature filtering** service (inverted index) to enforce business constraints (e.g., "only show items in the user's country").
- A separate **early-stage ranking** service with an embedding cache.

This decomposition causes three critical problems:
1. **Version inconsistency**: model updates must be synchronized across independently deployed services, creating correctness risks.
2. **Resource inefficiency**: CPU services cannot batch requests as efficiently as GPU, leading to high hardware costs per query.
3. **Development complexity**: engineers must understand and operate multiple systems, slowing iteration velocity.

SilverTorch addresses all three by encoding ANN search, feature filtering, OverArch scoring, and ESR as differentiable model layers within a single PyTorch/TorchScript model deployed on a single GPU.

## System Architecture

### Two-Phase Operation

#### Publish Phase

Before serving, SilverTorch executes an offline pipeline to materialize indices and quantized models:

1. **Embedding Evaluation**: The root model (item tower) is applied to the full item candidate pool on GPU to produce item embeddings $E \in \mathbb{R}^{N \times d}$, where $N$ is the number of items (up to 80M) and $d = 128$.
2. **ANN Index Building**: KMeans++ clustering partitions embeddings into $C$ centroids. Each embedding is quantized to Int8 (range $[-128, 127]$) using global min/max statistics computed at this stage. Cluster assignments map each item to a centroid.
3. **Bloom Index Construction**: For each item, a M-bit bloom filter signature is constructed using $K$ hash functions applied to its categorical features (see §Bloom Index).
4. **Model Quantization**: User Tower, OverArch layers, and Value Model are converted to BFloat16.
5. **Graph Optimization**: The composed eager-mode model is compiled to a graph representation with graph-level optimizations applied by the Model Optimizer.
6. **Snapshot Export**: The final artifact is a model snapshot with a C++ runtime predictor, deployable to a single GPU server.

#### Serving Phase

A single forward pass of the SilverTorch model processes one recommendation request:

1. **User Embedding Computation**: User Tower receives user features and produces user embedding $q \in \mathbb{R}^{d}$.
2. **Feature Filtering** (Bloom Index Layer): Given query features $Q = [f_1, f_2, \ldots, f_t]$, the bloom index produces a boolean mask tensor $M \in \{0,1\}^{N}$ marking valid items.
3. **ANN Search** (Fused Int8 Kernel): Using $q$ and mask $M$, the kernel finds the top-$k_0$ most similar items ($k_0 \in [10{,}000, 100{,}000]$) from the masked candidate pool.
4. **OverArch Scoring**: The retrieved $k_0$ item embeddings are passed through learned interaction layers (MLP or self-attention) to re-rank and return the final retrieval result.
5. **Value Model Aggregation** (multi-task): Task-specific scores are combined via a user-defined aggregation function.
6. **ESR Lookup**: For ESR models, pre-computed item embeddings are retrieved from GPU embedding cache and combined with user embeddings for interaction-layer scoring.

### Comparison with Traditional Approaches

| Dimension | CPU-based Multi-Service | SilverTorch |
|---|---|---|
| ANN service | Faiss-CPU IVF (separate server) | Fused Int8 GPU Kernel (in-model layer) |
| Feature filtering | Inverted index on dedicated CPU servers | Bloom Index GPU layer |
| Ranking | Separate ESR service + caching service | Embedding cache as model layer |
| Version sync | Required across 3+ services | Single model artifact |
| Batching | Limited by CPU latency | Full GPU batching |

## Bloom Index: GPU-Optimized Feature Filtering

### Problem Formulation

Given a candidate pool $V = \{V_i = [f_1, f_2, \ldots, f_t]\}$ of items with categorical features and a query $Q = [f_1, f_2, \ldots, f_t]$, feature filtering retrieves:

```math
\begin{align}
  R = \{V_i \in V \mid Q \cap V_i = Q\}
\end{align}
```

i.e., all items that contain all features required by the business constraint.

CPU inverted indexes solve this by maintaining per-feature item lists and computing set intersections, but require large memory (19.8 GB for 40M items) and high inter-service latency.

### Bloom Filter Construction

Each item $V_i$ is encoded as an $M$-bit bloom filter $VB_i$ using $K$ hash functions:

```math
\begin{align}
  \text{bit}_{h,j} = \text{hash}_j(f_{i,h}) \bmod N
\end{align}
```

where $f_{i,h}$ is the $h$-th feature value of item $i$, and the resulting bit position is set to 1 in $VB_i$. Similarly, a query bloom filter $QB$ is built from the query features using the same hash functions.

The filtering condition becomes:

```math
\begin{align}
  R = \{V_i \in V \mid QB \mathbin{\&} VB_i = QB\}
\end{align}
```

where $\mathbin{\&}$ is bitwise AND. If the AND equals $QB$ itself, the item contains all required features (with some false-positive rate controlled by $M$ and $K$).

### GPU Parallelization

The bloom index matrix $\mathbf{B} \in \{0,1\}^{N \times M}$ is stored in column-major (transposed) layout so that each column holds one bit position across all items. This enables:
- Processing 64 items simultaneously using a single 64-bit AND instruction (`PTX: and.b64`).
- Iterating only over the $K$ bit positions where $QB$ has a 1-bit, skipping the rest via early termination.

**Memory**: 4.7 GB for $M = 2048$ bits and 40M items, versus 19.8 GB for an inverted index — a 4.21× reduction.

**Latency**: 291–523× speedup over CPU-based inverted indexes.

## Fused Int8 ANN Kernel

### Standard IVF-ANN

A standard IVF (Inverted File) ANN index partitions the embedding space into $C$ clusters. At query time:
1. Compute dot products between query $q \in \mathbb{R}^{d}$ and all $C$ centroid embeddings.
2. Select the top $n_{\text{probe}}$ closest clusters.
3. Gather embeddings of all items in selected clusters into a temporary buffer $T \in \mathbb{R}^{K_{\text{items}} \times d}$.
4. Compute dot products between $q$ and all gathered embeddings.
5. Return top-$k$ results.

Step 3 creates a large intermediate tensor that is problematic at large $k$ and $n_{\text{probe}}$ (Faiss-GPU caps at topk ≤ 1024, nprobe ≤ 2048).

### SilverTorch's Fused Kernel

SilverTorch introduces a **fused index-matmul** operator that eliminates the intermediate tensor: instead of gathering embeddings into a buffer and then computing dot products, it streams item embeddings directly from the embedding table and computes dot products with the batched query on-the-fly:

```
Algorithm: Fused Int8 ANN Kernel
Input: query q ∈ ℝᵈ, centroid embeddings C ∈ ℤ^(C×d) (Int8),
       item embeddings E ∈ ℤ^(N×d) (Int8), cluster assignments,
       bloom mask M ∈ {0,1}^N, n_probe, topk
Output: top-k item indices

1. scores_c ← dot(q, C)                      // (C,) centroid scores
2. probes ← argsort(scores_c)[:n_probe]       // top-n_probe cluster IDs
3. For each cluster c in probes (streamed):
     For each item i in cluster c:
       If M[i] == 0: skip                     // bloom mask applied
       s_i ← dot(q, E[i])                     // streamed, no buffer
       Update top-k heap with (s_i, i)
4. Return top-k heap
```

**Quantization**: Embeddings are stored as Int8 values $\hat{e} \in [-128, 127]$, using global min/max normalization computed during publish. The GPU instruction `dp4a` performs four multiplications and additions per clock, yielding 4× throughput over FP32 at the cost of a small quality loss validated offline.

**Capability**: Supports 64 probes and topk up to 20,000+ without accuracy regression, compared to Faiss-GPU's hard limits.

## OverArch Scoring Layer

Beyond ANN dot-product similarity, SilverTorch supports **OverArch** layers — learned interaction modules applied after ANN retrieval to capture richer user-item relationships:
- **MLP**: Fully connected layers applied to concatenated user/item embeddings from the $k_0$ retrieved candidates.
- **Self-attention stacks**: Capture correlations across retrieved item sessions.
- **Mixture of Logits (MoL)**: Trains multiple logit heads with learned weights per user context; achieved 2.4–28.2% E-Task recall improvement at various scales, with 5.6% improvement at recall@1000.

> [!IMPORTANT]
> OverArch runs on the same GPU within the same forward pass, so adding it does not increase inter-service latency — a key advantage over the CPU multi-service design where an additional scoring service would require a network round-trip.

## Multi-Task Retrieval and Value Model

For platforms serving multiple recommendation objectives simultaneously (e.g., clicks, watches, purchases), SilverTorch natively supports multi-task retrieval:

- The user tower produces task-specific embedding vectors $q_1, q_2, \ldots, q_T \in \mathbb{R}^{d}$ by applying task-specific dense layers to a shared user representation.
- Item embeddings remain shared across tasks ($E$ unchanged).
- The ANN index handles all $T$ task embeddings in a single batched kernel call.
- A **Value Model** (defined as a JSON-like DSL supporting control flow) aggregates per-task scores into a final ranking score.

> [!NOTE]
> "Model designers are able to define customized merge operators such as an intersection of items or a union of items across tasks."

On CPU baselines, adding one task requires linearly scaling CPU server resources. SilverTorch's GPU batching absorbs additional tasks with negligible latency increase.

## Early-Stage Ranking (ESR) Integration

For ESR models, which re-rank thousands of ANN-retrieved candidates using heavier interaction layers, SilverTorch:

1. **Pre-computes** item embeddings during the publish phase and stores them as an **embedding cache** in GPU HBM (high-bandwidth memory).
2. At serving time, replaces the embedding lookup network call with a direct GPU memory read, eliminating the caching service dependency and network round-trip cost.
3. Combines cached item embeddings with the user embedding to feed interaction layers on the same GPU.

This is applicable to teams operating ESR at scales where the full embedding table fits in GPU HBM (e.g., 10M items × 128 dimensions × 4 bytes = ~5 GB, fitting within an A100's 40 GB).

# Experiments

- **Datasets**: 10M-item pool (128-dim embeddings) and 80M-item pool (128-dim embeddings); 5,000 real production recommendation requests as workload.
- **Hardware**: NVIDIA A100 40GB GPUs.
- **Baselines**:
  - *Baseline-Retrieval* (CPU): 1 GPU (user tower) + 2 CPU servers (Faiss-CPU IVF ANN, 64 OpenMP threads) + 4 CPU servers (inverted index filtering).
  - *Baseline-Retrieval-GPU*: Faiss-GPU IVF + forward-index filtering, scatter-gathered across multiple GPUs (Faiss2-Forward4 and Faiss2-Forward6 configurations).
- **Key Results**:
  - SilverTorch-Retrieval achieves **5.6× lower latency** and **23.7× higher throughput** (1,210 QPS vs. 51 QPS) compared to CPU baseline.
  - **13.35× better cost-efficiency** than CPU baseline while improving model accuracy.
  - Bloom Index provides 291–523× speedup over CPU inverted index; memory footprint reduced 4.21× (4.7 GB vs. 19.8 GB at 40M items, 2048-bit config).
  - ANN co-design with bloom mask reduces memory 30× and computation 30× compared to separate operations.
  - OverArch (MoL) improves recall@1000 by 5.6% vs. ANN-only retrieval.
  - Deployed in production serving hundreds of models for billions of daily active users.

> [!TIP]
> For comparison with the underlying ANN building block, see [HNSW (Hierarchical Navigable Small World Graphs)](https://arxiv.org/abs/1603.09320) and [Faiss](https://faiss.ai/) for alternative GPU and CPU ANN approaches.
