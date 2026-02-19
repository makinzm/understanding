# Meta Information

- URL: [Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System](https://arxiv.org/abs/2602.16124)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Jiang Zhang, Yubo Wang, et al. (2026). Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System. arXiv:2602.16124 [cs.IR].

# Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System

## Overview

Large-scale recommendation systems (e.g., video feeds with billions of users) employ a retrieval stage to narrow billions of items to hundreds of candidates before ranking. The dominant approach—**Approximate Nearest Neighbor (ANN) search** over learned embeddings—has two fundamental problems:

1. **Embedding-index decoupling**: Item embeddings and the ANN index are trained separately, so newly created items may have stale or missing index entries until the next offline rebuild.
2. **Per-request ANN cost**: ANN search must run at serving time for every user query, which becomes a computational bottleneck at industrial scale.

MFLI (MultiFaceted Learnable Index) unifies embedding learning and indexing into a single framework and eliminates ANN search at serving time through direct index lookup.

> [!IMPORTANT]
> MFLI is deployed at Meta and demonstrates up to 11.8% recall improvement on engagement tasks, 57.29% improvement on cold-content delivery, 13.5% improvement on semantic relevance, and a 60% QPS throughput gain compared to prior multi-ANN baselines.

---

## Problem Formulation

### Traditional ANN Retrieval (Baseline)

- **Input**: User trigger embedding $u \in \mathbb{R}^d$, pre-built ANN index over item embeddings $\{e_i \in \mathbb{R}^d\}_{i=1}^{N}$
- **Output**: Top-$K$ nearest items by cosine or $\ell_2$ similarity
- **Bottleneck**: ANN search (e.g., HNSW, IVF-PQ) runs per request; index rebuild takes 30+ min, making fresh item recall slow.

### MFLI Retrieval

- **Input at training**: (trigger, candidate) item pairs with co-engagement labels
- **Input at serving**: Trigger item (or user) → its learned indices
- **Output**: Set of candidate items retrieved via index-based lookup, bypassing ANN search

---

## Core Methodology

### Multifaceted Residual Quantization (MF-RQ)

Traditional product/residual quantization maps each item embedding to a flat codebook assignment. MF-RQ introduces a **3D hierarchical codebook** $\{C_1, \ldots, C_L\}$ where each layer $C_l \in \mathbb{R}^{F \times M \times d}$:

- $L$: number of codebook layers (depth)
- $F$: number of facets (parallel quantization channels per item)
- $M$: number of codewords per facet per layer
- $d$: embedding dimension

For each facet $f \in \{1, \ldots, F\}$ at layer $l$, the nearest codeword index is:

$$k_{l,f} = \arg\min_{k} \| r_{l-1}[f] - C_l[f, k] \|_2^2$$

where $r_{l-1}[f] \in \mathbb{R}^d$ is the residual from the previous layer for facet $f$. The item's index assignment is the tuple $(k_{1,f}, \ldots, k_{L,f})$ for each facet, yielding $F$ separate hierarchical indices per item.

> [!NOTE]
> "Instead of a 2D codebook, MF-RQ employs a 3D hierarchical codebook to simultaneously model multiple user engagement patterns."

Each facet captures a different aspect of user interest (e.g., one facet for watch-time engagement, another for explicit reactions). Parallel GPU processing across facets makes this tractable at scale.

### Training Loss

MFLI co-trains the item embeddings and the codebook via two complementary losses:

**Multifaceted loss** (embedding level):

$$\mathcal{L}_\text{MF} = \sum_f w_f \cdot \mathcal{L}_\text{SSM}(u, e_{\text{cand}}[f])$$

**Quantized multifaceted loss** (index level):

$$\mathcal{L}_\text{QMF} = \sum_f w_f \cdot \mathcal{L}_\text{SSM}(u, \hat{e}_{\text{cand}}[f])$$

where $\hat{e}_{\text{cand}}[f]$ is the quantized (codeword-reconstructed) candidate embedding for facet $f$, $w_f$ is a per-facet weight reflecting label type (e.g., view vs. like), and $\mathcal{L}_\text{SSM}$ is the Sampled SoftMax loss.

The total loss is $\mathcal{L} = \mathcal{L}_\text{MF} + \mathcal{L}_\text{QMF}$.

This joint training ensures that learned indices are predictive of relevance—not just a post-hoc compression of fixed embeddings.

### Index Balancing Strategies

Without constraints, residual quantization collapses: popular items dominate certain codewords, leaving others empty. MFLI uses:

**Training-time regularization**:
- **Delayed start**: Codebooks are activated only after warm embeddings stabilize (prevents early collapse).
- **Layer-wise progressive activation**: Lower layers of the codebook are activated before upper layers.
- **Over-utilization penalty**: Regularization term penalizes codewords that accumulate too many items.

**Serving-time Split-and-Merge** (most impactful ablation: $-18.47\%$ recall if removed):
- **Split**: If an index cluster exceeds $B_\text{upp}$ items, apply $k$-means on residuals within that cluster to split into sub-clusters.
- **Merge**: If an index cluster falls below $B_\text{low}$ items, merge it with its nearest neighbor cluster.
- Guarantees $B_\text{low} \leq |\text{cluster}| \leq B_\text{upp}$ at all times.

---

## Indexing System Architecture

### Unified Index Representation

Each item $i$ has $F$ index assignments $(k_{1,f}, \ldots, k_{L,f})$ per facet. To avoid storing sparse multi-key mappings, MFLI converts these into a single integer index via:

$$\text{unified\_index}(i, f) = \text{encode}(k_{1,f}, \ldots, k_{L,f}) \in [0, M^L)$$

This yields a dense **Item-to-Index** tensor of shape $(N, F)$ with $O(1)$ lookup.

The **Index-to-Item** mapping is stored as two tensors:
- A counts tensor: number of items per index
- An item IDs tensor: sorted list of items per index

Both enable $O(1)$ retrieval.

### Real-Time Update Strategy (Delta Snapshots)

| Snapshot Type | Frequency | Contains | Rebalancing |
|---|---|---|---|
| Full snapshot | ~30 min | Entire item pool | Yes (split/merge) |
| Delta snapshot | ~1 min | Newly arrived items | No |

At serving time, both snapshots are queried and results merged. Newly created items (cold content) appear in delta snapshots within ~1 minute, compared to 30+ minutes under traditional ANN rebuilds.

> [!IMPORTANT]
> A mapping layer translates pre-rebalancing indices (in delta snapshot) to post-rebalancing indices (in full snapshot) to maintain consistency after split/merge operations.

### Serving Pipeline (step-by-step)

1. **Index Lookup** ($O(1)$): Given trigger item(s), retrieve their unified indices from the Item-to-Index tensor.
2. **Index Selection**: Sample $K$ indices via multinomial sampling weighted by a frequency histogram $h[\cdot]$ built from the retrieved indices.
3. **Item Retrieval**: For each selected index, retrieve top-$N_k$ items from full+delta snapshots.
4. **Per-Index Reranking**: Within each index cluster, rerank items by embedding similarity to preserve multi-interest coverage across facets.

---

## Comparison with Related Methods

| Method | Index Type | ANN at Serving? | Real-time Updates | Multi-interest |
|---|---|---|---|---|
| HNSW/FAISS (ANN) | Graph/Inverted file | Yes | No (offline rebuild) | No |
| VQIndex | Single-facet codebook | No | Limited | No |
| Semantic IDs (NCI, TIGER) | Text-derived codes | No | No | No |
| MTMH (multi-ANN) | Multiple ANN indices | Yes ($\times F$) | No | Partial |
| **MFLI (ours)** | Multifaceted RQ codebook | **No** | **Yes (~1 min)** | **Yes ($F$ facets)** |

Key differentiators:
- **vs. VQIndex**: MFLI uses multifaceted (3D) codebooks capturing multiple engagement patterns; single-facet removal costs $-13.81\%$ recall.
- **vs. Semantic IDs**: MFLI indices are learned from engagement data, not text; indices evolve with user behavior rather than being fixed at item creation.
- **vs. MTMH**: MFLI replaces $F$ separate ANN searches with direct index lookup, yielding 60% QPS improvement.

---

# Experiments

- **Dataset**: Commercial video recommendation platform with billions of users and items (not publicly named; likely Meta's video feed)
- **Tasks evaluated**:
  - VVC (Video View Count / engagement recall)
  - Cold-content delivery (items with $< 5{,}000$ views)
  - Semantic relevance (T2: topic-to-video)
  - Low-watch-time (LWT) recall
- **Baselines**: Six prior methods including traditional ANN (HNSW/FAISS), VQIndex, and MTMH
- **Hardware**: Not explicitly stated
- **Optimizer**: Not explicitly stated; uses Sampled SoftMax with co-engagement labels
- **Online evaluation**: 7-day A/B test in production

### Key Offline Results (vs. prior state-of-the-art)

| Task | MFLI Improvement |
|---|---|
| VVC recall | +11.76% |
| Cold-content delivery | +57.29% |
| Semantic relevance (T2) | +126.32% (vs. best single-task baseline) |

### Ablation Results

| Component removed | VVC recall change |
|---|---|
| Split-and-merge | $-18.47\%$ |
| Index selection | $-10.11\%$ |
| Delayed start | $-1.83\%$ |
| Facets: 1 (vs. 3) | $-13.81\%$ |

### Online A/B Test Results (7-day production)

| Metric | Change |
|---|---|
| Low-VV items $[0, 5k)$ exposure | $+279\%$ |
| Ultra-fresh content $[0, 6h)$ exposure | $+221\%$ |
| Serving QPS capacity | $+60\%$ |
| Explicit engagement rate | $+0.08\%$ |

### Index Distribution

- 90% of all index clusters contain between 100 and 10,000 items after split-and-merge.
- 98% of clusters remain active throughout 7-day training (no dead codewords).
- Intra-index item semantic similarity substantially exceeds inter-index similarity, validating that codebook clusters are semantically coherent.
