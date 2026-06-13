# Meta Information

- URL: [Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis](https://arxiv.org/abs/2602.11443)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Amanbayev, A., Tsan, B., Dang, T., & Rusu, F. (2026). Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis. arXiv:2602.11443.

# Filtered Approximate Nearest Neighbor Search in Vector Databases

## Overview

Filtered approximate nearest neighbor search (FANNS) extends standard ANN search by combining vector similarity with metadata predicates: find the $k$ most similar vectors to a query $q$ among only those vectors satisfying filter condition $\phi(v) = 1$. This paper surveys the design space of FANNS across three production-grade systems (FAISS, Milvus, pgvector) under two index families (HNSW, IVFFlat), and introduces a new diagnostic metric and benchmark dataset to explain observed performance patterns.

**Applicability:** Engineers building vector databases or choosing ANN libraries for workloads with attribute-constrained queries (e.g., "find similar product embeddings that are in stock and under $50").

## Problem Formulation

Given a dataset $D$ of $N$ vectors $v \in \mathbb{R}^d$, a query vector $q \in \mathbb{R}^d$, a boolean filter predicate $\phi: \mathbb{R}^d \to \{0,1\}$, and retrieval count $k$, FANNS returns:

```math
\begin{align}
  \text{FANNS}(q, \phi, k) = \arg\min_{S \subseteq D,\, |S|=k,\, \forall v \in S:\, \phi(v)=1} \sum_{v \in S} \text{dist}(q, v)
\end{align}
```

The evaluation metric is:

```math
\begin{align}
  \text{Recall}@k = \frac{|\text{retrieved} \cap \text{ground truth top-}k|}{k}
\end{align}
```

## Filtering Strategies

Three strategies map metadata constraints onto index traversal differently:

| Strategy | When filter is applied | Strengths | Weaknesses |
|---|---|---|---|
| **Pre-filtering** | Before distance computation | Exact results; avoids wasted distance calls | Requires bitset construction overhead; graph connectivity can break |
| **Post-filtering** | After ANN returns candidates | Index traversal unmodified | May miss valid neighbors if selectivity is low; top-$k$ can be incomplete |
| **Runtime filtering** | Lazily during traversal | Reduces predicate evaluations | Adds I/O per candidate access; not exposed in in-memory systems |

**Pre-filtering implementation:** A bitset $B \in \{0,1\}^N$ is constructed upfront where $B[i] = \phi(v_i)$. For HNSW, candidates are still traversed for connectivity but only admitted to the result priority queue if $B[i] = 1$. For IVFFlat, entire clusters with no valid vectors can be pruned before distance computation.

## Index Structures

### HNSW (Hierarchical Navigable Small World)

A proximity graph with multiple layers. Layer $\ell$ contains a random subset of nodes with long-range edges; layer 0 contains all nodes with fine-grained local edges. Search begins at the top layer and greedily descends.

- **Construction parameters:** $M$ (max edges per node per layer), $\text{efConstruction}$ (candidate queue size during build)
- **Search parameter:** $\text{efSearch}$ (beam width during query)
- **Input:** $q \in \mathbb{R}^d$; **Output:** approximate top-$k$ neighbors

**FANNS challenge:** When selectivity is low (few valid vectors), HNSW traverses many invalid nodes before finding $k$ valid ones, causing premature termination and recall degradation.

### IVFFlat (Inverted File Index)

Partitions $D$ into $C$ Voronoi clusters via $k$-means. Each vector is assigned to its nearest centroid. At query time, the $\text{nprobe}$ closest clusters are scanned exhaustively.

- **Construction parameter:** $C$ (number of clusters)
- **Search parameter:** $\text{nprobe}$ (number of clusters to scan)
- **Input:** $q \in \mathbb{R}^d$; **Output:** exact top-$k$ within scanned clusters

**FANNS advantage at low selectivity:** If valid vectors concentrate in few clusters, cluster-level pruning skips large portions of $D$ without computing any distances.

## Global-Local Selectivity (GLS) Correlation Metric

Standard filter selectivity $\sigma_g$ only measures what fraction of the dataset passes the filter globally. It does not capture whether valid vectors cluster near the query or are spatially scattered. The paper introduces GLS to quantify this.

**Definitions:**

```math
\begin{align}
  \sigma_g &= \frac{|\{v \in D \mid \phi(v) = 1\}|}{N} \in (0, 1] \quad \text{(global selectivity)}
\end{align}
```

```math
\begin{align}
  \sigma_l(q) &= \frac{|\{v \in \mathcal{N}_q \mid \phi(v) = 1\}|}{k} \in [0, 1] \quad \text{(local selectivity in $k$-neighborhood of $q$)}
\end{align}
```

where $\mathcal{N}_q$ is the exact top-$k$ neighborhood of query $q$ ignoring filters.

**Selectivity ratio:**

```math
\begin{align}
  r(q) = \frac{\sigma_l(q)}{\sigma_g} \in [0, \infty)
\end{align}
```

**Per-query GLS** (Möbius transformation to $[-1, 1)$):

```math
\begin{align}
  \rho_q = \frac{r(q) - 1}{r(q) + 1} \in [-1, 1)
\end{align}
```

**Mean GLS over query workload $Q$:**

```math
\begin{align}
  \bar{\rho} = \frac{1}{|Q|} \sum_{q \in Q} \rho_q \in [-1, 1)
\end{align}
```

| $\rho_q$ value | Interpretation |
|---|---|
| $\rho_q > 0$ | Valid neighbors cluster near $q$; filtering is easy for any strategy |
| $\rho_q \approx 0$ | Valid neighbors are randomly distributed around $q$ |
| $\rho_q < 0$ | Valid neighbors are pushed to the periphery of $q$'s neighborhood; recall degrades |

> [!NOTE]
> GLS spans nearly the full range $[-1, 1)$ empirically, while distance-based correlation metrics only cover $[-0.3, 0.3]$, making GLS more expressive for diagnosing hard queries.

## System Architectures

### FAISS

A library (not a full database) requiring the caller to manage filter bitsets. FAISS uses vanilla HNSW from hnswlib and standard IVFFlat. Users integrate filtering via `IDSelector` objects passed at query time. No built-in optimizer or adaptive fallback — behavior is purely determined by the caller's integration.

### Milvus

A dedicated vector database using its own Knowhere engine (a heavily modified fork of hnswlib, not vanilla FAISS). Two key FANNS-specific innovations:

**Dual-Pool traversal:** Maintains two separate priority queues during HNSW search:
1. *Result pool*: tracks the $k$ best valid neighbors found so far
2. *Navigation pool*: tracks candidates for continued graph traversal (includes invalid nodes)

By decoupling navigation from result collection, Milvus prevents the result queue from saturating with invalid nodes, preserving recall at moderate selectivities (5%–10%) where FAISS degrades.

**Adaptive fallback:** When $\sigma_g > 0.93$, Milvus automatically switches to brute-force sequential scan, guaranteeing $\text{Recall}@k = 1.0$ for highly constrained queries without operator intervention.

### pgvector

A PostgreSQL extension that integrates ANN search into the relational query planner. The cost-based optimizer chooses between two plans:

1. **ANNS + Post-filtering:** Run vector index search, then apply $\phi$ on results
2. **Pre-filtering + kNNS:** Sequential scan with $\phi$ applied first, then exact $k$-NN on survivors

> [!IMPORTANT]
> The optimizer frequently selects plan 1 even when plan 2 would provide perfect recall at comparable latency, because its cost model does not accurately account for post-filtering recall loss under low selectivity.

## MoReVec Dataset

The paper introduces MoReVec (Movies and Reviews Vector dataset) as a relational FANNS benchmark with realistic metadata structure.

| Scale | Table | Rows | Embedding dim |
|---|---|---|---|
| Small | Reviews | ~248K | 768 |
| Medium | Reviews | ~1.5M | 768 |
| Large | Reviews | ~2.6M | 768 |

Embeddings are derived from movie review text using a pre-trained language model. The Movies table provides categorical metadata (genre, year, rating) enabling complex filter predicates. The design allows study of how filter correlation changes with semantic content.

> [!TIP]
> MoReVec, the extended ANN-Benchmarks framework, and GLS analysis tools are publicly available on GitHub alongside the paper.

## Experiments

- **Dataset:** MoReVec (small/medium/large), ANN-Benchmarks datasets
- **Hardware:** 56-core Intel Xeon E5-2660, 256 GB RAM; single-threaded query execution
- **Systems:** FAISS, Milvus v2.6.6, pgvector 0.8.1
- **Indexes:** HNSW ($M \in \{5,10,16,32\}$, $\text{efSearch}$ varied), IVFFlat ($\text{nprobe}$ varied)
- **Metrics:** QPS (queries per second), Recall@k

**Key results:**

- Filter selectivity universally degrades recall across all systems, but throughput impact varies: HNSW QPS is nearly invariant to selectivity (traversal cost dominates), while IVFFlat QPS improves at low selectivity due to cluster pruning.
- IVFFlat outperforms HNSW at low selectivity ($\sigma_g < 5\%$), contradicting standard ANN benchmarks where HNSW dominates — the performance curves cross at a system-dependent threshold.
- Milvus achieves near-perfect recall stability across selectivities 5%–100% due to dual-pool traversal and adaptive fallback; FAISS and pgvector degrade at low selectivity.
- GLS $\bar{\rho}$ predicts per-query recall difficulty independently of $\sigma_g$: queries with $\rho_q < -0.5$ lose recall even at moderate selectivities.
- Increasing HNSW $M$ beyond 10 yields negligible recall gains but significantly increases build time and memory usage.
- Milvus recall is robust to segment size variations, confirming the dual-pool algorithm (not data partitioning) drives performance.

## Practical Guidelines

| Scenario | Recommendation |
|---|---|
| Standard mixed-selectivity workload | Use HNSW; tune $\text{efSearch}$ for recall target |
| Restrictive filters ($\sigma_g < 10\%$) | Evaluate IVFFlat; may outperform HNSW |
| Maximizing recall stability | Use Milvus (dual-pool + adaptive fallback) |
| Relational workloads with PostgreSQL | Monitor pgvector's plan selection; force sequential scan plan when $\sigma_g$ is low |
| Rapid index rebuilds preferred | Use $M = 5\text{–}10$; larger $M$ has diminishing returns |
| Diagnosing hard queries | Compute per-query GLS $\rho_q$; queries with $\rho_q < 0$ need special handling |

## Comparison with Related Work

| Aspect | Standard ANN benchmarks | This paper |
|---|---|---|
| Filter modeling | Fixed or absent | Dynamic selectivity targets |
| Spatial correlation | Ignored | Quantified via GLS |
| Systems compared | Typically one | FAISS, Milvus, pgvector |
| Index winner | HNSW universally | IVFFlat wins at low selectivity |
| Dataset | Embedding-only | Relational (MoReVec) |

Compared to prior FANNS surveys, this work is the first to (1) introduce a normalized spatial correlation metric for filters, (2) demonstrate the IVFFlat/HNSW performance crossover empirically, and (3) analyze pgvector's optimizer pathologies in the FANNS context.
