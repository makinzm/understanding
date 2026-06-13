# Meta Information

- URL: [Survey of Filtered Approximate Nearest Neighbor Search over Vector-Scalar Hybrid Data](https://arxiv.org/abs/2505.06501)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yanjun Lin, Kai Zhang, Zhenying He, Yinan Jing, X. Sean Wang (2025). Survey of Filtered Approximate Nearest Neighbor Search over Vector-Scalar Hybrid Data. arXiv:2505.06501.

# Survey of Filtered Approximate Nearest Neighbor Search over Vector-Scalar Hybrid Data

## Overview and Motivation

Filtered Approximate Nearest Neighbor Search (FANNS) over hybrid datasets combines vector similarity search with scalar attribute filtering. In production retrieval systems—recommendation engines, multimodal search, e-commerce—data points have both a dense embedding vector and structured scalar metadata (e.g., category, price, timestamp). A query specifies both a scalar filter (e.g., `price < 100 AND category = "shoes"`) and a query vector; the goal is to retrieve the $k$ most similar vectors from the filtered subset.

Prior literature used inconsistent terminology (pre-filtering, post-filtering, in-filtering), lacked a unified classification framework for algorithms, and did not distinguish selectivity from distribution-based query difficulty. This survey fills those gaps by formalizing the problem, classifying 17 algorithms under a pruning-focused taxonomy, reviewing 9 hybrid datasets, and introducing the "distribution factor" as a new difficulty dimension.

> [!NOTE]
> "No dedicated survey on FANNS over vector-scalar hybrid data currently exists" — the paper addresses this gap with formal definitions and a systematic classification.

## Formal Problem Definition

### Hybrid Dataset

A hybrid dataset $\mathcal{D}$ is a collection of data points, each pairing a scalar tuple with a $d$-dimensional vector:

```math
\begin{align}
  \mathcal{D} = \{(\mathbf{s}_i, \mathbf{v}_i)\}_{i=1}^{N}, \quad \mathbf{s}_i \in \mathbb{S},\ \mathbf{v}_i \in \mathbb{R}^d
\end{align}
```

where $\mathbb{S}$ is the scalar schema (set of all possible scalar tuples) and $N$ is the number of data points.

### Hybrid Query

A hybrid query $q$ consists of four components:
- **Scalar filter** $f_s: \mathbb{S} \rightarrow \{0, 1\}$: a boolean predicate over scalar tuples
- **Vector similarity function** $f_v: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$: e.g., cosine similarity or Euclidean distance
- **Query vector** $\mathbf{v}_q \in \mathbb{R}^d$
- **Target result size** $k \in \mathbb{Z}^+$

The filtered subset is $\mathcal{D}_{f_s} = \{(\mathbf{s}_i, \mathbf{v}_i) \in \mathcal{D} \mid f_s(\mathbf{s}_i) = 1\}$.

**Objective**: Find a result set $\mathcal{R} \subseteq \mathcal{D}_{f_s}$ of size $\min(k, |\mathcal{D}_{f_s}|)$ whose vectors are the most similar to $\mathbf{v}_q$ under $f_v$.

### Evaluation Metrics

**Recall@k**: fraction of true top-$k$ results retrieved:

```math
\begin{align}
  \text{Recall@}k = \frac{|\mathcal{R} \cap \mathcal{R}^*|}{|\mathcal{R}^*|}
\end{align}
```

where $\mathcal{R}^*$ is the exact ground truth set.

**Selectivity** $\text{sel}_{f_s}$: fraction of data points *excluded* by the filter:

```math
\begin{align}
  \text{sel}_{f_s} = 1 - \frac{|\mathcal{D}_{f_s}|}{|\mathcal{D}|}
\end{align}
```

High selectivity (close to 1) means very few points satisfy the filter; low selectivity (close to 0) means most points pass. Selectivity directly governs algorithm behavior—high-selectivity queries benefit from scalar-first strategies, while low-selectivity queries favor vector-first strategies.

## Pruning Strategy Classification

Rather than the traditional pre/in/post-filtering categorization (which is ambiguous and inconsistent across papers), the authors define four classes by *what pruning happens* during search:

| Category | Abbreviation | Scalar Pruning | Vector Pruning | Best Regime |
|----------|-------------|---------------|---------------|-------------|
| Vector-Solely Pruning | **VSP** | None | Full vector index | Low selectivity |
| Vector-Centric Joint Pruning | **VJP** | Scalar check during traversal | Vector index guides traversal | Mid selectivity |
| Scalar-Solely Pruning | **SSP** | Full scalar filter first | Brute-force on remaining | High selectivity |
| Scalar-Centric Joint Pruning | **SJP** | Partitioned by scalar value | Vector index per partition | High selectivity with discrete scalars |

> [!IMPORTANT]
> The pruning-focused taxonomy is preferable to pre/post/in-filtering because it describes the *mechanism* of candidate reduction, enabling principled reasoning about tradeoffs without ambiguity from naming conventions.

## Algorithm Descriptions

### A1–A2: VSP (Vector-Solely Pruning)

**A1: Post-Filtering Family** — builds a standard ANN index (e.g., HNSW) over all vectors, retrieves an oversized candidate set of size $K' \gg k$, then applies the scalar filter to retain $k$ qualifying results. Simple but degrades rapidly at high selectivity, because retrieving $K'$ candidates before finding $k$ satisfying ones requires $K' \approx k / (1 - \text{sel})$.

**A2: VBase** — addresses the main weakness of post-filtering by dynamically selecting the optimal $K'$ via a "relaxed monotonicity condition": instead of fixing $K'$ in advance, VBase expands the candidate set incrementally until the marginal benefit of retrieving more candidates drops below a threshold. This avoids both over-retrieval and under-retrieval.

### A3–A11: VJP (Vector-Centric Joint Pruning)

These algorithms use a graph- or cluster-based ANN index as the primary structure but apply scalar checks *during* traversal.

**A3: ACORN (ANN Constraint-Optimized Retrieval Network)**
- Modifies HNSW's graph structure: uses a dense Delaunay Graph approximation (rather than the sparse Relative Neighborhood Graph) to maintain connectivity within the filtered predicate subgraph.
- During search, expansion only visits neighbors satisfying $f_s$; the dense connectivity prevents the traversal from becoming stranded in disconnected subgraphs.
- Applies memory compression to offset the cost of the denser graph.
- Trade-off: improved efficiency over VSP baselines but "uncertainty around the connectivity of the traversed subgraph makes search results potentially unreliable."

**A4: AIRSHIP (Attribute-Constrained Similarity Search on Proximity Graph)**
- Uses a probabilistic visiting strategy: nodes that *fail* $f_s$ are visited with some probability $p$ during neighbor expansion to maintain global connectivity, while *only* nodes satisfying $f_s$ are added to the result set.
- Balances exploitation (scalar-satisfying nodes near the query) with exploration (scalar-violating nodes acting as bridges).
- More efficient than VBase with graph indices at moderate selectivity.

**A5: Faiss-IVF with Scalar Filter**
- Inverted file index (IVF) partitions vectors into $C$ clusters via $k$-means; each cluster has an inverted list.
- During search, probes $n_{\text{probe}}$ nearest clusters, retrieves candidates, applies scalar filter, returns top-$k$.
- Degrades at high selectivity because clusters are built without scalar awareness.

**A6: CAPS (Cluster-Aware Predicate Search)**
- Extends IVF with an attribute frequency tree per cluster, enabling efficient identification of clusters enriched with scalar-satisfying data points.
- Refines cluster probe order based on predicted yield, reducing wasted computation on low-yield clusters.

**A7: NHQ (Navigating Hybrid Queries)**
- Constructs "fusion vectors" that concatenate encoded scalar attributes with the raw embedding vector.
- Graph index is built over these fusion vectors, so scalar proximity is encoded directly in edge structure.
- Search retrieves candidates using fusion-vector similarity, then verifies $f_s$ before inclusion.

**A8: HQANN (Hybrid Query Approximate Nearest Neighbor)**
- Similar fusion-vector approach to NHQ but with different encoding strategies for scalar attributes.

**A9: Filtered-DiskANN**
- Adapts DiskANN (disk-resident ANNS) for filtered queries by labeling edges in the graph with scalar metadata, enabling scalar-aware graph traversal on disk.

**A10: SeRF (Segment Graph for Range-Filtering)**
- Assumes a single orderable discrete scalar with range filter $[a, b]$.
- Precomputes overlaid range-specific subgraphs for all $O(n^2)$ possible ranges; during search, traverses the subgraph for the exact query range.
- Space complexity: $O(Mn^2)$ where $M$ is the graph degree, making it impractical for large $n$.

**A11: iRangeGraph (Improvising Range-dedicated Graphs)**
- Shares SeRF's assumption (single scalar, range filter) and search strategy.
- Uses a segment tree over scalar values to organize range-specific subgraphs; neighbors during search are the union of neighbors across all relevant segment-tree levels.
- Reduces space complexity to $O(Mn \log n)$.

### A12: SSP (Scalar-Solely Pruning)

**A12: Pre-Filtering Family** — applies $f_s$ exhaustively to the full dataset first to produce $\mathcal{D}_{f_s}$, then performs brute-force $k$-NN within $\mathcal{D}_{f_s}$.
- Time complexity is $O(N)$ for filtering and $O(|\mathcal{D}_{f_s}| \cdot d)$ for brute-force, making it impractical at low selectivity (large $|\mathcal{D}_{f_s}|$) but very efficient at high selectivity (tiny $|\mathcal{D}_{f_s}|$).

### A13–A17: SJP (Scalar-Centric Joint Pruning)

These algorithms partition the dataset by scalar values and build dedicated vector indices per partition.

**A13: Milvus-Partition**
- Partitions the dataset by discrete scalar values; builds a separate HNSW index per partition.
- Simple and effective when the number of distinct scalar values is small.
- Space scales as $O(|\text{Partitions}| \cdot N \cdot d)$ in the worst case.

**A14: HQI (Hybrid Query Index)**
- Multi-level partition structure with dedicated vector indices at each level, supporting hierarchical scalar conditions.

**A15: MA-NSW (Multiattribute ANNS based on Navigable Small World)**
- Handles multiple discrete scalar attributes with equality filters.
- Defines containment relationships between scalar-tuples (a tuple containing NULL values is more general).
- Builds NSW graph indices for each possible scalar-tuple combination, including subsets with NULL wildcards.
- Space complexity: $O(Mn^{m+1})$ where $m$ is the number of scalar attributes; exponential, limiting scalability.
- During search: retrieves all relevant subset indices, runs ANNS on each, merges results.

**A16: UNG (Unified Navigating Graph)**
- Builds separate graphs only for scalar-tuples actually observed in $\mathcal{D}$ (avoiding the exponential blowup of MA-NSW).
- Connects scalar-tuple subsets via "minimal containment relationships" organized in a prefix tree, giving $O(Mn)$ space.
- Efficiently navigates across related scalar-tuple subsets during search.

**A17: WST (Weighted Segment Tree)**
- Uses a segment tree over orderable scalar values, with a vector index at each node of the tree.
- Supports range filters by combining indices from $O(\log n)$ tree nodes covering the query range.

### Algorithm–Regime Mapping

| Selectivity | Recommended Strategy | Representative Algorithms |
|------------|---------------------|--------------------------|
| Low (< 0.3) | VSP | Post-Filtering, VBase |
| Medium (0.3–0.8) | VJP | ACORN, AIRSHIP, CAPS, NHQ |
| High (> 0.8), discrete scalars | SJP | UNG, MA-NSW, Milvus-Partition |
| High (> 0.8), arbitrary scalars | SSP + VJP hybrid | Pre-Filtering, Filtered-DiskANN |

## Hybrid Datasets

The survey identifies 9 representative hybrid datasets (D1–D9), grouped by whether scalars are synthesized or organic:

| ID | Dataset | Vectors | Dimensions | Scalar Type | Attributes |
|----|---------|--------|-----------|------------|-----------|
| D1 | SIFT-1M | 1M | 128 | Synthesized | — |
| D2 | GIST-1M | 1M | 960 | Synthesized | — |
| D3 | Deep-10M | 10M | 96 | Synthesized | — |
| D4 | MNIST-8M | 8.1M | 784 | Organic (digit label) | 1 |
| D5 | MTG (Million Song Tags) | ~1M | varies | Organic (tags) | multiple |
| D6 | GloVe-Twitter | 1.18M | 25–200 | Organic (word metadata) | 1 |
| D7 | LAION-1M | 1M | 512 × 2 | Organic (image+text) | 15 |
| D8 | YouTube | 6.1M | 1024+128 | Organic (video+audio) | 3 |
| D9 | MTG variants | varies | varies | Organic | varies |

**Synthesized-scalar datasets** (D1–D3): scalars drawn from uniform or Gaussian distributions independent of vectors; scalar and vector distributions are uncorrelated, making filters easier to plan for.

**Organic-scalar datasets** (D4–D8): scalars derived from real-world metadata; scalar and vector distributions are correlated, creating clustering effects that directly affect query difficulty.

> [!NOTE]
> MNIST-8M shows strong clustering: digit-label subsets have well-separated embedding distributions. MTG exhibits overlapping distributions across attribute values.

## Distribution Factor: A New Query Difficulty Dimension

Beyond selectivity, the survey introduces the **distribution factor** to describe the relationship between the query vector $\mathbf{v}_q$ and the base vectors $\mathcal{D}_{f_s}$ after filtering.

### Three Classes

| Class | Definition | Difficulty | Example |
|-------|-----------|-----------|---------|
| **ID** (In-Distribution) | $\mathbf{v}_q$ is drawn from the same distribution as $\mathbf{D}_{f_s}$ vectors | Easiest | Query from same digit cluster as filter |
| **POD** (Partially-Overlapping Distribution) | Query and base distributions partially overlap | Moderate | Query from adjacent but different semantic region |
| **OOD** (Out-of-Distribution) | $\mathbf{v}_q$ comes from a different distribution than $\mathcal{D}_{f_s}$ | Hardest | Query from digit-7 cluster, filter selects digit-0 cluster |

### Measurement

The Mahalanobis distance between the query vector and the filtered subset's empirical distribution measures distribution shift:

```math
\begin{align}
  d_M(\mathbf{v}_q, \mathcal{D}_{f_s}) = \sqrt{(\mathbf{v}_q - \boldsymbol{\mu}_{f_s})^\top \boldsymbol{\Sigma}_{f_s}^{-1} (\mathbf{v}_q - \boldsymbol{\mu}_{f_s})}
\end{align}
```

where $\boldsymbol{\mu}_{f_s} \in \mathbb{R}^d$ and $\boldsymbol{\Sigma}_{f_s} \in \mathbb{R}^{d \times d}$ are the empirical mean and covariance of vectors in $\mathcal{D}_{f_s}$.

- **ID** queries: histogram of $d_M$ values shows substantial overlap with the within-cluster distance distribution.
- **OOD** queries: histogram is shifted; Mahalanobis distances are significantly larger.
- The authors note this measure "remains imperfect" for distinguishing POD from OOD in borderline cases.

### Impact on Algorithms

Graph indices (e.g., HNSW) rely on navigating via nearest neighbors from the query; OOD queries start far from any node in $\mathcal{D}_{f_s}$, causing greedy traversal to get trapped far from the true top-$k$. Observed effects:
- Graph-based VJP algorithms show significant speedup for ID queries but minimal improvement (near-brute-force cost) for OOD queries.
- SSP and SJP methods are less sensitive to the distribution factor because they do not rely on proximity-guided traversal.

## Algorithm Search Pseudocode (Generalized VJP)

The following pseudocode represents the general search pattern for VJP methods:

```
function VJP_SEARCH(query_vector v_q, filter f_s, k):
    candidate_heap ← max-heap of size k (by similarity to v_q)
    visited ← empty set
    entry_point ← graph entry node

    priority_queue ← min-heap by distance(v_q, ·)
    priority_queue.push(entry_point)
    visited.add(entry_point)

    while priority_queue is not empty:
        node ← priority_queue.pop_nearest()
        if distance(v_q, node) > threshold:
            break

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                priority_queue.push(neighbor)
                if f_s(neighbor.scalar):  # scalar check
                    candidate_heap.push(neighbor)

    return top-k from candidate_heap
```

ACORN modifies the `graph.neighbors` step to use a denser graph; AIRSHIP adds probabilistic visits to neighbors failing `f_s`.

## Comparison with Standard ANNS

| Aspect | Standard ANNS | FANNS |
|--------|--------------|-------|
| Input | Query vector $\mathbf{v}_q$, dataset $\mathcal{D}$ | Query vector $\mathbf{v}_q$, filter $f_s$, dataset $\mathcal{D}$ |
| Search space | Full $\mathcal{D}$ | Filtered subset $\mathcal{D}_{f_s}$ |
| Index design | Vector structure only | Must integrate scalar and vector structures |
| Difficulty axis | Query proximity | Selectivity + distribution factor |
| Representative index | HNSW, DiskANN, IVF | ACORN, AIRSHIP, iRangeGraph, UNG |

## Open Research Directions

1. **Better distribution metrics**: Mahalanobis distance is insufficient for distinguishing POD from OOD; domain-specific or learned metrics are needed.
2. **Workload-aware optimization**: Dynamically selecting between VSP/VJP/SSP/SJP strategies based on query patterns at runtime.
3. **General scalar filter support**: Most SJP algorithms assume discrete equality or simple range filters; conjunctive/disjunctive predicates over continuous attributes remain unsolved.
4. **System-level integration**: Production FANNS systems need adaptive index selection, update support, and multi-tenancy.
5. **Index structure innovation**: Segment trees, prefix trees, and qd-trees show promise but remain tightly coupled to specific filter assumptions.

# Experiments

- **Datasets**: SIFT-1M, GIST-1M, Deep-10M (synthesized scalars); MNIST-8M, MTG, GloVe-Twitter, LAION-1M, YouTube (organic scalars)
- **Metrics**: Recall@k; Selectivity ($\text{sel}_{f_s}$)
- **Evaluation framework**: Oracle partition indices as theoretical optimal baseline
- **Key results**:
  - VSP degrades sharply as selectivity increases above ~0.8
  - SSP is optimal only when selectivity > 0.95 and $|\mathcal{D}_{f_s}|$ is small enough for brute-force
  - SJP methods (UNG, MA-NSW) are most efficient under high selectivity with discrete scalars
  - Distribution factor (ID vs. OOD) independently affects graph-based VJP recall, with OOD queries requiring near-exhaustive traversal even at moderate selectivity
- **Code**: https://github.com/lyj-fdu/FANNS
