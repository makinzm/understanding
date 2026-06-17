# Meta Information

- URL: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Malkov, Yu. A. and Yashunin, D. A. (2016). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. arXiv:1603.09320.

# Efficient and Robust Approximate Nearest Neighbor Search using HNSW

## Overview

**Problem**: K-nearest neighbor search (K-NNS) over large datasets suffers from the curse of dimensionality for exact algorithms, motivating approximate K-NNS (K-ANNS). Prior proximity graph methods (k-NN graphs, NSW) exhibit either power-law or polylogarithmic complexity scaling, limiting performance on low-dimensional or clustered data.

**Contribution**: HNSW (Hierarchical Navigable Small World) is a fully graph-based, incremental K-ANNS index with $O(\log N)$ complexity scaling. It requires no auxiliary vector-only coarse search structures (unlike kd-tree hybrids), making it applicable to arbitrary metric spaces.

**Who should use this**: Engineers building similarity search systems for large-scale embedding retrieval (images, text, audio), ANNS library implementors, and researchers working on metric space indexing. HNSW is especially beneficial when data is low-dimensional, highly clustered, or defined on non-Euclidean distances (e.g., Jensen-Shannon divergence, edit distance).

## Background: NSW and Its Limitations

Navigable Small World (NSW) graphs [Malkov et al., 2012, 2014] incrementally add each new element by bidirectionally connecting it to its $M$ nearest already-inserted neighbors. Early-inserted nodes accumulate many connections, becoming hubs that enable logarithmic greedy routing hops. However, the average hub degree itself grows logarithmically with dataset size $N$, so the total number of distance evaluations (hops × degree) scales as $O(\log^2 N)$ — polylogarithmic, not logarithmic.

Greedy routing in NSW has two phases: a "zoom-out" phase (traversing low-degree nodes) and a "zoom-in" phase (following short links near the query). Starting at a hub skips the zoom-out phase and improves speed, but the polylogarithmic bottleneck remains.

## Core Idea: Hierarchical Layer Decomposition

HNSW separates links by their characteristic length scale across discrete layers $l = 0, 1, \ldots, L$. Layer 0 contains all elements with short-range proximity links; each higher layer contains a decreasing nested subset with longer-range links.

**Level assignment**: When inserting element $q$, its maximum layer $l$ is sampled as:

```math
\begin{align}
  l = \lfloor -\ln(\text{unif}(0,1)) \cdot m_L \rfloor
\end{align}
```

where $m_L$ is a normalization factor (optimally $1/\ln(M)$, analogous to the skip list parameter $p = 1/M$). This produces an exponentially decaying probability distribution, yielding $O(\log N)$ expected layers.

**Search traversal**: Starting from the single fixed entry point at the top layer, greedy search descends layer by layer. At each layer the algorithm finds the local nearest element before descending, evaluating only a constant number of neighbors per layer (since the layer graph is uncorrelated with spatial positions). The total number of distance evaluations is therefore $O(\log N)$, breaking the polylogarithmic barrier of NSW.

## Algorithms

### INSERT (Algorithm 1)

Inserts element $q$ into the multilayer graph:

```
INSERT(hnsw, q, M, Mmax, efConstruction, mL):
  ep ← entry point of hnsw
  L  ← layer of ep                       // current top layer
  l  ← floor(-ln(uniform(0,1)) * mL)     // new element's layer

  // Phase 1: descend from top to l+1 using greedy ef=1 search
  for lc = L downto l+1:
    W  ← SEARCH-LAYER(q, ep, ef=1, lc)
    ep ← nearest element in W to q

  // Phase 2: build connections from layer min(L,l) down to 0
  for lc = min(L,l) downto 0:
    W         ← SEARCH-LAYER(q, ep, efConstruction, lc)
    neighbors ← SELECT-NEIGHBORS(q, W, M, lc)
    add bidirectional edges between q and neighbors at lc
    for each e in neighbors:                // shrink if over-connected
      if |neighborhood(e) at lc| > Mmax:
        set neighborhood(e) at lc to SELECT-NEIGHBORS(e, neighborhood(e), Mmax, lc)
    ep ← W

  if l > L: set entry point to q
```

### SEARCH-LAYER (Algorithm 2)

Greedy beam search returning `ef` nearest candidates to $q$ in layer `lc`. Maintains a visited set $v$, a candidate priority queue $C$, and a result set $W$ of size `ef`. Terminates when the nearest unexamined candidate is farther than the furthest element in $W$:

```
SEARCH-LAYER(q, ep, ef, lc):
  v, C, W ← {ep}
  while |C| > 0:
    c ← nearest element in C to q
    f ← furthest element in W to q
    if dist(c, q) > dist(f, q): break   // W is complete
    for each neighbor e of c at lc:
      if e not in v:
        v ← v ∪ {e}
        if dist(e, q) < dist(f, q) or |W| < ef:
          C ← C ∪ {e};  W ← W ∪ {e}
          if |W| > ef: remove furthest from W
  return W
```

### SELECT-NEIGHBORS-HEURISTIC (Algorithm 4)

Instead of simply returning the $M$ nearest candidates (Algorithm 3), the heuristic builds connections in diverse directions. It scans candidates from nearest to farthest and adds candidate $e$ to the result $R$ only if $e$ is closer to the inserted element $q$ than to any already-selected element in $R$:

```
SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections):
  W ← C
  if extendCandidates:
    for each e in C: W ← W ∪ {neighbors of e at lc}
  R ← ∅;  Wd ← ∅
  while |W| > 0 and |R| < M:
    e ← nearest in W to q
    if e is closer to q than to any element in R:
      R ← R ∪ {e}
    else:
      Wd ← Wd ∪ {e}
  if keepPrunedConnections:
    while |Wd| > 0 and |R| < M:
      R ← R ∪ {nearest in Wd to q}
  return R
```

This heuristic approximates the relative neighborhood graph, preserving global connectivity across cluster boundaries (see Fig. 2 in the paper). For 1D data it recovers the exact Delaunay graph, reducing HNSW exactly to a probabilistic skip list.

### K-NN-SEARCH (Algorithm 5)

```
K-NN-SEARCH(hnsw, q, K, ef):
  ep ← entry point;  L ← layer of ep
  for lc = L downto 1:
    W  ← SEARCH-LAYER(q, ep, ef=1, lc)
    ep ← nearest in W to q
  W ← SEARCH-LAYER(q, ep, ef, lc=0)
  return K nearest in W
```

The `ef` parameter (≥ $K$) controls recall vs. speed tradeoff.

## Construction Parameters

| Parameter | Role | Recommended value |
|-----------|------|-------------------|
| $M$ | Connections per element per layer | 5–48; smaller for low-dim/low-recall, larger for high-dim/high-recall |
| $M_{max}$ | Max connections per layer (layers > 0) | $M$ |
| $M_{max0}$ | Max connections at layer 0 | $2M$ (critical for high-recall performance) |
| $m_L$ | Level normalization | $1/\ln(M)$ (skip-list analogy) |
| `efConstruction` | Beam width during index build | ≥ target recall × (e.g., 100 for recall 0.9) |
| `ef` | Beam width at query time | ≥ $K$; trade off recall vs. speed |

> [!NOTE]
> Setting $m_L = 0$ and $M_{max0} = M$ degenerates HNSW to a directed k-NN graph (power-law complexity). Setting $m_L = 0$ and $M_{max0} = \infty$ degenerates it to NSW (polylogarithmic). Non-zero $m_L$ introduces the hierarchy giving $O(\log N)$ complexity.

## Complexity Analysis

**Search complexity**: Under the assumption of exact Delaunay graphs, each layer requires a constant number of distance evaluations $C \cdot S$ where $C$ is the bounded average Delaunay degree and $S = 1/(1 - e^{-m_L})$. Since the maximum layer index scales as $O(\log N)$, total search complexity is $O(\log N)$.

**Construction complexity**: Inserting one element requires a sequence of K-ANNS searches across its layers. The expected number of layers per element is $E[l] = \frac{m_L}{1 - e^{-m_L}}$, a constant. Therefore construction scales as $O(N \log N)$.

**Memory**: Average per-element memory is $(M_{max0} + m_L \cdot M_{max}) \cdot \text{bytes\_per\_link}$. For $M \in [6, 48]$ with 4-byte integer links, this is approximately 60–450 bytes per object.

## Comparison with Similar Methods

| Method | Complexity | Data type | Cluster robustness |
|--------|-----------|-----------|-------------------|
| k-NN graph | $O(N^\alpha)$ power-law | Euclidean (with aux.) | Poor |
| NSW | $O(\log^2 N)$ | Arbitrary metric | Moderate |
| **HNSW** | $O(\log N)$ | Arbitrary metric | High (heuristic) |
| FLANN | $O(\log N)$ (tree) | Euclidean only | Moderate |
| LSH (FALCONN) | Sub-linear | Euclidean/cosine | Moderate |
| Product Quantization (Faiss) | Sub-linear | Euclidean, compressed | Low memory, lower accuracy |

> [!IMPORTANT]
> HNSW's key advantage over NSW is that it fixes the "zoom-out" problem by assigning a single fixed entry point at the top layer, and reduces per-layer degree evaluations to a constant by separating links into scale-specific layers. Its advantage over PQ methods is higher accuracy at comparable or lower query latency, at the cost of higher RAM usage.

## Experiments

- **Datasets (vector space benchmark)**:
  - SIFT: 1M 128-d image feature vectors (L2); brute-force 94 ms
  - GloVe: 1.2M 100-d word embeddings trained on tweets (cosine); BF 95 ms
  - CoPhIR: 2M 272-d MPEG-7 image features (L2); BF 370 ms
  - Random vectors (hypercube): 30M 4-d (L2); BF 590 ms
  - DEEP: 1M 96-d deep image features (L2); BF 60 ms
  - MNIST: 60k 784-d handwritten digit images (L2); BF 22 ms
  - 200M SIFT subset (1B SIFT dataset) for PQ comparison

- **Datasets (non-metric space benchmark)**:
  - Wiki-sparse: 4M TF-IDF vectors ($10^5$-d sparse cosine)
  - Wiki-8: 2M topic histograms, 8-d Jensen–Shannon divergence
  - Wiki-128: 2M topic histograms, 128-d Jensen–Shannon divergence
  - ImageNet: 1M signatures (SQFD distance)
  - DNA: 1M sequences (Levenshtein edit distance)

- **Hardware**: 4× Xeon E5-4650 v2 (4×10 cores), 128 GB RAM; also Core i5-2400, i7-5930K, i7-6850K for parameter tuning
- **Baseline**: nmslib 1.1, FLANN 1.8.4, Annoy (02/2016), FALCONN 1.2, Faiss (2017 May build), VP-tree, NAPP
- **Results**:
  - On SIFT, GloVe, DEEP, CoPhIR: HNSW outperforms all rivals by a large margin across all recall levels
  - On 4-d random vectors: HNSW slightly faster than Annoy at high recall, strongly outperforms others
  - Non-metric spaces: HNSW is the clear leader on all 5 datasets; on Wiki-8 (8-d JS-divergence), improves over NSW by ~3 orders of magnitude in distance computations
  - vs. Faiss (200M SIFT, 1-NN): HNSW achieves higher recall at lower query latency; Faiss requires 23–30 GB RAM vs. 64 GB for HNSW, but builds in 11–12 hours vs. 42 min–5.6 hours
  - Construction on 10M SIFT with 40 threads: ~3 minutes at `efConstruction=100`
