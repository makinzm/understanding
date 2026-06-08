# Meta Information

- URL: [BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs](https://arxiv.org/abs/2603.20309)
- LICENSE: [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Reference: Duyi Pan, Tianao Lou, Xin Li, Haoze Song, Yiwen Wu, Mengyi Deng, Mingyu Yang, and Wei Wang (2026). BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs. arXiv:2603.20309 [cs.IR].

# BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs

## Abstract

Large Language Models (LLMs) are prone to hallucinations in knowledge-intensive tasks. Graph-based retrieval-augmented generation (RAG) has emerged as a promising remedy, but existing approaches suffer from fundamental recall and precision limitations when operating over **black-box knowledge graphs** — KGs whose schema and relational structure are unknown in advance.

The authors identify three core challenges causing this failure:

1. **Semantic instantiation uncertainty** — a query concept may appear in a KG under multiple heterogeneous surface forms (labels, aliases, attribute values, implicit patterns), causing relevant entities to be missed before any reasoning begins (recall loss).
2. **Structural path uncertainty** — even after finding relevant entities, without schema knowledge the retriever cannot identify which relational paths lead to the answer; the same high-level relation may appear as a direct edge, a multi-hop chain, or a composite structure (recall loss).
3. **Evidential comparison uncertainty** — when multiple candidates satisfy the constraints, the KG rarely encodes notions like expertise or importance explicitly; the retriever must aggregate implicit signals to rank them (precision loss).

The paper formalizes retrieval as the **Optimal Informative Subgraph Retrieval (OISR)** problem — a variant of the Group Steiner Tree problem — and proves it NP-hard and APX-hard. The proposed **BubbleRAG** system is a training-free pipeline that optimizes both recall and precision through semantic anchor grouping, heuristic bubble expansion to discover candidate evidence graphs (CEGs), composite ranking, and reasoning-aware expansion.

## 1. Introduction

### Problem Context

Standard text-chunk retrieval (NaiveRAG) treats all passages independently and ranks them by embedding similarity, missing cross-document dependencies that are made explicit in a KG. Graph-based RAG preserves relational structure, enabling structured evidence composition and symbolic reasoning across documents.

### Prior Solution Paradigms and Their Limitations

| Paradigm | Examples | Core Limitation |
|---|---|---|
| Schema-translation methods | SimGRAG, KG-GPT, BYOKG-RAG | Generated patterns only cover what the LLM already knows; fail when actual graph topology diverges from LLM priors |
| Random-walk methods | HippoRAG, HippoRAG2, LinearRAG, AGRAG | Hub bias — high-degree nodes absorb probability mass regardless of query relevance; single-anchor sensitivity |
| Iterative multi-hop methods | ToG, ToG2, RoG, LevelRAG, GraphSearch | Highly sensitive to initial anchor; a single misalignment causes cascading retrieval failures along the chain |
| Pre-indexed structure methods | GraphRAG, KAG, ClueRAG, RAPTOR | Static, query-agnostic structures; expensive to construct; ill-suited for diverse dynamic queries |

None of these jointly optimize recall and precision in a unified retrieval objective. BubbleRAG addresses this gap.

## 2. Preliminaries

### 2.1 Graph-Based RAG

A graph-based RAG system operates in three stages:

1. **Indexing**: the corpus is chunked, an LLM extracts triples, and these are indexed into a graph $G = (V, E)$.
2. **Retrieval**: given query $q$, the system searches $G$ to collect a subgraph $G^* \subseteq G$ as evidence.
3. **Generation**: the LLM produces an answer conditioned on the retrieved subgraph (and optionally the original text chunks).

### 2.2 Black-Box Knowledge Graphs

A KG is called **black-box** if its schema (entity types, relation types, and constraints) is not provided to the retrieval system. The retriever can access only the graph's topology and the text content on nodes and edges — no predefined traversal patterns, meta-paths, or type constraints are available.

> [!NOTE]
> "LLM-extracted KGs from heterogeneous corpora have no standardized schema; the same real-world relation may appear under different labels across documents."

### 2.3 Motivations for the Design

- **Motivation 1 (Semantic Anchoring)**: An evidence subgraph capable of answering $q$ must contain nodes that align — explicitly or implicitly — with the key concepts in the query.
- **Motivation 2 (Topological Cohesion)**: Relevant evidence tends to form a connected and compact structure; the concepts in $q$ are likely linked through relatively tight relational paths.

### 2.4 Problem Formulation — OISR

**Notation Table:**

| Notation | Description |
|---|---|
| $G = (V, E)$ | Knowledge graph with nodes $V$, edges $E$ |
| $\text{val}(\cdot)$ | Value function for nodes and edges |
| $q$ | Input user query |
| $t_i$ | Keyword extracted from query $q$ |
| $\mathcal{S} = \{S_1, \ldots, S_m\}$ | Semantic anchor groups |
| $w_i$ | Importance weight of group $S_i$ |
| $G^* = (V^*, E^*)$ | Optimal informative subgraph (target) |
| $G' = (V', E')$ | Localized search space subgraph |
| $h$ | Hop threshold for subgraph extraction |
| $B$ | Search budget for bubble expansion |
| $\text{Cost}_\text{sem}(T)$ | Semantic dissonance cost of CEG $T$ |
| $r_\text{miss}$ | Missing mass (uncovered group weights) |
| $\alpha$ | Penalty factor for structural completeness |
| $n$ | Number of top CEGs selected |
| $d$ | Max depth for multi-hop expansion |
| $G_\text{final}$ | Unified evidence graph merging CEGs |
| $C_\text{text}$ | Text chunks linked with evidence graph |

**Definition (OISR):**

- **Input**: A knowledge graph $G = (V, E)$ with value function $\text{val}(\cdot)$ on each node and edge, and a collection of anchor sets $\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$ where each $S_i \subseteq (V \cup E)$.
- **Output**: A subgraph $G' = (V', E')$ with $G' \subseteq G$.
- **Constraints**:
  - *Connectivity*: $G'$ is a connected graph.
  - *Multi-Set Coverage*: $(V' \cup E') \cap S_i \neq \emptyset$ for all $i \in \{1, \ldots, k\}$.
- **Objective**: Maximize the average value of nodes and edges in the selected subgraph:

```math
\begin{align}
  \max_{G' \subseteq G} \Phi(G') = \frac{\sum_{v \in V'} \text{val}(v) + \sum_{e \in E'} \text{val}(e)}{|V'| + |E'|}
\end{align}
```

> [!IMPORTANT]
> The average-value objective encodes an **information density** principle: it rewards compact subgraphs whose nodes and edges are all query-relevant, while penalizing subgraphs that achieve coverage through long, irrelevant paths. Using total value would favor sprawling subgraphs; using minimum value would be too brittle.

## 3. The BubbleRAG Framework

BubbleRAG has one offline phase and five online retrieval stages.

### 3.1 Data Preparation (Step 1)

The KG is built from the text corpus through: (1) chunking the corpus, (2) extracting triples with an LLM, and (3) indexing the triples into a graph. A key design choice is **edge representation**: rather than treating relations as simple labels, each edge connecting node $A$ and $B$ with relation $R$ stores the concatenated text $A\ R\ B$. This enrichment enables edge-level semantic matching, allowing the system to match query concepts against both entities and relations.

### 3.2 Semantic Anchor Grouping (Step 2)

This step maps each query concept to a **group** of candidate anchor nodes/edges, tolerating the heterogeneous realizations typical of black-box KGs. Maintaining multiple candidates per concept rather than committing to a single best match directly serves recall.

**Sub-steps:**

**Keyword Extraction and Latent Inference**: A standard NER extractor only recovers explicit surface mentions. BubbleRAG prompts an LLM to also infer implicit but necessary concepts. For the query *"Find scientific papers authored by the winner of the 1921 Nobel Prize in Physics"*, a naive extractor identifies *Nobel Prize, 1921, scientific papers*; BubbleRAG additionally infers *Albert Einstein*.

**Anchor Specialization**: After extracting keywords $\{t_i\}$, BubbleRAG retrieves a top-$k$ pool of candidate nodes for each $t_i$. Generic keywords like *mother* match thousands of nodes; the LLM rewrites each underspecified keyword into a query-conditioned constraint. For example, *mother* in the query *"When did Lothair II's mother die?"* is refined to *Lothair II's mother*, forcing similarity search to prioritize locally compatible candidates.

**Schema Relaxation**: A query concept may not match any KG label exactly. BubbleRAG uses the text chunks retrieved in the pre-retrieval step as previews of local graph communities. If a retrieved chunk confirms the co-occurrence of multiple keyword intents, the system can strategically relax a schema-sensitive condition to preserve recall. For example, the relation *"second marriage"* may be relaxed to *"marriage"* if contextual neighbors (*son*, *war*) confirm the correct locality.

**Anchor Grouping and Importance Weighting**: The LLM merges candidates referring to the same underlying concept into cohesive groups $\mathcal{S} = \{S_1, \ldots, S_m\}$ and assigns normalized importance weights $w_i$ with $\sum_{i=1}^m w_i = 1$. Weights reflect centrality to the query's answering logic: core subject entities receive higher weights, peripheral modifiers lower weights. For *"When did Lothair II's mother die?"*: $w_\text{Lothair} \approx 0.5$, $w_\text{mother} \approx 0.3$, $w_\text{death date} \approx 0.2$.

### 3.3 Candidate Evidence Graph Discovery (Step 3)

Given the weighted anchor groups $\mathcal{S}$, BubbleRAG seeks **Candidate Evidence Graphs (CEGs)** — connected subgraphs of $G$ covering all anchor groups while maximizing information density. This is the OISR problem.

**Complexity:**

**Theorem 1 (NP-hardness)**: OISR is NP-hard. Proof by polynomial-time reduction from Group Steiner Tree (GST): given a GST instance with size limit $\gamma$, construct an OISR instance where $\text{val}(v^*) = 1$ for a single designated node and $0$ elsewhere, with threshold $\alpha = 1/\gamma$. A Group Steiner tree of size $\leq \gamma$ exists iff a feasible OISR subgraph exists.

**Theorem 2 (APX-hardness)**: OISR cannot be approximated within a constant factor, because such an approximation would imply a constant-factor approximation for GST, contradicting the established result that GST on general graphs cannot be approximated within $O(\log^{2-\epsilon} k)$ unless $\text{NP} \subseteq \text{ZTIME}(n^{\text{polylog}(n)})$.

**Heuristic Solution — Bubble Expansion:**

The value function is instantiated as $\text{val}(v) = \cos(z_q, z_v)$ (cosine similarity between query embedding $z_q$ and node embedding $z_v$), and node cost is defined as:

```math
\begin{align}
  \text{cost}(v) = 1 - \text{val}(v) = 1 - \cos(z_q, z_v)
\end{align}
```

The algorithm operates in three phases:

**Phase 1 — Localized Graph Construction**: For each anchor group $S_i$, collect the $h$-hop neighborhood of all anchor nodes; take the union to form $G' = (V', E')$. This reduces computation by focusing exclusively on regions topologically reachable from anchors within a limited hop budget.

**Phase 2 — Anisotropic Expansion (Multi-source, Cost-guided Search)**: Within $G'$, a multi-source Dijkstra-like expansion is initiated from all anchor nodes simultaneously. For each node, the algorithm tracks: (i) minimum accumulated cost from each source group, (ii) predecessor pointer for path reconstruction, (iii) a bitmask recording which semantic anchor groups have already reached it. Unlike BFS (uniform expansion by hop count), this search is anisotropic: it propagates preferentially through nodes with low accumulated semantic cost (query-aligned regions) and stalls in high-cost (irrelevant) regions.

**Phase 3 — Collision Detection and Subgraph Fusion**: When an expansion frontier reaches a node whose bitmask indicates coverage by multiple different anchor groups, that node is treated as a candidate **Steiner node** (meeting point). A backtracing procedure reconstructs the low-cost paths from this node to the involved anchor nodes using predecessor pointers. The paths are fused into a connected CEG and added to the candidate set.

**Algorithm 1 — Bubble Expansion:**

```
Input:  G = (V, E), S = {S_1, ..., S_m}, h (localization hop), B (expansion budget)
Output: C (set of candidate subgraphs)

Phase 1: Localized Graph Construction
  G' ← ∅
  for each S_i in S:
    N_i ← h-hop neighborhood of nodes in S_i
    G' ← G' ∪ N_i
  G' ← (V', E') where V', E' are nodes and edges in the union

Phase 2: Anisotropic Expansion
  Initialize PQ with all terminals t ∈ ∪_i S_i
  Initialize cost[t] ← 0, mask[t] ← group bitmask of t, pred[t] ← ∅
  while PQ is not empty and |C| < B:
    (v, c) ← PQ.pop() with minimum accumulated cost
    if c > cost[v]: continue
    for each neighbor u of v in G':
      new_cost ← c + cost(u)
      new_mask ← mask[v] OR group bitmask of u
      if new_cost < cost[u]:
        cost[u] ← new_cost
        mask[u] ← new_mask
        pred[u] ← v
        PQ.push((u, new_cost))
      // Collision Detection:
      if new_mask covers multiple groups:
        T ← backtrack paths from u to terminals using pred
        C ← C ∪ {T}

Phase 3: Fallback
  if C is empty: C ← all terminals
  return C
```

> [!NOTE]
> Intra-group connections (fusion of anchors within the same group) are permitted but receive a mild penalty during ranking, to encourage inter-group coverage.

**Complexity Analysis:**

- Localized Graph Construction: $O(n \cdot d_\text{avg}^h)$ where $n = \sum_i |S_i|$ is the total number of anchor nodes.
- Anisotropic Bubble Expansion: $O(|E'| \cdot m \cdot \log |V'|)$ worst case; early pruning of high-cost paths keeps empirical runtime well below this bound.
- Overall: $O(n \cdot d_\text{avg}^h + |E'| \cdot m \cdot \log |V'| + |V'| \cdot m)$

> [!IMPORTANT]
> Because $|V'| \ll |V|$ due to localized construction, retrieval complexity is largely independent of the global graph size $|V|$. In experiments, the localized subgraph $G'$ typically contains fewer than $10^3$ nodes even when the full KG has $10^5$+ nodes.

### 3.4 Candidate Evidence Graph Ranking (Step 4)

The ranking phase re-evaluates CEGs using a richer criterion than the discovery heuristic. The **composite score** is defined as:

```math
\begin{align}
  \text{Score}(T) = \frac{1}{\text{Cost}_\text{sem}(T) \cdot \text{Penalty}_\text{miss}(T) + \epsilon}
\end{align}
```

where $\epsilon$ is a small constant to prevent division by zero.

**Semantic Dissonance** measures average per-node semantic cost in a CEG $T$:

```math
\begin{align}
  \text{Cost}_\text{sem}(T) = \frac{1}{|V_T|} \sum_{v \in V_T} \text{cost}(v)
\end{align}
```

Mean (rather than sum) is used to ensure **size invariance**: this prevents longer but uniformly query-relevant chains from being penalized relative to shorter ones.

**Structural Incompleteness Penalty** enforces coverage of all anchor groups with weighted tolerance. Let $r_\text{miss} = \sum_{i: S_i \cap V_T = \emptyset} w_i$ be the total weight of uncovered groups. Then:

```math
\begin{align}
  \text{Penalty}_\text{miss}(T) = e^{\alpha \cdot r_\text{miss}}
\end{align}
```

With large $\alpha$, missing a high-weight group sharply suppresses the score; missing a low-weight group incurs only a mild penalty.

**Support for Diverse Query Semantics via $\alpha$:**

| Query Type | $\alpha$ Setting | Effect |
|---|---|---|
| AND (all concepts required) | Large ($\alpha \gg 1$) | Missing any group causes exponential score drop; enforces conjunction |
| OR (any concept sufficient) | Near 0 ($\alpha \approx 0$) | Coverage penalty collapses to 1; ranking driven purely by semantic cost |
| Comparison queries | Default $\alpha$, top-$n$ selection | Each top-$n$ CEG captures evidence for a different candidate; LLM compares them side-by-side |

### 3.5 Reasoning-Aware Expansion (Step 5)

The minimal connected backbone (CEG) captures the **reasoning chain** leading to the answer, but the final answer entity may sit one or two hops beyond the backbone's boundary. BubbleRAG applies LLM-guided multi-hop expansion only to the top-$n$ ranked CEGs to serve precision.

Starting from each selected candidate $T^*$, the algorithm iteratively expands up to depth $d$. At each hop $\ell \in [1, d]$:

1. Retrieve the immediate neighbors $\mathcal{N}_d(T^*)$ adjacent to the current frontier.
2. Prompt the LLM with the query and current evidence to select the most promising neighbors.
3. Add selected nodes/edges to the evidence:

```math
\begin{align}
  T^*_d = T^*_{d-1} \cup \text{Selected}(\mathcal{N}_d)
\end{align}
```

Expansion terminates when the depth limit $d$ is reached or when the LLM selects no new neighbors. This is naturally an **anytime algorithm**: it can be stopped at any point when the time budget is exhausted.

### 3.6 Answer Generation (Step 6)

After reasoning-aware expansion, BubbleRAG merges the expanded subgraphs from top-$n$ candidates into a single **Unified Evidence Graph** $G_\text{final}$, consolidating redundant nodes and edges. Nodes in $G_\text{final}$ are mapped back to their source text chunks (via chunk pointers retained during indexing). The LLM receives a hybrid context combining:

1. Structured triples serialized from $G_\text{final}$ as a reasoning skeleton.
2. Associated raw text chunks $C_\text{text}$ as descriptive grounding.

# Experiments

## Setup

- **Datasets**: Three multi-hop QA benchmarks, each with 1,000 validation questions:
  - **MuSiQue**: multihop questions requiring 3–4 hop reasoning; composited from single-hop questions.
  - **HotpotQA**: diverse, explainable multi-hop QA; 2-hop reasoning.
  - **2WikiMultiHopQA**: multi-hop QA constructed from Wikipedia.
- **Hardware**: Single NVIDIA A100 GPU (efficiency analysis).
- **Embedding model**: Qwen3-Embedding-8B (applied uniformly to all methods for fairness).
- **LLM judge**: Qwen3-7B for LLM-as-a-Judge Accuracy evaluation.
- **Retrieval budget**: At most 15 text chunks per query for all methods.
- **Default hyperparameters**: $B = 10$, $h = 6$, $d = 6$, $\alpha = 1.0$, $n = 2$.

## Metrics

- **F1 Score**: Lexical overlap between generated answer and ground truth (token precision and recall).
- **LLM-as-a-Judge Accuracy ($\text{ACC}_L$)**: Qwen3-7B judges semantic equivalence; accounts for synonymous phrasings that exact match would penalize.

## Baselines

- **Vanilla LLM**: Parametric knowledge only, no retrieval.
- **CoT + LLM**: Chain-of-Thought prompting, no retrieval.
- **NaiveRAG**: Standard vector embedding retrieval over text chunks.
- **ToG (Think-on-Graph)**: Constrained beam search over KG topology (iterative multi-hop).
- **RAPTOR**: Hierarchical tree index with recursive summarization (pre-indexed structure).
- **LightRAG (Local/Global/Hybrid)**: Entity and relation graph with local, global, and hybrid retrieval modes (pre-indexed structure).
- **HippoRAG2**: Personalized PageRank over an open KG with passage nodes integrated (random-walk).
- **ClueRAG**: Multi-partite graph index with query-driven iterative retrieval (pre-indexed structure).

## Results

BubbleRAG achieves state-of-the-art performance across all three benchmarks in both the 30B and 8B model settings.

**Key quantitative results (30B model):**
- BubbleRAG vs. strongest baseline (HippoRAG2): +2.52% average F1, +2.23% average accuracy.
- On MuSiQue (most challenging, 3–4 hop): BubbleRAG F1 = **53.03** vs. HippoRAG2 F1 = 45.04 (≈+8 points).

**Performance with 8B model:**
- BubbleRAG with an 8B model (avg F1 = 63.02) is competitive with or better than many baselines using 30B models, demonstrating that retrieval quality — not model size — is the primary bottleneck for multi-hop QA.

**Ablation Study (on 2Wiki and HotpotQA):**

| Variant Removed | 2Wiki F1 | HotpotQA F1 | Primary Effect |
|---|---|---|---|
| Full BubbleRAG | 64.97 | 71.82 | — |
| w/o Anchor Specialization | 60.45 (-4.52) | 69.28 (-2.54) | Recall loss (fails to locate correct anchors) |
| w/o Schema Relaxation | 53.62 (-11.35) | 66.65 (-5.17) | Recall loss (largest drop; schema mismatch) |
| w/o CEG Ranking | 58.76 (-6.21) | 70.20 (-1.62) | Precision loss (noisy candidates selected) |

**Parameter Sensitivity (subset of 100 queries on 2Wiki):**

| Parameter | Value | F1 | Latency (s) |
|---|---|---|---|
| Budget ($B$) | 5 | 59.48 | 18.72 |
| | 10 | 60.52 | 20.99 |
| | 20 | 58.32 | 33.67 |
| | 50 | 60.42 | 44.98 |
| Depth ($d$) | 2 | 58.25 | 16.17 |
| | 4 | 59.07 | 19.39 |
| | 6 | 60.52 | 20.99 |
| | 8 | 62.28 | 32.67 |
| Penalty ($\alpha$) | 0.1 | 59.11 | — |
| | 1.0 | 60.52 | — |
| | 2.0 | 60.37 | — |
| | 5.0 | 57.35 | — |

**Efficiency Analysis (100 sampled queries, single A100 GPU):**

| Method | Latency (s) | Query Tokens | Index Tokens |
|---|---|---|---|
| NaiveRAG | 0.67 | 249,476 | — |
| HippoRAG2 | 4.26 | 418,812 | 4,575,580 |
| ToG | 45.93 | 765,915 | — |
| BubbleRAG | 20.99 | 1,064,052 | 3,840,320 |

> [!NOTE]
> BubbleRAG is ~5× slower than HippoRAG2 due to LLM calls in anchor grouping and reasoning-aware expansion, but 2× faster than ToG while delivering substantially higher accuracy. Index construction cost is amortized across all queries, comparable to HippoRAG2.

## 5. Related Work

- **Query Rewriting and Schema-Aligned Matching** (SimGRAG, KG-GPT, BYOKG-RAG): align queries with LLM-generated structural templates; fail when actual graph topology differs from LLM priors.
- **Iterative Multi-Hop Exploration** (ToG, ToG2, RoG, LevelRAG, GraphSearch): beam search or BFS from semantic seed nodes; cascading failures from initial anchor errors.
- **Stochastic Traversal via Random Walks** (HippoRAG, HippoRAG2, LinearRAG, AGRAG): Personalized PageRank from seed nodes; upper-bounded by anchor extraction accuracy; hub bias.
- **Structure-Augmented Retrieval with Auxiliary Graphs** (GraphRAG, KAG, ClueRAG): pre-built hierarchical indices; domain-dependent; ill-suited for dynamic queries.

BubbleRAG is differentiated on all four axes: it does not require schema knowledge, initializes from groups of anchors (not a single seed), constructs evidence structures dynamically at retrieval time, and jointly optimizes recall and precision as a formal optimization problem (OISR).

## 6. Conclusion

BubbleRAG is a training-free, plug-and-play graph RAG pipeline for black-box KGs. It formalizes retrieval as the OISR problem (NP-hard and APX-hard), then addresses this through a four-stage heuristic pipeline: semantic anchor grouping (recall), heuristic bubble expansion (recall), composite CEG ranking (precision), and reasoning-aware expansion (precision). It achieves state-of-the-art performance on multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) and maintains competitive performance with 8B-parameter models compared to baselines using 30B-parameter models. Its localized subgraph construction ensures scalability to massive KGs.

## References

- [1] Cai et al. (2025). SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven RAG. ACL 2025.
- [3] Edge et al. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv:2404.16130.
- [5] Gao et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.
- [6] Guo et al. (2025). LightRAG: Simple and Fast Retrieval-Augmented Generation. EMNLP 2025.
- [7] Jiménez Gutiérrez et al. (2024). HippoRAG: neurobiologically inspired long-term memory for LLMs. NeurIPS 2024.
- [8] Jiménez Gutiérrez et al. (2025). From RAG to Memory: Non-Parametric Continual Learning for LLMs. ICML 2025.
- [9] Halperin and Krauthgamer (2003). Polylogarithmic inapproximability. STOC 2003.
- [11] Ho et al. (2020). Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. COLING 2020.
- [17] Lewis et al. (2021). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401.
- [19] Luo et al. (2024). Reasoning on Graphs: Faithful and Interpretable LLM Reasoning. ICLR 2024.
- [21] Ma et al. (2025). Think-on-Graph 2.0. ICLR 2025.
- [26] Page et al. (1999). The PageRank citation ranking: Bringing order to the web. Stanford Technical Report.
- [27] Sarthi et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv:2401.18059.
- [28] Su et al. (2025). Clue-RAG: Towards Accurate and Cost-Efficient Graph-based RAG. arXiv:2507.08445.
- [29] Sun et al. (2024). Think-on-Graph. ICLR 2024.
- [32] Trivedi et al. (2022). MuSiQue: Multihop Questions via Single-hop Question Composition. TACL 2022.
- [37] Yang et al. (2025). Qwen3 Technical Report. arXiv:2505.09388.
- [39] Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. EMNLP 2018.
- [41] Zhang et al. (2025). Qwen3 Embedding. arXiv:2506.05176.
- [44] Zhou et al. (2026). In-Depth Analysis of Graph-Based RAG in a Unified Framework. VLDB 2026.
