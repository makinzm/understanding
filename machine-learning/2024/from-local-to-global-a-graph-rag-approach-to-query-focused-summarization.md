# Meta Information

- URL: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130.

# From Local to Global: A Graph RAG Approach to Query-Focused Summarization

## Problem Statement

Standard Retrieval-Augmented Generation (RAG) retrieves semantically similar text chunks to answer queries, which works well for **local** questions ("What does person X say about topic Y?") but fails for **global** questions that require reasoning across an entire corpus ("What are the main themes across all interviews?"). No single chunk contains enough context to answer holistically.

> [!NOTE]
> The authors define "query-focused summarization (QFS)" as generating answers that reflect an entire document collection rather than a single passage—a fundamentally different task from standard passage retrieval.

Graph RAG addresses this by pre-indexing the corpus into a hierarchical knowledge graph and summarizing communities of related entities, enabling parallel, comprehensive responses to corpus-level questions.

## Core Idea: Graph RAG Pipeline

The pipeline transforms raw documents into a hierarchical entity-relationship graph, then uses community detection to create multi-level summaries that can be queried efficiently.

**Input**: A collection of documents (e.g., 1 million tokens of interview transcripts or news articles).
**Output**: A community-structured knowledge graph with pre-computed summaries at multiple abstraction levels.

### Stage 1: Source Documents → Text Chunks

Documents are split into chunks of configurable size (600–2400 tokens). Smaller chunks yield higher extraction quality at greater computational cost.

### Stage 2: Text Chunks → Element Instances (Entity and Relationship Extraction)

An LLM processes each chunk to extract:
- **Entities**: Named nodes with types (person, place, organization, concept) and descriptions.
- **Relationships**: Directed edges between entity pairs, each with a textual description and numeric weight reflecting the strength of evidence.
- **Claims** (optional): Factual assertions attributed to entities.

**Gleaning** improves recall: after the first extraction pass, the LLM is prompted again with logit bias forcing an explicit "yes/no" response to "Are there any missing entities?". If yes, another extraction round runs. This multi-pass approach recovers entities missed in the initial pass without introducing noise.

```
function EXTRACT_ELEMENTS(chunk):
    entities, relationships = LLM_extract(chunk)
    while LLM_ask("Any missed entities?", logit_bias={yes: +10}) == "yes":
        new_entities, new_rels = LLM_extract(chunk, hint="focus on missed")
        entities += new_entities
        relationships += new_relationships
    return entities, relationships
```

### Stage 3: Element Instances → Element Summaries

Multiple mentions of the same entity across chunks are merged. The LLM generates a single unified description per entity and per relationship, consolidating all evidence. This prevents duplicate nodes and creates richer node/edge attributes.

### Stage 4: Element Summaries → Graph Communities (Leiden Algorithm)

All entity nodes and relationship edges form a weighted undirected graph $G = (V, E, w)$ where:
- $V$ is the set of entities,
- $E$ is the set of relationships,
- $w: E \to \mathbb{R}^+$ is the relationship weight.

The **Leiden algorithm** partitions $G$ into a hierarchy of communities by maximizing **modularity**:

```math
\begin{align}
  Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
\end{align}
```

where $A_{ij}$ is the adjacency weight between nodes $i$ and $j$, $k_i = \sum_j A_{ij}$ is the degree of node $i$, $m = \frac{1}{2}\sum_{ij} A_{ij}$ is the total edge weight, $c_i$ is the community of node $i$, and $\delta(c_i, c_j) = 1$ if $c_i = c_j$ else $0$.

Leiden produces multiple hierarchical levels (C0 = root/coarsest, C1, C2, C3 = leaf/finest), where each level represents a different abstraction granularity.

> [!TIP]
> The Leiden algorithm is an improvement over the Louvain algorithm for community detection in large graphs. See [Traag et al. (2019)](https://www.nature.com/articles/s41598-019-41695-z) for details.

### Stage 5: Graph Communities → Community Summaries

The LLM generates a natural-language summary for each community. To fit within the context window, the community report is built from prioritized elements:

1. Community entities (ordered by degree—more connected entities first).
2. Relationships of those entities (ordered by combined source + target degree).
3. Claims associated with those entities.

Each community summary is a standalone document describing the topic, key entities, relationships, and relevant claims within that cluster.

### Stage 6: Community Summaries → Global Answer (Map-Reduce)

**Query time** uses a two-phase map-reduce approach:

```
function GLOBAL_QUERY(question, community_summaries, context_window):
    # Map phase: score each community summary independently
    intermediate_answers = []
    for summary in community_summaries:
        if fits_in(summary, context_window):
            rating, points = LLM_answer(question, context=summary)
            intermediate_answers.append((rating, points))

    # Reduce phase: combine top-rated partial answers
    intermediate_answers.sort(by=rating, descending=True)
    final_answer = LLM_reduce(question, intermediate_answers)
    return final_answer
```

Each LLM call during the map phase rates its partial answer's helpfulness (0–100) so that the reduce phase can prioritize the most informative summaries. This allows parallel processing and scales with corpus size.

## Comparison with Related Methods

| Method | Graph Role | Scope | Summarization |
|---|---|---|---|
| Standard RAG | None (flat retrieval) | Local (per-chunk) | Single-pass |
| RAPTOR | Hierarchical clustering (embeddings) | Local to global | Tree traversal |
| CAiRE-COVID | Keyword-graph retrieval | Multi-document | Extractive |
| **Graph RAG** | Entity-relationship graph + community detection | **Global (corpus-level)** | **Map-reduce over communities** |

The key difference from RAPTOR is that Graph RAG builds an **explicit entity-relationship graph** (not embedding clusters) and uses **graph modularity** for partitioning, enabling meaningful community labels and descriptions beyond geometric clustering.

> [!IMPORTANT]
> Graph RAG is designed for **sensemaking over large, heterogeneous corpora** where the user does not know in advance what questions to ask. It is not optimized for precise fact lookup (use standard RAG for that) but for exploratory analysis requiring broad coverage.

## Evaluation

### Datasets

| Dataset | Domain | Size | Source |
|---|---|---|---|
| Podcast transcripts | Technology interviews | ~1 million tokens | Kevin Scott interviews |
| News articles | Current events | ~1.7 million tokens | MultiHop-RAG benchmark |

### Graph Statistics

| Dataset | Nodes | Edges |
|---|---|---|
| Podcast | 8,564 | 20,691 |
| News | 15,754 | 19,520 |

### Query Generation

Rather than using pre-existing QA benchmarks (which assume known answers), the authors generate **activity-centered sensemaking questions**: an LLM role-plays as a domain expert performing analysis tasks (e.g., "investigate trends in tech leadership") and generates 125 realistic questions per dataset. This avoids questions requiring knowledge of specific text passages.

### Metrics (LLM-as-Judge)

Head-to-head pairwise comparisons between systems, each question evaluated 5 times for statistical robustness:

| Metric | Description |
|---|---|
| Comprehensiveness | Does the answer address all relevant aspects of the question? |
| Diversity | Does the answer present multiple distinct perspectives? |
| Empowerment | Does the answer provide enough information for informed judgment? |
| Directness | Is the answer specific rather than vague? (control metric) |

### Results Summary

**Graph RAG vs. Naive RAG (full corpus):**
- Comprehensiveness: Graph RAG wins 72–83% of comparisons.
- Diversity: Graph RAG wins 62–82% of comparisons.
- Directness: Roughly equal (confirming the control metric behaves as expected).

**Community level efficiency:**

| Level | Token Cost vs. Source Text | Comprehensiveness vs. Naive RAG |
|---|---|---|
| C0 (root) | ~3% (97% reduction) | +72% win rate |
| C1 | ~16% | +72–82% win rate |
| C2 | ~26% | Highest comprehensiveness |
| C3 (leaf) | ~100% | Highest detail, highest cost |

C0 achieves strong comprehensiveness at a fraction of the token cost of processing raw source documents, demonstrating that the community summary hierarchy efficiently compresses corpus knowledge.

### Context Window Size

An 8K token context window optimizes comprehensiveness. Smaller windows (4K) hurt quality; larger windows (16K, 32K) show diminishing returns.

## Applicability

Graph RAG is appropriate when:
- The corpus is large (hundreds of thousands to millions of tokens) and heterogeneous.
- Users need **exploratory, open-ended analysis** rather than precise fact retrieval.
- Questions are **global** (require synthesizing information spread across many documents).
- Index build time is acceptable (offline preprocessing), but query latency must be low.

It is **not** appropriate for:
- Real-time indexing requirements (the pipeline is compute-intensive).
- Simple fact lookup or question-answering from a small, focused corpus.
- Applications where entity extraction quality is limited by LLM capabilities on domain-specific jargon.

## Limitations

- Evaluation covers only corpus-level sensemaking questions; performance on local retrieval tasks is not measured.
- The LLM-as-judge metric may exhibit biases (verbosity preference, self-consistency preference).
- Index construction cost scales with corpus size and number of gleaning rounds.
- Entity extraction quality depends on the underlying LLM's named entity recognition capability.
