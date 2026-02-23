# Meta Information

- URL: [RAG-Anything: All-in-One RAG Framework for Multimodal Documents](https://arxiv.org/abs/2510.12323)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang (2025). RAG-Anything: All-in-One RAG Framework for Multimodal Documents. arXiv:2510.12323. The University of Hong Kong.

# RAG-Anything: All-in-One RAG Framework for Multimodal Documents

## Overview

RAG-Anything addresses a critical limitation of conventional Retrieval-Augmented Generation (RAG) systems: their inability to handle non-textual content in real-world documents. Knowledge repositories in domains like academia, finance, and law contain rich combinations of textual content, visual elements, structured tables, and mathematical expressions. Existing RAG frameworks treat all content as flat text, causing information loss when visual or structured data is critical for answering a query.

RAG-Anything introduces a unified framework that:
1. Decomposes multimodal documents into typed atomic content units
2. Builds two complementary knowledge graphs (cross-modal and text-based) and merges them
3. Retrieves via a dual-pathway mechanism combining structural graph navigation and semantic vector search

**Applicability**: Organizations and researchers who need to query heterogeneous document corpora (annual reports, scientific papers, legal filings) where text-only RAG fails to surface information locked in tables, figures, or equations.

## Problem Formulation

Given a knowledge source $k_i$, the framework decomposes it into $n_i$ atomic content units:

$$k_i \rightarrow \text{Decompose}\; \{c_j = (t_j, x_j)\}_{j=1}^{n_i}$$

where $t_j \in \{\text{text}, \text{image}, \text{table}, \text{equation}, \ldots\}$ is the modality type and $x_j$ is the actual content. For non-textual units, two textual representations are generated:
- $d_j^{\text{chunk}}$: a detailed description optimized for chunk-level retrieval
- $e_j^{\text{entity}}$: an entity summary containing name, type, and description

> [!NOTE]
> Examples of $d_j^{\text{chunk}}$ is a caption-like description for an image, while $e_j^{\text{entity}}$ is a concise entity name and type (e.g., "Figure 3: Bar chart showing sales by region"). These representations enable the system to treat non-textual content as first-class citizens in both graph construction and retrieval.

## Architecture

The system has three stages: Indexing, Retrieval, and Synthesis.

### Stage 1: Indexing — Dual-Graph Construction

#### Cross-Modal Knowledge Graph

For each non-text unit $c_j$, a vision-language model generates $d_j^{\text{chunk}}$ and $e_j^{\text{entity}}$. Then a graph extraction routine $R(\cdot)$ produces entities and relations:

$$(\mathcal{V}_j, \mathcal{E}_j) = R(d_j^{\text{chunk}})$$

$\mathcal{V}$ means the set of entities and $\mathcal{E}$ means the set of relations.

> [!NOTE]
> The example of $R(\cdot)$ could be a combination of object detection for images, table structure parsing for tables, and equation component extraction for mathematical expressions. The resulting graph captures the internal structure of the non-textual content.

Each non-text unit also becomes a multimodal entity node $v_j^{\text{mm}}$ (identified by $e_j^{\text{entity}}$). The merged multimodal graph is:

$$\tilde{\mathcal{V}} = \{v_j^{\text{mm}}\}_j \cup \bigcup_j \mathcal{V}_j$$
$$\tilde{\mathcal{E}} = \bigcup_j \mathcal{E}_j \cup \bigcup_j \{(u \xrightarrow{\text{belongs\_to}} v_j^{\text{mm}}) : u \in \mathcal{V}_j\}$$

Each sub-entity $u \in \mathcal{V}_j$ links to its parent multimodal node via a `belongs_to` edge, preserving provenance.

#### Text-Based Knowledge Graph

Traditional named entity recognition and relation extraction are applied to textual segments to yield a separate set of entity and relation tuples.

#### Entity Alignment and Unified Index

The cross-modal graph and the text-based graph are merged via entity name matching. The final index is:

$$\mathcal{I} = (\mathcal{G}, \mathcal{T})$$

where $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is the unified knowledge graph and $\mathcal{T} = \{\text{emb}(s) : s \in \mathcal{V} \cup \mathcal{E} \cup \{c_j\}_j\}$ is the dense embedding table. Embeddings use `text-embedding-3-large` with 3072-dimensional vectors.

### Stage 2: Retrieval — Cross-Modal Hybrid Retrieval

For a query $q$, two complementary candidate sets are computed:

- **Structural Knowledge Navigation** $\mathcal{C}^{\text{stru}}(q)$: Keyword matching on $\mathcal{G}$ followed by neighborhood expansion within a fixed hop distance. Surfaces graph-connected entities and their contexts.
- **Semantic Similarity Matching** $\mathcal{C}^{\text{seman}}(q)$: Dense vector search between query embedding $e_q$ and $\mathcal{T}$, returning top-$k$ semantically similar chunks regardless of graph connectivity.

The unified candidate pool is:

$$\mathcal{C}(q) = \mathcal{C}^{\text{stru}}(q) \cup \mathcal{C}^{\text{seman}}(q)$$

A multi-signal fusion ranking $\mathcal{C}^\star(q)$ integrates three signals:
1. **Structural importance**: graph-topology-based node importance score
2. **Semantic similarity**: cosine similarity between $e_q$ and chunk embeddings
3. **Modality preference**: lexical analysis of $q$ to up-weight modality-specific candidates

> [!NOTE]
> The exact fusion formula combining the three signals is not provided in the paper. The fusion is described qualitatively as integrated scoring rather than a weighted sum.

### Stage 3: Synthesis — Multimodal Response Generation

The final response is generated by a Vision-Language Model (VLM) conditioned on both retrieved text and recovered visual artifacts:

$$\text{Response} = \text{VLM}(q,\; \mathcal{P}(q),\; \mathcal{V}^\star(q))$$

where $\mathcal{P}(q)$ is the retrieved textual context and $\mathcal{V}^\star(q)$ is the set of recovered visual elements (images, tables rendered to image) associated with top-ranked candidates.

### Document Parsing (MinerU)

RAG-Anything uses MinerU as the document parser to extract text, images, tables, and equations from PDFs while preserving layout and structural context. Each element's position and parent section are retained in the atomic unit metadata.

## Algorithms

### Knowledge Graph Construction (Indexing)

```
Input: Document k_i
Output: Unified index I = (G, T)

1. Parse k_i with MinerU → {c_j = (t_j, x_j)}
2. For each c_j where t_j != text:
     d_j^chunk, e_j^entity = VLM_describe(x_j)
     V_j, E_j = GraphExtract(d_j^chunk)
     v_j^mm = create_mm_node(e_j^entity)
     Add v_j^mm to V_tilde
     For each u in V_j: add edge (u --belongs_to--> v_j^mm)
3. For each c_j where t_j == text:
     V_j, E_j = NER_RE(x_j)  # named entity recognition + relation extraction
4. G_mm = (V_tilde, E_tilde)  # cross-modal graph
5. G_text = merge all (V_j, E_j) from text units
6. G = EntityAlign(G_mm, G_text)  # merge by entity name matching
7. T = {emb(s) for s in V ∪ E ∪ {c_j}}  # 3072-dim embeddings
8. Return I = (G, T)
```

### Cross-Modal Hybrid Retrieval

```
Input: Query q, Index I = (G, T)
Output: Ranked candidates C*(q)

1. e_q = embed(q)
2. C_stru = KeywordMatch(q, G) → expand neighbors up to hop h
3. C_seman = TopK(cosine(e_q, T))
4. C = C_stru ∪ C_seman
5. For each c in C:
     score(c) = f(structural_importance(c, G),
                   cosine(e_q, emb(c)),
                   modality_preference(q, t_c))
6. C*(q) = sort(C, by score, descending)
7. Return top-m candidates from C*(q)
```

## Comparison with Similar Systems

| Feature | Naive RAG | LightRAG | MMGraphRAG | RAG-Anything |
|---|---|---|---|---|
| Modalities | Text only | Text only | Text + Image (limited) | Text, Image, Table, Equation |
| Graph structure | None | Single KG | Multimodal KG (spectral clustering) | Dual KG (cross-modal + text) |
| Retrieval | Dense vector | Graph + dense | Graph + dense | Dual-path: structural + semantic |
| Non-text provenance | Lost | Lost | Partial | Preserved via `belongs_to` edges |
| VLM in synthesis | No | No | No | Yes |

> [!TIP]
> LightRAG ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779)) is the closest text-only predecessor that also combines graph and vector retrieval. RAG-Anything extends it by introducing multimodal entity nodes and cross-modal edges.

## Experiments

### Datasets

- **DocBench**: 229 multimodal documents, 1,102 QA pairs across 5 domains (Academia, Finance, Government, Laws, News). Documents range from single-page to 200+ pages.
- **MMLongBench**: 135 documents, 1,082 QA pairs across 7 document types. Focuses on long documents requiring cross-page reasoning over multimodal content.

### Baselines

- **GPT-4o-mini** (128K context): Full-document context stuffing without retrieval
- **LightRAG**: Graph-enhanced RAG, text-only
- **MMGraphRAG**: Multimodal knowledge graphs using spectral clustering for community detection

### Hardware / Models

- Embeddings: `text-embedding-3-large` (3072-dim)
- LLM for graph construction and synthesis: GPT-4o-mini
- Document parser: MinerU

### Key Results

**DocBench (accuracy %):**

| Method | Overall | Academia | Finance | Government | Legal | News |
|---|---|---|---|---|---|---|
| GPT-4o-mini | 49.6 | 40.3 | 46.9 | 60.3 | 59.2 | 61.0 |
| LightRAG | 59.7 | 53.8 | 56.2 | 59.5 | 61.8 | 65.7 |
| MMGraphRAG | 66.0 | 64.3 | 52.8 | 64.9 | 40.0 | 61.5 |
| **RAG-Anything** | **76.3** | **61.4** | **67.0** | **61.5** | **60.2** | **85.0** |

**MMLongBench (overall accuracy %):**

| Method | Overall |
|---|---|
| GPT-4o-mini | 33.5 |
| LightRAG | 38.9 |
| MMGraphRAG | 37.7 |
| **RAG-Anything** | **42.8** |

**Long-document performance (DocBench, 101–200 page documents):** RAG-Anything achieves 68.2% versus the next best at 54.6%, a 13.6-point improvement attributable to graph-based structural navigation preserving cross-page entity relationships.

### Ablation

| Variant | DocBench Overall |
|---|---|
| Full RAG-Anything | 76.3 |
| w/o graph (chunk-only) | ~60.0 |
| w/o reranking | ~75.3 |

Removing the knowledge graph causes a ~16-point drop, confirming that graph-based structural navigation is the primary driver of gains. Reranking contributes only marginally (~1 point).

## Identified Limitations

1. **Text-centric retrieval bias**: The system still tends to preferentially retrieve textual candidates even when a query explicitly requires visual information (e.g., "describe the bar chart in Figure 3").
2. **Rigid spatial processing**: Fixed layout parsing patterns in MinerU fail on non-standard document layouts (e.g., multi-column academic PDFs with interleaved figures).

> [!CAUTION]
> The fusion formula for multi-signal ranking is not publicly specified. Reproduction requires reverse-engineering the open-source implementation at [GitHub](https://github.com/HKUDS/RAG-Anything).
