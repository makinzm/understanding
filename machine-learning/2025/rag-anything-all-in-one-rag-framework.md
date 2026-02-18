# Meta Information

- URL: [RAG-Anything: All-in-One RAG Framework](https://arxiv.org/abs/2510.12323)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Guo, Z., Ren, X., Xu, L., Zhang, J., & Huang, C. (2025). RAG-Anything: All-in-One RAG Framework. arXiv:2510.12323.

# RAG-Anything: All-in-One RAG Framework

## Overview

RAG-Anything is a multimodal Retrieval-Augmented Generation (RAG) framework that extends standard text-only RAG to handle heterogeneous document content including images, tables, figures, and mathematical equations. While conventional RAG pipelines extract and index plain text, real-world documents contain rich non-textual content. RAG-Anything addresses this by converting all modality types into a unified graph-based representation and then retrieves across all modalities simultaneously.

**Who would use this:** NLP/ML practitioners building document QA systems over technical reports, academic papers, financial documents, or any corpus with mixed-modality content. **When:** When queries require understanding of figures, tables, or equations—not just running text. **Where:** Enterprise document intelligence, scientific literature QA, financial analysis automation.

## Problem Setting

- **Input:** A heterogeneous document corpus $\mathcal{D}$ containing text paragraphs, images, tables, and equations; a natural language query $q$.
- **Output:** A grounded answer $a$ synthesized from retrieved multimodal evidence spanning the corpus.

**Why existing methods fail:** GraphRAG and LightRAG build knowledge graphs only from textual extraction. Multimodal RAG variants treat each modality independently in isolated silos, losing cross-modal relationships (e.g., a caption describing a figure, a table referenced in text).

## System Architecture

RAG-Anything consists of three main stages:

1. **Structure-Aware Multimodal Parsing** – decompose documents into typed atomic units.
2. **Dual-Graph Construction** – build two complementary knowledge graphs and fuse them.
3. **Cross-Modal Hybrid Retrieval** – navigate both graphs and dense vector indices to retrieve evidence, then synthesize an answer.

### Stage 1: Structure-Aware Multimodal Parsing

Documents are parsed into typed **atomic knowledge units** (AKUs):

| Type | Content | Preserved Context |
|------|---------|------------------|
| Text chunk | Paragraph or section | Section hierarchy, surrounding headings |
| Figure | Raster/vector image | Caption, panel labels, surrounding text |
| Table | Cell grid | Row/column headers, units, caption |
| Equation | LaTeX or rendered math | Variable definitions, surrounding text |

Spatial layout (bounding boxes) is retained for each AKU so structural relationships (e.g., "figure 3 is placed below equation 2 in section 4") are preserved.

### Stage 2: Dual-Graph Construction

Two graphs are constructed in parallel and then fused.

#### 2.1 Cross-Modal Knowledge Graph (CMKG)

Non-textual AKUs (images, tables, equations) are processed by a **multimodal LLM** to generate structured natural-language descriptions. Each AKU becomes a graph node with:
- A text description $d_i$ produced by the multimodal LLM (e.g., "a line chart showing accuracy vs. training steps for three methods").
- A type label $t_i \in \{\text{figure}, \text{table}, \text{equation}\}$.
- Its original visual/structured content $c_i$ (image bytes or table cells).

Edges in CMKG encode cross-modal relationships:
- **Text-to-Figure:** a paragraph's AKU references a figure AKU (via caption match or "see Figure X" mentions).
- **Text-to-Table:** text AKU references a table AKU.
- **Figure-to-Figure:** panels within a multi-panel figure share an edge.
- **Equation-to-Text:** equations are linked to the surrounding text defining their variables.

Pseudocode for CMKG construction:
```
Input: document D, multimodal LLM M
For each non-text AKU u in D:
    d_u = M.describe(u)       # Generate text description
    G_cm.add_node(u, description=d_u, content=u.raw)
For each pair (u, v) in D where reference(u, v):
    G_cm.add_edge(u, v, type=relationship_type(u, v))
Return G_cm
```

#### 2.2 Text-Based Knowledge Graph (TBKG)

Standard NLP entity extraction is applied to textual AKUs:
1. Named entity recognition extracts entities $E = \{e_1, \dots, e_n\}$ from text chunks.
2. Relation extraction identifies typed relations $(e_i, r, e_j)$ between entities.
3. Nodes are entities; edges are typed relations.

This graph captures semantic facts entirely within the textual modality—serving as the backbone of factoid QA.

#### 2.3 Graph Fusion

CMKG and TBKG are merged into a **unified knowledge graph** $G = G_{cm} \cup G_{tb}$ through entity alignment:
- Node descriptions $d_i$ from CMKG are embedded into dense vectors $\mathbf{v}_i \in \mathbb{R}^d$.
- Text entity nodes from TBKG that are semantically similar (cosine similarity $> \theta$) to a CMKG node are merged.
- Cross-graph edges are created wherever an entity from TBKG is mentioned within the description of a CMKG node.

> [!NOTE]
> "The dual-graph construction enables the system to capture both structured semantic relationships and cross-modal interactions, providing a comprehensive foundation for sophisticated multi-hop reasoning across modalities."

### Stage 3: Cross-Modal Hybrid Retrieval

Given query $q$, two complementary retrieval pathways are combined:

#### 3.1 Structural Knowledge Navigation (Graph Traversal)

1. **Entity recognition:** Extract query entities $Q_E$ from $q$ using NER.
2. **Node lookup:** Find nodes in $G$ matching $Q_E$ via keyword matching.
3. **Multi-hop traversal:** Expand from matched nodes along edges for up to $k$ hops, collecting neighboring nodes (both CMKG and TBKG).
4. **Result set:** $R_{graph} = \{\text{nodes visited during traversal}\}$.

This pathway excels at multi-hop reasoning: e.g., "What does Figure 5 show about accuracy?" requires linking the query entity "accuracy" (TBKG) → cross-graph edge → figure node (CMKG).

#### 3.2 Semantic Similarity Matching (Dense Retrieval)

All AKU descriptions $\{d_i\}$ and text chunks are embedded into a vector index. The query $q$ is embedded as $\mathbf{q} \in \mathbb{R}^d$ and nearest neighbors are retrieved:

$$R_{dense} = \text{top-}k \arg\max_{i} \frac{\mathbf{q} \cdot \mathbf{v}_i}{\|\mathbf{q}\| \|\mathbf{v}_i\|}$$

This pathway captures semantically related content even when no direct structural path exists.

#### 3.3 Result Fusion and Reranking

$R_{final} = \text{rerank}(R_{graph} \cup R_{dense}, q)$

A reranker model scores each retrieved item by relevance to $q$. For visual AKUs (figures, tables), the original content $c_i$ is recovered alongside the description $d_i$ and passed to the answer synthesis step.

### Stage 4: Answer Synthesis

Retrieved evidence is assembled into a multimodal context:
- Text chunks and entity descriptions are concatenated as $C_{text}$.
- Visual/table content $\{c_i\}$ for retrieved non-text AKUs is included as $C_{visual}$.
- A vision-language model receives $(q, C_{text}, C_{visual})$ and generates the final answer $a$.

## Comparison with Similar Methods

| Method | Modalities | Knowledge Representation | Cross-Modal Retrieval |
|--------|-----------|-------------------------|----------------------|
| Naive RAG | Text only | Dense index of chunks | No |
| GraphRAG | Text only | Single text knowledge graph | No |
| LightRAG | Text only | Single text knowledge graph | No |
| MMGraphRAG | Text + Images | Separate per-modality graphs | Limited (no fusion) |
| **RAG-Anything** | Text + Images + Tables + Equations | Dual-graph with fusion | Yes (structural + semantic) |

> [!TIP]
> GraphRAG: [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — RAG-Anything extends graph-based RAG to multimodal content.
> LightRAG: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) — same research group, predecessor text-only system.

## Experiments

- **Datasets:**
  - *DocBench:* 229 documents across 5 domains (finance, science, law, medicine, education), 1,102 QA pairs, average document length ~66 pages.
  - *MMLongBench:* 135 documents of 7 document types (papers, reports, slides, books, brochures, newspapers, product manuals), 1,082 QA pairs, average length ~47.5 pages.
- **Hardware:** Not explicitly specified.
- **Key Results:**
  - On documents exceeding 100 pages, RAG-Anything outperforms MMGraphRAG by **13+ percentage points** on DocBench.
  - Ablation on MMLongBench: chunk-only variant achieves 60.0%, removing the reranker gives 62.4%, full system reaches 63.4%.
  - Consistent improvements on MMLongBench across all document length categories.

## Failure Modes

Two systematic failure patterns were identified:

1. **Text-Centric Retrieval Bias:** When queries could be partially answered by text, the retrieval pathway preferentially retrieved text nodes even when a visual AKU was the correct answer. Mitigation requires query-type classification before retrieval.

2. **Rigid Spatial Processing:** Documents with non-standard layouts (e.g., multi-column PDFs, landscape slides) cause incorrect segmentation of AKUs, breaking structural relationships. Mitigation requires layout-adaptive parsing.

## Open Source

Framework available at: [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)
