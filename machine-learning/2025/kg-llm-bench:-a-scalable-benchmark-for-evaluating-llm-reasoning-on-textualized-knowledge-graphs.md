# Meta Information

- URL: [KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs](https://arxiv.org/abs/2504.07087)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Markowitz, E., Galiya, K., Ver Steeg, G., & Galstyan, A. (2025). KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs. arXiv:2504.07087.

---

# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs

## Overview

KG-LLM-Bench is a scalable benchmark for measuring how five different **knowledge graph (KG) textualization strategies** affect LLM reasoning performance across five KG-specific tasks. Choosing the right textualization strategy can change overall benchmark performance by up to **17.5% absolute difference**, with even larger gaps on specific tasks. The benchmark uses the WikiDataSets Countries KG, applies pseudonymization to control for memorization, and evaluates seven models ranging from large closed-source APIs to small open-source models.

**Who benefits**: NLP researchers and ML engineers who embed knowledge graphs into LLM prompts for knowledge-augmented generation (RAG-with-KG, KGQA, and similar settings), and need to decide how to serialize triples into text.

---

## Background: Knowledge Graph Textualization

A knowledge graph $G = (G_E, G_R, G_T)$ consists of:
- $G_E$: a set of entities (nodes)
- $G_R$: a set of relation types (edge labels)
- $G_T \subseteq G_E \times G_R \times G_E$: a set of directed triples $(s, r, o)$

When an LLM must reason over a KG, the graph must be serialized to a text string $x_G$ that is injected into the prompt. The serialization choice is a **textualization strategy** $\tau: G \to x_G$.

### Five Textualization Strategies

| Strategy | Format | Avg. tokens / prompt |
|---|---|---|
| List-of-Edges | `(South Korea, diplomatic relation, Ukraine)` — one triple per line, sorted by subject then relation | ~2,600 |
| Structured YAML | Subject as top-level key; relations and objects as nested lists | ~2,903 |
| Structured JSON | Subject as JSON key mapping to `{relation: [objects]}` dicts | ~4,505 |
| RDF Turtle | W3C Turtle syntax with namespace prefixes and semicolons | ~8,171 |
| JSON-LD | JSON-based linked data with `@context` and full URI expansion | ~13,503 |

> [!NOTE]
> List-of-Edges and Structured YAML are the most token-efficient (below 3,000 tokens). JSON-LD is the least efficient at over 13,000 tokens per prompt, reflecting overhead from semantic web annotations.

---

## Five Benchmark Tasks

Each task instance is a tuple $(x_G, q, a)$ where $x_G$ is the textualized subgraph, $q$ is a natural-language question, and $a$ is the gold answer. Model output $\hat{y} = \pi(x_G, q)$ is evaluated with exact-match scoring: $s = S(\hat{y}, a) \in \{0, 1\}$.

### Task 1: Triple Retrieval

- **Input**: Textualized graph $x_G$, question "Does entity $s$ have relation $r$ to entity $o$?"
- **Output**: Binary yes/no answer
- **Positive samples**: Drawn from real triples $(s, r, o) \sim G_T$
- **Negative samples**: Replace source, relation, or object with an alternative from $G_E$ or $G_R$; split 50/50 positive/negative
- **Average accuracy**: 86.2% — easiest task in the benchmark

### Task 2: Shortest Path

- **Input**: Textualized graph $x_G$, question "What is the shortest path from entity $e_1$ to entity $e_2$?"
- **Output**: Ordered list of entities forming the path $p_1$
- **Construction**: At least one shortest path $p_1$ is guaranteed to exist in the sampled subgraph; all entities of $p_1$ are included as seed nodes during sampling
- **Evaluation**: Exact-match on the full path sequence
- **Average accuracy**: 7.5% — hardest overall; requires structured multi-hop traversal

### Task 3: Aggregation by Relation (AggByRelation)

- **Input**: Textualized graph $x_G$, question "How many entities have relation $r$ with entity $s$ in direction $\text{dir}$?"
- **Output**: Integer count
- **Formula**:
$$\text{Agg}(s, r, \text{dir}) = |\{t = (s, r, e) \text{ or } (e, r, s) \mid e \in G_E,\; t \in G_T\}|$$
- **Average accuracy**: 42.3%

### Task 4: Aggregation of Neighbor Properties (AggNeighborProperty)

- **Input**: Textualized graph $x_G$, question requiring two-hop reasoning: "How many neighbors of $s$ have relation $r$ to some entity?"
- **Output**: Integer count
- **Formula**:
$$\text{Agg}(s, r) = |\{e_1 \in G_E \mid \exists\, t_1, t_2 \in G_T :\; t_1 \in \{(s, \_, e_1), (e_1, \_, s)\} \land t_2 = (e_1, r, \_)\}|$$
- **Average accuracy**: 44.3%

### Task 5: Highest Degree Node

- **Input**: Textualized graph $x_G$, question "Which entity has the most outgoing (or incoming, or total) edges?"
- **Output**: Entity name
- **Three sub-variants**: Outgoing degree, incoming degree, total degree
- **Key finding**: Models score significantly higher on outgoing than incoming degree, because most textualization formats group outgoing edges together under a subject, making them easier to count
- **Average accuracy**: 12.4%

---

## Benchmark Construction

### Dataset

**WikiDataSets Countries** knowledge graph:
- 3,552 core entities (historical and modern countries/territories)
- 49 core relation types (geographical, political, diplomatic, temporal)
- 11,361 core triples

### Subgraph Sampling

For each task instance, a 200-edge subgraph is sampled using an ego-graph strategy:

$$\text{EgoGraph}(e, r) = \{t = (s, r', o) \mid d(e, s) \leq r,\; d(e, o) \leq r,\; t \in T\}$$

where $d$ is shortest-path distance in the KG. Steps:
1. Sample seed entities $e$ from $G_E$
2. Extract ego-graph at radius $r$
3. Remove low-degree entities with only a single edge (avoid trivial answers)
4. Prune randomly until the subgraph contains exactly 200 edges

Each task generates **100 instances**; total benchmark size is 500 task instances × 5 textualization strategies × 7 models.

### Pseudonymization

To test whether models rely on memorized world knowledge rather than in-context graph content:
- Replace all real entity names with synthetic names (e.g., fake country names) via a deterministic mapping $p(G, \hat{E})$ that preserves graph structure
- Synthetic names generated by a name-generation tool (first 100 names) plus 600 additional names generated with Claude, with inappropriate names filtered out
- **Result**: Pseudonymization changed overall accuracy by only **0.2%**, confirming that models answer based on graph structure rather than memorized entity facts

---

## Experimental Setup

- **Models evaluated** (7 total):
  - Claude-3.5-Sonnet (Anthropic)
  - Gemini-1.5-Flash (Google)
  - GPT-4o-Mini (OpenAI)
  - Llama 3.3-70B and Llama 3.2-1B (Meta, open-source)
  - Amazon Nova Lite and Nova Pro

- **Prompt structure**: System prompt + textualized graph $x_G$ + task question $q$; models instructed to output only the answer
- **Subgraphs**: 200 edges per instance, 100 instances per task

---

## Results

### Overall Textualization Ranking

| Textualization | Overall Accuracy |
|---|---|
| Structured JSON | **0.42** |
| Structured YAML | 0.41 |
| List-of-Edges | 0.41 |
| RDF Turtle | 0.35 |
| JSON-LD | 0.34 |

Structured JSON performs best overall, but the **best format is model-dependent**:

| Model | Best Format |
|---|---|
| Claude-3.5-Sonnet | RDF Turtle |
| Gemini-1.5-Flash | List-of-Edges |
| GPT-4o-Mini | List-of-Edges |
| Llama 3.2-1B | List-of-Edges |
| Llama 3.3-70B | Structured JSON |
| Nova Lite | Structured JSON |
| Nova Pro | JSON-LD |

> [!NOTE]
> Claude-3.5-Sonnet achieves 61.5% on Highest Degree with RDF Turtle, while the task average is only 12.4%. Nova-Pro achieves 47% on Shortest Path with RDF Turtle vs. 7.5% average, suggesting that semantic web training data gives certain models an edge on RDF-formatted inputs.

### Task Difficulty

| Task | Avg. Accuracy |
|---|---|
| Triple Retrieval | 86.2% |
| AggNeighborProperty | 44.3% |
| AggByRelation | 42.3% |
| Highest Degree Node | 12.4% |
| Shortest Path | 7.5% |

### Aggregation Performance vs. Degree

For AggByRelation and AggNeighborProperty, accuracy degrades sharply with increasing degree:
- Degree ≤ 4: above 50% accuracy
- Degree > 4: rapidly drops to ~10%

This reflects a counting limitation rather than a structural reasoning failure.

---

## Comparison with Related Work

| Work | Scope | Formats Compared |
|---|---|---|
| Fatemi et al. (2023) | Natural language vs. structured KG | 3 formats, single KG |
| Frey et al. (2023) | RDF Turtle parsing evaluation | RDF Turtle only |
| KGQA benchmarks (MetaQA, HotpotQA) | Question answering over KGs | Not focused on textualization |
| **KG-LLM-Bench** | Comprehensive textualization impact | **5 formats, 5 tasks, 7 models** |

KG-LLM-Bench is the **first systematic comparison** of textualization strategies across multiple tasks and model families, with pseudonymization controls and a scalable subgraph sampling framework.

---

# Experiments

- **Dataset**: WikiDataSets Countries KG — 3,552 entities, 49 relations, 11,361 triples
- **Hardware**: Not specified (uses external API calls)
- **Optimizer**: Not applicable (inference-only evaluation)
- **Results**:
  - Best overall format: Structured JSON (0.42 average accuracy)
  - Textualization choice accounts for up to 17.5% absolute performance difference
  - Pseudonymization effect: 0.2% difference (models rely on graph content, not memorized facts)
  - Easiest task: Triple Retrieval (86.2%); Hardest: Shortest Path (7.5%)
  - Token cost: List-of-Edges (~2,600 tokens) vs. JSON-LD (~13,500 tokens) — 5× difference for the same graph
