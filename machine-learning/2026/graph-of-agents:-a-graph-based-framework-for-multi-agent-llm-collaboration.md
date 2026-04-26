# Meta Information

- URL: [Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration](https://arxiv.org/abs/2604.17148)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yun, S., Peng, J., Li, P., Fan, W., Chen, J., Zou, J., Li, G., & Chen, T. (2026). Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration. ICLR 2026.

# Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration

## Overview

Graph-of-Agents (GoA) is a test-time inference framework that coordinates multiple specialized LLMs by treating them as nodes in a directed graph. Each node is a domain-expert language model (e.g., a math-specialized model, a code model, a biomedical model), and edges represent information flow based on the relevance between their responses. Unlike flat ensembles or chain-of-thought prompting, GoA adapts its topology per query: it selects the most relevant agents, constructs response-driven edges, performs bidirectional message passing, and aggregates results through graph pooling.

The framework targets scenarios where a single general LLM underperforms specialized models but full-ensemble approaches waste compute. Practitioners running inference pipelines with heterogeneous model pools — e.g., research labs or enterprise deployments serving both general and domain-specific queries — benefit most from GoA's selective, structured orchestration.

The three core challenges GoA addresses are:
1. **Agent selection** — which subset of specialized models to activate for a given task
2. **Intra-agent communication** — how agents share and refine information with each other
3. **Response integration** — how to merge multiple responses into a single high-quality answer

## 2. Framework Design

### 2.1 Model Pool and Model Cards

GoA maintains a fixed pool $\mathcal{M} = \{m_1, m_2, \ldots, m_M\}$ of $M$ specialized LLMs. In the reference implementation $M = 6$:

| Model | Domain |
|---|---|
| Qwen2.5-7B | General-purpose |
| Qwen2.5-Coder-7B | Code / programming |
| Mathstral-7B | Mathematics |
| (Biomedical model) | Biomedical NLP |
| (Finance model) | Finance |
| Saul-7B | Legal reasoning |

Each model $m_i$ is associated with a **model card** $c_i$ — a structured text description of its domain coverage, task strengths, and maximum context length. Model cards are generated automatically via a dedicated script and serve as the lookup interface for the meta-LLM during agent selection.

### 2.2 Node Sampling (Agent Selection)

Given an input query $q$, a meta-LLM reads all model cards $\{c_1, \ldots, c_M\}$ and the query, then selects the top-$k$ most relevant agents:

```math
\begin{align}
  \mathcal{S} = \text{TopK-Select}(q,\, \{c_i\}_{i=1}^{M},\, k)
\end{align}
```

where $|\mathcal{S}| = k$ and $k < M$. Experiments show $k = 3$ out of $M = 6$ agents already outperforms methods that use all 6 agents simultaneously, demonstrating that relevance-based selection is more important than breadth.

The meta-LLM reasons about which domains are activated by the query (e.g., a medical coding question activates both the biomedical and code models), producing a set of agent identifiers rather than a ranked list.

### 2.3 Graph Construction via Response Relevance

After the selected agents $\mathcal{S}$ each generate an initial response $r_i$ to query $q$, GoA constructs a directed graph $\mathcal{G} = (\mathcal{S}, \mathcal{E})$ by scoring response similarity:

```math
\begin{align}
  e_{ij} = \text{cos\_sim}(\text{embed}(r_i),\, \text{embed}(r_j))
\end{align}
```

An edge $(i \to j)$ is included in $\mathcal{E}$ only if $e_{ij} \geq \tau$, where $\tau = 0.05$ is the default threshold. Agents whose responses are more similar have stronger communication channels, while dissimilar agents — which may have conflicting information — are weakly connected or disconnected.

> [!NOTE]
> The directed edge $(i \to j)$ means agent $j$ receives agent $i$'s response as context in the next round. Edges are not symmetric; the direction encodes who informs whom.

### 2.4 Bidirectional Message Passing

GoA uses a two-phase message passing scheme over $T$ rounds (default $T = 1$):

**Forward pass** (high-relevance → low-relevance):
Agents with higher average edge scores send their responses first. Each receiving agent $j$ appends the responses of its in-neighbors to its own context and regenerates its answer:

```math
\begin{align}
  r_j^{(t)} = m_j\!\left(q,\, r_j^{(t-1)},\, \{r_i^{(t-1)} : (i \to j) \in \mathcal{E}\}\right)
\end{align}
```

**Backward pass** (low-relevance → high-relevance):
The direction is reversed so that agents that initially received information now propagate their refined responses back. This ensures every agent incorporates perspectives from diverse specialists.

The bidirectionality distinguishes GoA from unidirectional chain-based systems (e.g., Mixture of Agents) where information only flows from "proposer" agents to a "aggregator."

### 2.5 Graph Pooling (Response Integration)

After $T$ rounds, the final responses $\{r_j^{(T)} : j \in \mathcal{S}\}$ are aggregated into a single answer. GoA supports two strategies:

- **Mean pooling**: a meta-LLM prompt that synthesizes all responses with equal weight
- **Max pooling**: select the single best response based on confidence or consistency scores

The pooling step maps from $k$ response strings to one output answer, completing the inference pipeline.

## 3. Algorithm Summary

```
Input: query q, model pool {m_i, c_i}_{i=1}^M, k, τ, T
Output: final answer a

1. S ← TopK-Select(q, {c_i}, k)           // node sampling
2. For each i in S: r_i^(0) ← m_i(q)      // initial responses
3. Build E: add (i→j) if cos_sim(r_i, r_j) ≥ τ  // graph construction
4. For t = 1 to T:
   a. Forward pass: update r_j^(t) using in-neighbors (high→low relevance)
   b. Backward pass: update r_i^(t) using in-neighbors (low→high relevance)
5. a ← GraphPool({r_j^(T) : j in S})      // response aggregation
Return a
```

## 4. Comparison with Related Methods

| Method | Agent Selection | Communication | Integration |
|---|---|---|---|
| Single LLM | N/A | None | Direct |
| Self-Consistency | None (same model) | None | Majority vote |
| Mixture of Agents (MoA) | None (all agents) | One-directional chain | Aggregator LLM |
| Chain-of-Thought | N/A | Sequential prompts | Last step |
| LLM Debate | None (same/different) | Turn-based debate | Vote or judge |
| **GoA** | Top-k via model cards | Bidirectional graph MP | Graph pooling |

> [!IMPORTANT]
> Mixture of Agents (MoA) uses all $M$ agents unconditionally and passes responses from layer-$l$ agents to layer-$(l+1)$ agents in a fixed layered pipeline. GoA differs by (1) dynamically selecting a subset of agents per query, (2) constructing task-specific graph topology based on response similarity, and (3) applying bidirectional message passing rather than unidirectional layered flow.

## Experiments

- **Datasets**:
  - MMLU (Massive Multitask Language Understanding) — 57-domain multiple choice, general knowledge
  - MMLU-Pro — harder version with 10 choices per question
  - GPQA (Graduate-level Google-Proof QA) — PhD-level science reasoning
  - MATH — competition-level mathematics word problems
  - HumanEval — Python code generation (pass@1)
  - MedMCQA — medical entrance exam multiple choice
- **Models in pool**: 6 specialized 7B-scale LLMs served via vLLM
- **Key result**: GoA with $k = 3$ selected agents outperforms all baselines that use all 6 agents simultaneously across MMLU, MMLU-Pro, GPQA, MATH, HumanEval, and MedMCQA
- **Ablation findings**:
  - Removing bidirectionality (using only forward pass) degrades performance
  - Random agent selection without model cards also degrades performance
  - Increasing message-passing rounds beyond 1 yields diminishing returns

> [!TIP]
> The official implementation is available at [UNITES-Lab/GoA](https://github.com/UNITES-Lab/GoA). It requires vLLM for serving individual models and includes `generate_model_card.py` for automatic model card generation.
