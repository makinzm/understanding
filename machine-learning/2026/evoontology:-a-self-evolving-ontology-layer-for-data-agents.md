# Meta Information

- URL: [EvoOntology: A Self-Evolving Ontology Layer for Data Agents](https://arxiv.org/abs/2609.15779)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Chong, M., Zhang, S., Fan, J., & Du, X. (2026). EvoOntology: A Self-Evolving Ontology Layer for Data Agents. arXiv:2609.15779.

---

# EvoOntology: A Self-Evolving Ontology Layer for Data Agents

## 1. Introduction

Data agents are LLM-driven systems that must fulfill natural-language instructions over heterogeneous data sources — relational databases, spreadsheets, and files — using only a small set of generic tools (e.g., `run_sql`, `read_file`). The paper calls the resulting mismatch between an agent's shallow, tool-level access and the deep domain semantics required to answer a query the **agent-data gap**.

Two existing strategies fail to close this gap:
- **Raw data exploration**: the agent repeatedly issues probing queries against the underlying data to rediscover schema and semantics from scratch on every task, which is inefficient and does not accumulate knowledge across a workload.
- **Static semantic layers**: a human or a one-shot process manually builds a fixed metadata layer (glossaries, table/column descriptions) that is injected into the prompt. This does not scale to large data sources and cannot adapt to how a specific agent or LLM backbone actually behaves; the paper shows static layers can even degrade accuracy.

> [!IMPORTANT]
> EvoOntology's core contribution is an **agent-first, self-evolving ontology** exposed as an MCP (Model Context Protocol) server: instead of injecting all metadata into the prompt, the agent actively queries the ontology at runtime, and the ontology itself is iteratively rewritten based on evidence from the agent's own successes and failures.

**Who uses this / when / where:**
- Builders of LLM-based data agents (text-to-SQL systems, business-intelligence copilots, multi-source data research assistants) who need agents to reliably reuse domain knowledge across many queries against the same data source.
- Applicable when a fixed workload of natural-language tasks is run repeatedly against a stable but heterogeneous data source (tables, files, databases), so that the cost of building and evolving an ontology is amortized over many queries.
- Not intended for one-off, single-query interactions where there is no repeated workload to learn from.

---

## 2. Ontology Architecture

### 2.1 Three-Layer Structure

At any point in time $t$, the ontology state is represented as

```math
\begin{align}
  \mathcal{L}_t = (\mathcal{S}_t, \Gamma_t, \mathcal{R}_t)
\end{align}
```

where $\mathcal{S}_t$ is the **Content Layer**, $\Gamma_t$ is the **Schema Layer**, and $\mathcal{R}_t$ is the **Tool Layer**.

| Layer | Role | Contents |
|---|---|---|
| Schema Layer ($\Gamma_t$) | Structural rules | Object types and the typing rules that constrain how Content-Layer entries can be composed |
| Content Layer ($\mathcal{S}_t$) | Domain knowledge (typed semantic graph) | **Terms** (domain concepts), **Mappings** (terms → physical database fields and join paths), **Constraints** (valid usage rules for a term), **Evidence** (probe-query results that support a semantic claim), and **Semantic Relations** between terms (association, hierarchy, composition, equivalence, derivation) |
| Tool Layer ($\mathcal{R}_t$) | Runtime access surface | MCP server tools that let the agent selectively query the ontology instead of receiving it all injected into the prompt |

Structural references connect Terms to their Mappings and attach Constraints/Evidence to the relevant Term, so every semantic claim in the Content Layer is traceable back to a concrete, verifiable probe against the data.

### 2.2 Evidence-Grounded Initialization (Phase 1)

A **builder agent** constructs the initial ontology $\mathcal{S}_0$ from the training workload $\mathcal{D}$ (a set of representative natural-language tasks) without any human curation:

1. Analyze the workload $\mathcal{D}$ to propose a candidate set of concepts $\mathcal{C}$.
2. For each candidate concept $c \in \mathcal{C}$, issue a probe query `probe(c, D)` against the actual data and keep only concepts whose probe succeeds:

```math
\begin{align}
  \mathcal{C}^{+} = \{\, c \in \mathcal{C} \mid \mathrm{verify}(\mathrm{probe}(c, \mathcal{D})) = 1 \,\}
\end{align}
```

3. Build the initial Content Layer from only the evidence-supported concepts, under the initial Schema Layer $\Gamma_0$:

```math
\begin{align}
  \mathcal{S}_0 = \mathrm{construct}(\mathcal{C}^{+}, \mathcal{D}; \Gamma_0)
\end{align}
```

This guarantees that every entry in the initial ontology is backed by a real, executed query result rather than an LLM's unverified guess about the schema.

### 2.3 Self-Evolution Loop (Phase 2)

After initialization, EvoOntology repeatedly refines the ontology using the agent's own task trajectories $\Sigma_t$ (the sequence of tool calls, successes, and failures produced while attempting tasks with the current ontology $\mathcal{L}_t$):

1. **Failure attribution**: an attribution function maps each problematic trajectory signature to the layer responsible for it,

```math
\begin{align}
  \alpha: \Sigma_t \rightarrow \{\mathsf{c}, \mathsf{t}, \mathsf{s}\}
\end{align}
```

  where $\mathsf{c}$, $\mathsf{t}$, $\mathsf{s}$ denote the Content, Tool, and Schema layers respectively.
2. **Typed edit proposal**: based on the attributed layer, the evolver agent proposes a targeted, typed edit (e.g., add a missing Mapping, correct a Constraint, add a new MCP tool) producing a candidate next state $\mathcal{L}'_t$.
3. **Paired validation**: the candidate is accepted only if it improves a validation metric $\phi$ (task performance under backbone $m$ on a held-out validation set $\mathcal{V}$) by at least a threshold $\tau$ over the current state:

```math
\begin{align}
  \mathcal{L}_{t+1} =
  \begin{cases}
    \mathcal{L}'_t & \text{if } \phi(\mathcal{L}'_t, \mathcal{V}, m) - \phi(\mathcal{L}_t, \mathcal{V}, m) \geq \tau \\
    \mathcal{L}_t & \text{otherwise}
  \end{cases}
\end{align}
```

**Pseudocode (self-evolution loop):**

```
SelfEvolve(L_0, D_train, V, m, tau, T):
  L ← L_0
  FOR t = 1 .. T:
    trajectories ← run_agent(L, D_train, m)     # collect Sigma_t
    failures ← extract_failure_signatures(trajectories)
    FOR each signature in failures:
      layer ← alpha(signature)                  # attribute to content/tool/schema
      L_candidate ← propose_typed_edit(L, layer, signature)
      IF phi(L_candidate, V, m) - phi(L, V, m) >= tau:
        L ← L_candidate                          # accept edit
      # else reject, keep L unchanged
  RETURN L
```

> [!NOTE]
> Because attribution and validation are conditioned on the backbone $m$, different LLM backbones evolve structurally different ontologies from the same starting point — the paper reports pairwise content overlap of under 0.62 between ontologies evolved by different backbones on the same data source.

---

## 3. Experiments

### Benchmarks

- **DDR-Bench (Deep Data Research)**: open-ended data-research tasks over heterogeneous sources in a 10-K financial-filing scenario, scored with Message-Wise accuracy (per-turn correctness) and Trajectory-Wise accuracy (correctness of the full multi-turn synthesis).
- **InsightBench**: business-intelligence benchmark where each task pairs a CSV dataset with a ground-truth business insight; scored with Insight and Summary scores.
- **BIRD**: a text-to-SQL benchmark of natural-language questions over real-world relational databases, evaluated in the Oracle Knowledge setting with Execution Accuracy (EX) as the primary metric and Valid Efficiency Score (VES) as a secondary metric.

The paper does not report exact task counts per benchmark split; each is used in its standard, publicly released form.

### LLM Backbones

Six backbones were evaluated: GPT-5.5, GPT-5.6-sol, Claude-Sonnet-5, Claude-Opus-4.8, DeepSeek-V4-Flash, and Qwen3.5-Flash.

### Results

- **DDR-Bench**: average Trajectory-Wise accuracy improves by +17.8 points over the strongest baseline.
- **InsightBench**: a more modest average improvement of +1.9 points, which the paper attributes to domain-dependent variability in how much an ontology can help.
- **BIRD**: +7.4 points in Execution Accuracy and +8.6 points in Valid Efficiency Score on average.
- Static, manually-injected semantic layers were sometimes *worse* than no semantic layer at all, while EvoOntology's interactive, MCP-queried ontology produced gains across every tested backbone.
- Ablations attribute 57% of cumulative gains to Tool-Layer edits, even though Content-Layer edits occur more frequently — individual tool-level changes have larger marginal impact than individual content-level changes.
- Total token cost was about 20% lower than the baseline despite the accuracy gains, because agents no longer need to re-explore the schema from scratch on every task.

---

## 4. Comparison with Related Work

| Approach | Ontology construction | Adapts after deployment | Access pattern |
|---|---|---|---|
| Raw data exploration | None (rediscovered per task via probing) | No accumulated knowledge | Direct probing queries |
| Static semantic layer | Manual / one-shot automated construction | No | Fixed metadata injected into prompt |
| Trajectory memory (e.g., ReAct + Memory) | Implicit, from replayed past trajectories | Partially (replay only) | Unstructured trajectory replay |
| **EvoOntology** | Evidence-grounded automated construction | **Yes**, via attribution-guided typed edits with paired validation | **Selective runtime queries via MCP server tools** |

> [!NOTE]
> The paper reports that trajectory-memory baselines (ReAct augmented with memory) score 13.7 points lower than EvoOntology, because replaying past trajectories "does not expose typed, composable structure" the way an explicit ontology with Terms, Mappings, and Constraints does.

> [!CAUTION]
> This section reflects the paper's own stated comparison to baselines; exact baseline implementations and hyperparameters were not reproduced independently for this summary.

---

## Experiments

- Dataset: DDR-Bench (10-K financial-filing data research), InsightBench (CSV + ground-truth insight pairs), BIRD (text-to-SQL over real-world relational databases, Oracle Knowledge setting)
- Hardware: not specified in the available paper excerpt
- LLM backbones: GPT-5.5, GPT-5.6-sol, Claude-Sonnet-5, Claude-Opus-4.8, DeepSeek-V4-Flash, Qwen3.5-Flash
- Results: +17.8 points Trajectory-Wise accuracy on DDR-Bench, +1.9 points on InsightBench, +7.4 EX / +8.6 VES on BIRD (all averaged over backbones vs. the strongest baseline); ~20% lower total token cost than baseline; 57% of cumulative gains attributed to Tool-Layer edits
