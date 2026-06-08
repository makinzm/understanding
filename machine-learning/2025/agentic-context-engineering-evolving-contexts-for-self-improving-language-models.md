# Meta Information

- URL: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., Thakker, U., Zou, J., & Olukotun, K. (2025). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv:2510.04618.

# Agentic Context Engineering (ACE)

## Overview

ACE is a framework for context adaptation in LLMs that treats system prompts and agent memories as **evolving playbooks** rather than static summaries. Instead of compressing knowledge into concise snapshots (which leads to information loss), ACE accumulates and refines domain-specific strategies through a structured generate–reflect–curate cycle. ACE is applicable to any LLM-based system where performance improves with domain experience: agentic task execution, financial analysis, and knowledge-intensive classification.

**Target users:** ML engineers deploying LLM agents in specialized domains who need lightweight, interpretable adaptation without fine-tuning or labeled supervision.

## Problem: Limitations of Existing Context Adaptation

Existing natural language feedback methods (Reflexion, TextGrad, GEPA, Dynamic Cheatsheet) suffer from two failure modes:

| Problem | Description | Effect |
|---|---|---|
| **Brevity Bias** | Methods prioritize short, concise prompts over detailed heuristics | Domain-specific strategies and edge-case knowledge are discarded |
| **Context Collapse** | LLM-based monolithic rewriting compresses accumulated knowledge each iteration | Progressive information loss causes sharp performance drops over time |

> [!NOTE]
> "Existing context adaptation methods suffer from brevity bias and context collapse, causing them to prune valuable domain-specific heuristics and progressively lose information across iterations."

## ACE Architecture

ACE uses three specialized LLM components operating in a pipeline:

```
Task Trajectory → [Generator] → Traces
                → [Reflector] → Lessons (δ entries)
                → [Curator]   → Merged Context
```

### 1. Generator

- **Input:** Current context $C$ (system prompt or agent memory), task description
- **Output:** Reasoning trajectories that surface effective strategies and common pitfalls
- The generator executes the task using the current context, producing execution traces that expose both successes and failure modes

### 2. Reflector

- **Input:** Execution traces from the Generator
- **Output:** A set of delta lessons $\delta_1, \delta_2, \ldots, \delta_k$ in structured bullet format
- The Reflector applies **iterative self-critique**: it drafts lessons, then refines them by asking "Are these lessons concrete, actionable, and non-redundant?" This multi-pass refinement is a key contributor to ACE's performance (confirmed in ablation studies)
- Each delta lesson $\delta_i$ contains:
  - A **unique identifier** (for deduplication and in-place updates)
  - A **helpfulness counter** (tracks empirical utility across episodes)
  - **Content**: reusable strategy, domain concept, or pitfall warning

### 3. Curator

- **Input:** Current context $C$, new delta lessons $\{\delta_i\}$
- **Output:** Updated context $C'$
- The Curator performs **deterministic merging** (not LLM rewriting) to avoid context collapse:

**Grow-and-Refine Algorithm:**
```
for each δ_i in new lessons:
    if similar entry exists in C (via embedding similarity):
        update existing entry in-place
        increment helpfulness counter
    else:
        append δ_i as new bullet with fresh identifier

if context size > threshold:
    semantic deduplication via embeddings
    prune lowest-helpfulness entries
```

> [!IMPORTANT]
> The deterministic merge (append + in-place update) is the critical design choice that prevents context collapse. Unlike GEPA or Dynamic Cheatsheet, the Curator never rewrites the entire context — only individual bullets are created or updated.

## Incremental Delta Updates vs. Monolithic Rewriting

| Property | Monolithic Rewriting (GEPA, Dynamic Cheatsheet) | ACE Delta Updates |
|---|---|---|
| Update granularity | Entire context rewritten each step | Individual bullets added/updated |
| Information retention | Vulnerable to brevity bias and collapse | Preserved via deduplication and counters |
| Computational cost | Full LLM rewrite per step | O(k) edits where k = number of new lessons |
| Interpretability | Opaque global changes | Auditable per-bullet provenance |
| Retrieval | Full context must be scanned | Fine-grained bullet-level retrieval |

## Offline vs. Online Adaptation Modes

ACE supports two operating modes:

**Offline Adaptation (System Prompt Optimization):**
- Runs multiple training episodes using ground-truth feedback
- Optimizes a shared system prompt used at inference time
- Analogous to few-shot prompt engineering but fully automated

**Online Adaptation (Agent Memory):**
- Operates during inference without ground-truth labels
- Uses execution feedback and environment signals (API responses, error codes) as supervision
- Context updates are applied between episodes in a rolling fashion
- Enables continual learning in deployed agents

## Experiments

- **Datasets:**
  - **AppWorld**: Autonomous agent benchmark with API understanding, code generation, and environment interaction tasks. Split into normal and challenge difficulty levels. Evaluates offline and online adaptation.
  - **FiNER**: Token-level financial named entity classification with 139 XBRL entity types. Tests domain knowledge accumulation.
  - **Formula**: Numerical reasoning over XBRL financial filings requiring formula derivation and calculation.
- **Baseline models:** DeepSeek-V3.1 (primary), GPT-4.1 (comparison)
- **Comparison methods:** Reflexion, GEPA, Dynamic Cheatsheet, IBM CUGA (top AppWorld production agent)
- **Hardware:** Not specified

**Key Results:**

| Setting | Method | Gain over baseline |
|---|---|---|
| AppWorld (offline) | ACE | +12.5% average |
| AppWorld (online) | ACE | +17.1% (no ground-truth labels) |
| FiNER | ACE | +7.6% accuracy |
| Formula | ACE | +18.0% accuracy |
| Adaptation latency | ACE vs. GEPA | 82.3% reduction |
| Token cost | ACE vs. baselines | 83.6% reduction |

> [!NOTE]
> ACE with DeepSeek-V3.1 matches the top-ranked AppWorld production agent (IBM CUGA using GPT-4.1), demonstrating that context evolution can substitute for a larger underlying model in agentic settings.

**Ablation findings (Table 3):**
- Iterative Reflector refinement is the largest single contributor to gains
- Multi-epoch adaptation provides incremental gains beyond single-pass
- Offline warmup is necessary for effective online adaptation (cold-start problem)

## Comparison with Related Methods

| Method | Update strategy | Brevity bias? | Context collapse? | Labeled supervision? |
|---|---|---|---|---|
| Reflexion | Verbal critique appended | No | No | Yes |
| TextGrad | Gradient-like text edits | Yes | Yes | Yes |
| GEPA | Monolithic LLM rewrite | Yes | Yes | Yes |
| Dynamic Cheatsheet | Monolithic LLM rewrite | Yes | Yes | No (online) |
| **ACE** | Deterministic delta merge | **No** | **No** | **No** |

## Limitations

- Requires a "reasonably strong Reflector" — if the Reflector generates low-quality lessons, the context becomes polluted with spurious strategies that degrade performance
- Underperforms on tasks where concise context is beneficial (ACE assumes more detail is better, which is not universally true)
- Requires reliable feedback signals; noisy or misleading signals propagate into accumulated context

> [!CAUTION]
> The helpfulness counter mechanism assumes that bullets used more frequently are more valuable. This heuristic may fail when rare but critical strategies are pruned due to low counter values.
