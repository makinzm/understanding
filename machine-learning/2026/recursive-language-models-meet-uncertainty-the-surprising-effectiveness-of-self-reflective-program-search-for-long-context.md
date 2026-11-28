# Meta Information

- URL: [Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context](https://arxiv.org/abs/2603.15653)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Alizadeh, K., Shojaee, P., Cho, M., & Farajtabar, M. (2026). Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context. arXiv:2603.15653.

# Recursive Language Models Meet Uncertainty (SRLM)

## Overview

SRLM (Self-Reflective Language Model) addresses long-context reasoning in LLMs by replacing recursive program decomposition with **uncertainty-guided trajectory selection**. Instead of relying on explicit sub-calls (as in Recursive Language Models, RLM), SRLM samples $K$ independent candidate programs and selects the best one based on three intrinsic uncertainty signals derived from the model's own generation process—requiring no external verifier, reward model, or labeled data.

The key insight challenges the prevailing assumption that recursion drives RLM performance: **self-reflective selection mechanisms can match or exceed recursive decomposition**, while also being more robust to short contexts where RLM degrades.

> [!NOTE]
> "Recursion itself is not the primary driver of performance" — the self-reflection mechanism provides at least as strong a signal for trajectory selection as hierarchical decomposition.

**Applicability**: Practitioners working on retrieval-augmented generation, agent-based long-document QA, or any LLM pipeline that must process contexts far exceeding the native context window ($N \gg L$).

## Problem Formulation

Let $\mathcal{C} = (c_1, c_2, \ldots, c_N)$ be a long context with $N \gg L$ tokens (where $L$ is the model's context window). A **context-interaction program** $p = (p_1, p_2, \ldots, p_T)$ consists of $T$ executable operations. Each step's execution state is updated as:

```math
\begin{align}
  e_t = \text{Exec}(p_t,\; e_{t-1},\; \mathcal{C})
\end{align}
```

where $e_t$ is the execution state after step $t$. The model operates in a REPL environment, issuing programmatic queries against the externalized long context.

## Three Uncertainty Signals

SRLM samples $K = 8$ independent program trajectories $\mathcal{P} = \{p^{(1)}, \ldots, p^{(K)}\}$ from policy $\pi_\theta$ and scores them with three signals:

### 1. Self-Consistency (Sampling-Based)

Empirical answer frequency across $K$ samples:

```math
\begin{align}
  \text{prob}(a) = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}[\text{out}(p^{(k)}) = a]
\end{align}
```

The plurality answer $\hat{a} = \arg\max_a \text{prob}(a)$ defines the **consistent set** $\mathcal{S} = \{p^{(k)} \in \mathcal{P} : \text{out}(p^{(k)}) = \hat{a}\}$. Selection is then restricted to programs in $\mathcal{S}$.

### 2. Verbalized Confidence (Semantic Signal)

At each step $t$, the model reports a step-level confidence $\nu_t^{(k)} \in (0, 100]$. The aggregate verbalized confidence is:

```math
\begin{align}
  \text{VC}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \log\!\left(\frac{\nu_t^{(k)}}{100}\right) \leq 0
\end{align}
```

Values closer to zero indicate higher overall confidence across all reasoning steps.

### 3. Behavioral Uncertainty (Trace Length)

Total token count of the reasoning trace as a proxy for epistemic effort:

```math
\begin{align}
  \text{Len}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \ell_t^{(k)}
\end{align}
```

Longer traces signal that the model expended more effort resolving ambiguity, correlating with higher uncertainty.

## Joint Selection

The two fine-grained signals are combined into a single score:

```math
\begin{align}
  s(p) = \text{VC}(p) \cdot \text{Len}(p)
\end{align}
```

Lower $s(p)$ (i.e., more negative) indicates a worse candidate. The optimal program is:

```math
\begin{align}
  p^* = \arg\max_{p \in \mathcal{S}} s(p)
\end{align}
```

with final prediction $\hat{y} = \text{out}(p^*)$.

> [!IMPORTANT]
> The three signals are complementary: ablation studies show that combining all three substantially outperforms any individual signal, confirming that self-consistency, verbalized confidence, and trace length capture distinct aspects of uncertainty.

## Algorithm: SRLM Inference

**Input**: Query $q$, long context $\mathcal{C}$, LLM policy $\pi_\theta$, sample count $K$
**Output**: Final prediction $\hat{y}$

1. Sample $K$ independent program trajectories: $\mathcal{P} = \{p^{(k)}\}_{k=1}^K$ from $\pi_\theta(q, \mathcal{C})$
2. For each $p^{(k)}$: execute in REPL to obtain output $\text{out}(p^{(k)})$ and collect $\{\nu_t^{(k)}, \ell_t^{(k)}\}_{t=1}^{T^{(k)}}$
3. Compute plurality answer $\hat{a}$ and consistent set $\mathcal{S}$
4. For each $p^{(k)} \in \mathcal{S}$: compute $\text{VC}(p^{(k)})$ and $\text{Len}(p^{(k)})$
5. Select $p^* = \arg\max_{p \in \mathcal{S}} \text{VC}(p) \cdot \text{Len}(p)$
6. Return $\hat{y} = \text{out}(p^*)$

## Comparison with RLM

| Aspect | RLM (Recursive LM) | SRLM |
|---|---|---|
| Core mechanism | Hierarchical sub-call decomposition | Parallel sampling + uncertainty selection |
| Context handling | Recursively splits long context | Programs query externalized context |
| Selection criterion | None (single trajectory) | Three uncertainty signals over $K$ trajectories |
| Short-context behavior | Degrades below native window | Robust across all context lengths |
| Semantic tasks | Struggles (structurally biased) | More uniform across task types |
| Recursion required | Yes | Optional (sub-calls can be disabled) |

> [!NOTE]
> RLM frequently underperforms the base model within its native context window ($< 131$K tokens), indicating that forced recursion on manageable contexts introduces overhead. SRLM avoids this by not requiring recursive decomposition.

> [!TIP]
> Related methods: CodeAct (code-execution agents), ReSum (summarization-based agents), Mem0/G-Memory (memory-augmented agents), and confidence estimation via self-consistency (Wang et al., 2022).

## Experiments

### Datasets

| Dataset | Instances | Context Range | Task Type | Metric |
|---|---|---|---|---|
| **BrowseComp-Plus** | 150 | 6M–11M tokens | Multi-hop QA over 1,000 documents | Accuracy |
| **OOLONG** (trec_coarse) | 650 (50 per length bin) | 1K–8M tokens | Aggregation tasks | Partial credit: $0.75^{|y - \hat{y}|}$ |
| **LongBench-v2** | ~500 | 8K–4.2M tokens | Multiple-choice; 6 domains (Code QA, Dialogue, Documents) | Accuracy |

- Hardware: Not explicitly stated
- Backbone models: Qwen3-Coder-480B and GPT-5
- Evaluation judge: GPT-5-mini (semantic judge for open-ended answers)
- Time budget: 600 seconds per trajectory step; max 30 program interactions
- Token limit: 260K for Qwen3-Coder-480B

### Key Results

| Benchmark | RLM (Qwen3) | SRLM (Qwen3) | RLM (GPT-5) | SRLM (GPT-5) |
|---|---|---|---|---|
| BrowseComp-Plus | 37.1% | **59.7%** (+22.6%) | 86.0% | **92.4%** (+6.4%) |
| OOLONG (≥131K) | 45.7% | **51.8%** (+6.1%) | 53.0% | **65.5%** (+12.5%) |
| LongBench CodeQA | 59.8% | **64.9%** (+5.1%) | 59.5% | **68.9%** (+9.4%) |

SRLM maintains consistent gains from 1K to 8M token contexts, while RLM performance degrades on contexts within the native window.

**Task-type analysis**: Recursion helps most on structured/search-oriented tasks (Code QA); SRLM provides more uniform gains across semantically complex tasks (Dialogue, Document understanding).

**Pareto efficiency**: SRLM outperforms recursive RLM on long contexts ($\geq 131$K tokens) in both accuracy and wall-clock time, meaning reflection is not only more accurate but also cheaper than forced recursion.
