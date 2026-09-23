# Meta Information

- URL: [Reasoning Structure of Large Language Models](https://arxiv.org/abs/2606.03883)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) (CC-BY 4.0)
- Reference: Berdoz, F., Lanzendörfer, L. A., Farestam, F., & Wattenhofer, R. (2026). Reasoning Structure of Large Language Models. ICML 2026.
- Code: https://github.com/ETH-DISCO/llm-reasoning-efficiency

# Reasoning Structure of Large Language Models

## Overview

Large Reasoning Models (LRMs) are commonly evaluated with two metrics: final-answer accuracy and token count. Both are insufficient because two models with identical scores can exhibit fundamentally different reasoning structures — one may follow a tight deductive chain while the other wanders through redundant claims and verification loops before reaching the same answer.

This paper introduces three things:

1. A **scalable benchmark** of 21 deterministic 2D grid logic puzzles at four difficulty levels.
2. A **pipeline** that converts free-form textual reasoning traces into verifiable *reasoning graphs* of atomic claims and deductive dependencies.
3. A **reasoning flow efficiency metric** $\eta \in [0, 1]$ that measures how concentrated a model's logical mass is relative to the minimal claim set needed to specify the solution.

The framework is grounded in an executable puzzle environment, so every extracted claim is verified deterministically — not just the final answer. Analysis on open-source models (Qwen3 235B, DeepSeek V3.2, Kimi K2) and GPT-5 shows that $\eta$ separates reasoning behaviors that accuracy and token count conflate.

> [!TIP]
> Code and puzzle environment: https://github.com/ETH-DISCO/llm-reasoning-efficiency

---

## Background and Motivation

### Large Reasoning Models (LRMs)

LRMs exploit test-time compute via long Chain-of-Thought (CoT) traces. They have shown strong results on coding, logical deduction, math, and spatial reasoning. However, most evaluations reduce multi-step reasoning to one-dimensional scalars — final accuracy or trace length — masking *how* the model reasoned.

### Prior Work on Trace Analysis

| Approach | What it measures | Limitation |
|---|---|---|
| Accuracy metrics | Final answer correctness | Ignores reasoning path |
| Token count | Verbosity of trace | Uncorrelated with quality |
| ProofWriter / LogicBench | Multi-step inference (text) | No executable verification |
| SATBench | SAT formula solving | Text-centric, no state machine |
| ZebraLogic | Logic grid puzzle accuracy | Outcome only |
| Landscape embeddings | Convergence visualization | No environment grounding |
| Tree-jump / DAG metrics | Exploration & backtracking | Task-specific, not general |

This work adds an **environment-grounded** structural layer on top of outcome metrics, enabling claim-level verification across arbitrary puzzle families.

---

## Benchmark: PUZZLE

The benchmark is built on a scalable grid-based puzzle RL environment derived from Simon Tatham's puzzle collection. 21 puzzle families are included, spanning diverse constraint types:

- **Placement** puzzles (e.g., Tents — place tents in a grid near trees)
- **Connectivity** puzzles
- **Counting** constraint puzzles
- **Latin-square-style** puzzles (e.g., Sudoku variants)

Each puzzle has four difficulty levels: *Trivial*, *Human easy*, *Human normal*, *Human hard*. Difficulty is controlled by grid size and clue density, not rule changes. For each (puzzle, difficulty) pair, 5 fixed instances are evaluated; all models see the same instances.

> [!NOTE]
> "Logic puzzles have long attracted human curiosity because they are 'easy to learn, but hard to master.' Unlike many real-world tasks, they are fully specified and admit unambiguous verification. Their difficulty can be scaled without changing the underlying rules."

**Key advantage over prior benchmarks**: the executable environment validates *intermediate* claims, not only the submitted final solution.

---

## Methodology

### Input / Output

- **Input**: A free-form textual reasoning trace $S = (s_1, s_2, \dots)$ produced by an LRM on a specific puzzle instance, where each $s_i$ is a sentence.
- **Output**: A directed acyclic reasoning graph $G = (V, E)$ where:
  - $V \subseteq \mathcal{C} \times \mathbb{N}$: claim occurrences (claim $c$ asserted in sentence $s_i$)
  - $E$: edges representing inference (premise → conclusion) or restatement links
  - Each claim in $V$ is verified against the executable environment

### Step 1 — Claim Extraction

A two-pass hybrid extraction balances precision and recall:

1. **Deterministic extractor** (high precision): pattern-matching to extract claims; LLM fills in schema violations and missing implied fields.
2. **LLM-based extractor** (high recall): unconstrained LLM extraction without rules.

Both operate on token-balanced chunks of the trace. After chunk-level extraction:

- Exact-duplicate candidates are removed locally.
- An LLM verifies claims in batches of 200 against a localized support window around source text; unsupported or ill-formed claims are discarded.
- Conservative normalization is applied without collapsing repeated events.
- Claims are globally deduplicated, ordered by trace position, and assigned identifiers.

Claim types are puzzle-specific (e.g., "Cell (3,4) contains value 9" for Sudoku; "Column 5 already has a 4" is a partial-state claim).

### Step 2 — Rule Extraction

For each non-tentative claim, the pipeline attempts to find a single rule application explaining how it follows from earlier claims:

- Claims are processed in trace order.
- Context = truncated trace prefix up to the claim's last supporting sentence + all previously extracted claims.
- LLM is prompted to return either one rule application (with explicit premises) or no rule (claim is "directly stated").
- If a premise is referenced but absent from the trace, a placeholder claim is inserted to mark the gap.
- If the same claim appeared before, it is labeled *restated* with a direct edge from its prior occurrence.

### Step 3 — Claim Verification

Each verifiable claim is independently checked against the executable puzzle environment. The environment returns a deterministic correctness label. Labels: **verified correct** (green), **verified wrong** (red), **unverifiable** (grey), **tentative** (orange).

> [!IMPORTANT]
> Verification is deterministic, not LLM-based. This makes the correctness signal reliable and comparable across models.

---

## Reasoning Graph Formalism

### Reasoning Graph $G = (V, E)$

- $V \subseteq \mathcal{C} \times \mathbb{N}$: each vertex $v = (c, i)$ is claim $c$ asserted in sentence $s_i$; uniqueness guaranteed (same claim cannot appear twice in the same sentence).
- $E$: directed edges from premises to conclusions (inference) or from prior to restated occurrence (restatement).
- All premises of an inference appear before the conclusion in $S$, so $G$ and all subgraphs are DAGs.
- Restated static claims are merged into the first occurrence for metric computation.

### Minimal Claim Set $C^*$

$C^*$ is the minimal set of claims that fully determines the puzzle solution — the formatted statement of the solution. If the model solves the puzzle, $C^* \subseteq \{c \mid \exists (c, s) \in V\}$.

### Solution Subgraph $G_{\text{sol}}$

```math
\begin{align}
V_{\text{sol}} &= V^*_\text{sol} \cup \{ v \in V \mid \exists\, u \in V^*_\text{sol} : v \leadsto_G u \} \\
E_{\text{sol}} &= \{ (u,v) \in E \mid u, v \in V_{\text{sol}} \}
\end{align}
```

where $V^*_\text{sol}$ is the set of *first occurrences* of claims that belong to $C^*$, and $v \leadsto_G u$ means there exists a directed path from $v$ to $u$ in $G$. Intuitively: all nodes that directly or indirectly contribute to the final solution.

### Verification Subgraph $G_{\text{ver}}$

```math
\begin{align}
V'_{\text{ver}} &= V^*_\text{sol} \cup \{ v \in V \setminus V_{\text{sol}} \mid \exists\, u \in V^*_\text{sol} : u \leadsto_G v \} \\
V_{\text{ver}} &= V'_{\text{ver}} \cup \{ v \in V \setminus V_{\text{sol}} \mid \exists\, u \in V'_{\text{ver}} : v \leadsto_G u \} \\
E_{\text{ver}} &= \{ (u,v) \in E \mid u, v \in V_{\text{ver}} \}
\end{align}
```

Intuitively: descendants of the solution claims and all their ancestors not in $V_{\text{sol}}$ — the reasoning the model spent verifying rather than constructing the solution.

---

## Reasoning Flow Efficiency Metric $\eta$

### Absorbing Markov Chain on the DAG

The model's reasoning process is modeled as an **absorbing Markov chain** on the claim graph. An absorbing state $a$ is added by connecting every leaf node to it:

```math
\begin{align}
G_{\text{abs}} = (V \cup \{a\},\; E \cup \{(v, a) \mid v \in V,\, d^+(v) = 0\})
\end{align}
```

The transition matrix $P$ is the row-normalized adjacency matrix of $G_{\text{abs}}$, partitioned as:

```math
\begin{align}
P = \begin{pmatrix} Q \in \mathbb{R}^{|V| \times |V|} & R \in \mathbb{R}^{|V| \times 1} \\ \mathbf{0} \in \mathbb{R}^{1 \times |V|} & 1 \end{pmatrix}
\end{align}
```

where $Q$ is the transient component and $R$ is the absorbing component.

### Logical Mass Distribution

Initial logical mass $\boldsymbol{\pi}$ is uniform over source nodes $V_{\text{src}} = \{v \in V \mid d^-(v) = 0\}$:

```math
\begin{align}
\pi(v) = \begin{cases} \frac{1}{|V_{\text{src}}|} & v \in V_{\text{src}} \\ 0 & \text{otherwise} \end{cases}
\end{align}
```

Total logical mass through each node is given by:

```math
\begin{align}
\mathbf{m} = \boldsymbol{\pi} N, \quad N = \sum_{t=0}^\infty Q^t
\end{align}
```

where $N$ is the **fundamental matrix** of the absorbing chain.

### Structural Entropy

```math
\begin{align}
H_{\text{str}}(G) = -\sum_{v \in V} \frac{m(v)}{\|\mathbf{m}\|} \log\left(\frac{m(v)}{\|\mathbf{m}\|}\right)
\end{align}
```

Low entropy = reasoning mass concentrated on a single deterministic path. High entropy = mass scattered across many competing claims and inference branches. Restated and unused claims diffuse logical flow, increasing entropy.

### Reasoning Flow Efficiency

```math
\begin{align}
\eta = \frac{\log|V| - H_{\text{str}}(G)}{\log|V| - \log|C^*|}
\end{align}
```

- **$\eta \approx 1$**: logical flow is highly concentrated; the model reasons in a focused chain toward the minimal solution set.
- **$\eta \approx 0$**: logical flow is diffuse; the model explores many irrelevant branches and accumulates verification overhead.
- Denominator $\log|V| - \log|C^*|$ normalizes out graph scale, making $\eta$ comparable across puzzle sizes and difficulty levels.

> [!NOTE]
> The normalization removes the confound of puzzle size. A small trivial puzzle and a large hard puzzle both yield $\eta \in [0,1]$.

**Difference from token count**: token count is a proxy for verbosity; $\eta$ measures the *structure* of logical flow. They are essentially uncorrelated ($r = -0.05$, $p = 0.64$).

**Difference from simpler graph metrics**: Width, $|V|$, depth, and diameter all correlate *negatively* with accuracy and track puzzle difficulty. $\eta$ is the only metric that correlates *positively* with accuracy while remaining uncorrelated with token count because its size normalization factors out graph scale.

---

## Graph Extraction Pipeline (Algorithm)

```
Input: reasoning trace S = (s1, s2, ...), puzzle instance, solution C*
Output: reasoning graph G = (V, E), verified claim set

Stage 1 — Claim Extraction:
  1. Segment S into sentences; partition into token-balanced chunks.
  2. For each chunk:
     a. Run deterministic extractor → candidate claims with schema
     b. Run LLM extractor (unconstrained) → additional candidates
     c. Remove local exact-duplicate candidates
  3. Verify all candidates in batches of 200 with LLM + source window:
     - Discard unsupported or ill-formed claims
  4. Globally deduplicate; order by trace position; assign IDs → V_raw

Stage 2 — Rule Extraction:
  For each claim c in V_raw (in trace order):
    context = trace prefix up to c's last supporting sentence
            + all previously extracted claims
    if c appears before in trace:
      label c as "restated"; add restatement edge from prior occurrence
    else:
      prompt LLM to find single rule application linking c to premises
      if rule found: add edge (premise, c) for each premise
      if premise missing from trace: insert placeholder node
      if no rule: label c as "directly stated"

Stage 3 — Claim Verification:
  For each verifiable claim in V_raw:
    label = env.check(claim, puzzle_instance, solution)
    # deterministic; returns: correct / wrong / unverifiable / tentative
  Attach labels to nodes.

Post-processing:
  Merge restated static claims into first occurrence for metrics.
  Construct G = (V, E) as DAG.
```

---

## Experiments

### Dataset

| Attribute | Details |
|---|---|
| Puzzle families | 21 grid puzzles from Simon Tatham's collection |
| Difficulty levels | Trivial / Human easy / Human normal / Human hard |
| Instances per (puzzle, difficulty) | 5 fixed instances, same IDs across all models |
| Constraint types | Placement, connectivity, counting, Latin-square |

### Models Evaluated

- **GPT-5** (closed-source; outcome metrics only, traces not extracted)
- **Qwen3 235B** (open-source; graphs extracted)
- **DeepSeek V3.2** (open-source; graphs extracted)
- **Kimi K2** (open-source; graphs extracted)

Graph extraction uses **GPT-5.2** for claim extraction/screening and **GPT-5-mini** for rule extraction. Open-source models only, since closed-source models do not expose traces.

### Hardware / Decoding

- Temperature $T = 1$ (fixed across all models for consistency)
- No explicit hardware details reported; compute for puzzle solving and extraction is reported via token counts

### Key Results: Accuracy and Token Count by Difficulty

| Model | Trivial Acc (%) | Trivial Tok | H.Easy Acc (%) | H.Easy Tok | H.Normal Acc (%) | H.Normal Tok | H.Hard Acc (%) | H.Hard Tok | Avg Acc (%) | Avg Tok |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-5 | 83.8 | 4,154 | 69.5 | 10,180 | 58.1 | 17,274 | 5.7 | 19,862 | 54.3 | 12,867 |
| Qwen3 235B | 69.5 | 10,257 | 44.8 | 19,033 | 21.0 | 23,104 | 0.0 | 23,609 | 33.8 | 19,001 |
| DeepSeek V3.2 | 77.1 | 7,695 | 53.3 | 20,633 | 44.8 | 27,038 | 0.0 | 36,787 | 43.8 | 23,038 |
| Kimi K2 | 77.1 | 10,602 | 56.2 | 29,714 | 41.0 | 43,751 | 1.0 | 61,307 | 43.8 | 36,343 |

### Key Results: $\eta$ vs Structural Metrics (Pearson correlations)

| Metric | vs. Accuracy | vs. $\eta$ |
|---|---|---|
| Depth | −0.263 | +0.046 |
| Diameter | −0.329 | +0.010 |
| Avg. path length | −0.182 | +0.051 |
| Width | −0.618 | −0.431 |
| $|V|$ (graph size) | −0.666 | −0.419 |
| Token count | −0.576 | −0.120 |
| $\eta$ | **+0.368** | — |

### Graph Extraction Stability (same trace, 3 repeated runs)

| Model | Unique claims | Jaccard overlap | $H_{\text{str}}$ | $\eta$ |
|---|---|---|---|---|
| Kimi K2 Thinking | 22.7 | 0.89 ± 0.08 | 3.184 ± 0.191 | 0.946 ± 0.059 |
| DeepSeek V3.2 | 25.0 | 0.79 ± 0.07 | 3.573 ± 0.092 | 0.840 ± 0.022 |
| Qwen3 235B Thinking | 37.7 | 0.98 ± 0.01 | 3.660 ± 0.063 | 0.820 ± 0.014 |

---

## Key Findings

### 1. More tokens ≠ better reasoning

Kimi K2 consistently uses the largest token budgets but does not outperform GPT-5. GPT-5 achieves the best accuracy at every difficulty while remaining the most token-efficient. Reasoning-flow efficiency $\eta$ is essentially uncorrelated with token count ($r = -0.05$, $p = 0.64$).

### 2. Extra tokens translate into verification overhead

Longer traces primarily manifest as increased verification overhead: token count correlates strongly with $|V_{\text{ver}}|/|V_{\text{sol}}|$ ($r = 0.53$, $p = 3 \times 10^{-9}$). Additional compute is spent on checking, not on expanding the core solution-supporting chain.

### 3. $\eta$ tracks solution-focused reasoning

$\eta$ increases with the solution-supporting fraction of the graph ($r = 0.55$, $p = 1.1 \times 10^{-9}$) and decreases as the overall graph grows ($r = -0.33$, $p = 0.001$), consistent with graph bloat from branching, detours, and redundant structure.

### 4. Early errors are associated with inefficiency

Traces where the first incorrect claim appears *later* tend to be more efficient ($r = 0.28$, $p = 0.015$). Early errors induce drift and trigger corrective exploration that lowers $\eta$.

### 5. Some redundancy is beneficial

Efficiency correlates positively with the average number of restatements per unique claim ($r = 0.27$, $p = 0.0078$). Structured restatement of key constraints ("anchoring") is different from aimless repetition — it slightly concentrates logical flow.

### 6. Hard difficulty remains largely unsolved

All models reach $\approx 0$–$5.7\%$ accuracy on Human hard, despite token budgets of 20k–61k. Increasing compute does not resolve the fundamental capability ceiling.

---

## Comparison with Similar Approaches

| Method | Grounding | Intermediate verification | Puzzle-agnostic metrics | What is measured |
|---|---|---|---|---|
| ZebraLogic | None | No | Yes | Accuracy vs. constraint count |
| Enigmata | Final answer only | No | Yes | Generator-verifier pairs for RLVR |
| Tree-jump / DAG metrics | None | No | Partially | Exploration and backtracking |
| Landscape embeddings | None | No | No | Convergence visualization |
| **This work** | Executable environment | **Yes (deterministic)** | **Yes** | Structural entropy + flow efficiency |

> [!IMPORTANT]
> The structural layer (graph construction, Markov chain, $\eta$) is **fully puzzle-agnostic**. Only claim types and verification logic are puzzle-specific. This generalizes to any setting with verifiable intermediate steps: math (symbolic checkers), code (unit tests), or agentic tool-use.

---

## Limitations

1. **LLM-based extraction is non-deterministic**: Variance is modest (Jaccard ≈ 0.79–0.98; $\eta$ CV ≈ 1.9% across 6 extractors), but extraction is not fully static.
2. **Domain-specific claim types**: Each puzzle family requires custom claim and rule definitions. This is inherent to process-level evaluation of reasoning traces.
3. **Graphs only from traces**: Closed-source models that do not expose CoT traces cannot have graphs extracted.
4. **Metric gaming risk**: Optimizing $\eta$ in training could produce superficially focused traces that are not actually more correct.

---

## Conclusion

The paper demonstrates that reasoning can be treated as a *structured, measurable object*. By converting free-form LRM traces into verifiable DAGs and computing reasoning-flow efficiency $\eta$ via an absorbing Markov chain model, the framework exposes behavioral differences that accuracy and token count cannot. The key insight: extra tokens primarily become verification overhead; they do not systematically improve the focus or correctness of solution-supporting reasoning chains.

The framework extends naturally to math reasoning (symbolic intermediate verification), code generation (unit tests), and agentic settings. If latency improves, $\eta$ could also serve as a shaping reward in RLVR post-training to encourage focused, solution-oriented reasoning without sacrificing correctness.
