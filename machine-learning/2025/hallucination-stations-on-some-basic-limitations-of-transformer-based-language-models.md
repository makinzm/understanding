# Meta Information

- URL: [Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](https://arxiv.org/abs/2507.07505)
- LICENSE: [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- Reference: Sikka, V., & Sikka, V. (2025). Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models. arXiv:2507.07505. (Submitted to AAAI-26)

# Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models

## Introduction

LLM hallucinations—instances where a model produces spurious, factually incorrect, or nonsensical outputs—have become a central concern as transformer-based language models are deployed in production systems. Concurrently, "agentic AI" (autonomous or semi-autonomous LLM-based agents executing real-world tasks such as financial transactions, booking travel, or managing equipment) has gained significant commercial traction.

This paper argues that hallucinations and agentic failures are not stochastic bugs but **structural, complexity-theoretic inevitabilities**: when the computational complexity of a requested task exceeds the fixed complexity budget of LLM inference, the model *must* respond incorrectly. The paper formalizes this intuition, provides three illustrative examples, proves a theorem, and discusses practical consequences.

## Computational Complexity of LLM Inference

### LLM Operation Model

An LLM maps an input token sequence to an output token sequence, generating one token at a time. Its vocabulary has $n$ tokens $t_1, t_2, \ldots, t_n$. At each generation step the model attends over the context of $N$ tokens, where each token is a $d$-dimensional embedding vector.

The self-attention mechanism has the following per-step complexity:

```math
\begin{align}
  \mathcal{O}(N^2 \cdot d + N \cdot d^2)
\end{align}
```

Because $N \gg d$ in modern long-context models (Gemini and GPT-4.1 support $\geq 10^6$ tokens), the dominant term is $N^2 \cdot d$, so the paper treats the total inference complexity as:

```math
\begin{align}
  \mathcal{O}(N^2 \cdot d)
\end{align}
```

> [!NOTE]
> The authors empirically confirmed this on Llama-3.2-3B-Instruct: every 17-token prompt — regardless of content — executes exactly 109,243,372,873 floating-point operations, consistent with $N^2 \cdot d$ scaling.

The key insight is that this budget is **fixed by the prompt length $N$ and model dimension $d$**, not by the inherent difficulty of the requested task.

## The Complexity Threshold Argument

If a task encoded in a prompt of length $N$ requires a minimum of $\mathcal{O}(n^k)$ operations (where $n < N$ is the "problem size" embedded in the prompt), and if $n^k$ grows faster than $N^2 \cdot d$, then the LLM *cannot* correctly execute the task. It will still produce an output — token by token, each selected by a $\mathcal{O}(N^2 \cdot d)$ softmax — but that output is not the result of correctly carrying out the task.

This applies to:
- Tasks where the LLM response is *necessarily wrong* (the answer space is too large to enumerate)
- Tasks where the LLM response is *accidentally correct* despite an incorrect procedure

Both cases are treated as hallucinations in this paper.

### Example 1: Token Composition

**Task:** "Given a set of $n$ tokens $\{t_1, t_2, \ldots, t_n\}$, list every string of length $k$ tokens."

The number of such strings is $n^k$, so the minimum computation to enumerate them is:

```math
\begin{align}
  \mathcal{O}(n^k)
\end{align}
```

For $n = 2$, this is $\mathcal{O}(2^k)$ — exponential in $k$. For any $n$ and $k$ large enough that $n^k \gg N^2 \cdot d$, the LLM cannot correctly enumerate the strings. The LLM will generate tokens greedily, picking each by maximum likelihood, but this is not the same as computing the exponentially large combinatorial space.

> [!IMPORTANT]
> $N$ here is the full prompt length (including the task description), while $n$ is the set size encoded *within* the prompt. Since the prompt contains additional tokens to express the task, $n < N$ always holds.

### Example 2: Matrix Multiplication

**Task:** Multiply matrix $A \in \mathbb{R}^{m \times n}$ by matrix $B \in \mathbb{R}^{n \times p}$ to produce $C \in \mathbb{R}^{m \times p}$.

Each entry is:

```math
\begin{align}
  C_{ij} = \sum_{k=0}^{n-1} A_{ik} \cdot B_{kj}
\end{align}
```

The naive algorithm:

```
Initialize C with dimensions m × p
For each row i in A:            # m iterations
    For each column j in B:     # p iterations
        C_ij = Σ_{k=0}^{n-1} A_ik · B_kj   # n steps
```

This runs in $\mathcal{O}(m \cdot n \cdot p) = \mathcal{O}(n^3)$ for square matrices. When $m$, $n$, $p$ exceed the vocabulary size, the LLM cannot correctly execute this multiplication.

Other tasks with cubic or super-cubic complexity include:
- Floyd-Warshall all-pairs shortest paths: $\mathcal{O}(|V|^3)$
- Subset enumeration: $\mathcal{O}(2^n)$
- Ackermann function (Petri net reachability): super-exponential
- Navier-Stokes CFD simulation
- Multi-way join operations in relational databases

### Example 3: Agentic AI and Verification Failure

Agentic AI tasks — financial transactions, logistics scheduling, software verification — can clearly have complexity $> \mathcal{O}(N^2 \cdot d)$, so the first two examples apply directly.

A further problem is that **LLM-based agents cannot verify each other's solutions** when verification itself is hard. Consider:

**Traveling Salesperson Problem (TSP) verification:** Given $n$ cities, verifying that a claimed route $R$ with distance $D_R$ is truly shortest requires comparing $R$ against all $(n-1)!/2$ possible routes (brute force, no precomputed bounds). For $n = 20$, this is $\approx 10^{17}$ candidate routes — far beyond $\mathcal{O}(N^2 \cdot d)$.

The same argument applies to:
- Vehicle routing and bin packing
- Quadratic assignment problems
- Formal model checking (state explosion: system states grow exponentially with component count)
- Software correctness verification

Let $A_1$ be an LLM agent assigned problem $P$ of complexity $\mathcal{O}(n^3)$, and let $A_2$ be an LLM agent assigned to verify $A_1$'s solution. Since all of $A_2$'s operations are bounded by $\mathcal{O}(N^2 \cdot d)$, and since reliable verification of $P$ requires at least $\mathcal{O}(n^3)$ operations, $A_2$ cannot accurately verify $A_1$'s solution.

## Theorem and Proof

### Theorem 1

> [!NOTE]
> **Theorem 1:** Given a prompt of length $N$ which includes a computational task of complexity $\mathcal{O}(n^3)$ or higher, where $n < N$, an LLM (or LLM-based agent) will *unavoidably hallucinate* in its response.

**Proof sketch:** Hartmanis and Stearns's time-hierarchy theorem (1965) states that if $t_2(n)$ is asymptotically larger than $t_1(n)$ (e.g., $t_2(n) = n^2$ and $t_1(n) = n$), then there exist decision problems solvable in $\mathcal{O}(t_2(n))$ but *not* in $\mathcal{O}(t_1(n))$. Since an LLM is limited to $\mathcal{O}(N^2 \cdot d)$ operations, any task requiring $> \mathcal{O}(N^2 \cdot d)$ steps cannot be correctly computed.

**Corollary:** There exist tasks that LLM agents can be asked to perform whose *verification* also cannot be correctly performed by LLMs, because many verification procedures have complexity $> \mathcal{O}(N^2 \cdot d)$.

> [!IMPORTANT]
> The theorem requires the problem size $n$ to be embedded *within* the input prompt (so $n < N$). The complexity bound $\mathcal{O}(n^3)$ in the theorem statement is a concrete threshold; the argument applies to any complexity class that asymptotically exceeds $\mathcal{O}(N^2 \cdot d)$.

## Comparison with Related Work

| Approach | Key claim | Relationship to this work |
|---|---|---|
| Xu et al. 2024 (arXiv:2401.11817) | Hallucination is an innate limitation of LLMs | Complementary; this paper grounds the limitation in computational complexity theory |
| Apple "Illusion of Thinking" (2025) | Reasoning models collapse on high-complexity tasks (e.g., Towers of Hanoi) | Empirical evidence consistent with Theorem 1; Towers of Hanoi requires exponential time |
| Composite/multi-agent systems | Multiple agents can collectively achieve higher ability | Acknowledged; individual LLM complexity bound still holds per agent |
| Formal augmentation (AlphaGeometry, Cyc) | Combining LLMs with rigorous symbolic reasoning | Mitigation strategy; does not remove the complexity ceiling on the LLM component |

## Discussion

The authors frame an LLM's "intelligence" as bounded by $\mathcal{O}(N^2 \cdot d)$: any prompt engineering, few-shot prompting, or chain-of-thought that fits within this budget cannot bridge the gap for tasks that inherently require more computation.

**On reasoning models (o3, R1):** Reasoning models generate a large number of "think" tokens before their final response. The authors argue this does not overcome the complexity ceiling for two reasons:
1. Each individual token generation still runs in $\mathcal{O}(N^2 \cdot d)$, and that single step may itself require higher-complexity computation.
2. The total token budget for reasoning chains is finite and far smaller than what exponential-time tasks require.

**Practical implications:**
- LLMs should not be deployed as sole decision-makers for tasks with $> \mathcal{O}(N^2 \cdot d)$ complexity without external verification mechanisms.
- LLM-based agents cannot reliably audit each other's correctness for complex tasks.
- Composite systems (LLM + symbolic solvers, verified computation, or classical algorithms) remain necessary for accuracy-critical applications.

> [!TIP]
> The time-hierarchy theorem referenced in the proof is: Hartmanis, J., & Stearns, R. E. (1965). On the computational complexity of algorithms. *Transactions of the American Mathematical Society*, 117, 285–306. https://doi.org/10.1090/s0002-9947-1965-0170805-7

# Experiments

- Dataset: None (theoretical paper; no empirical datasets used)
- Hardware: Authors ran Llama-3.2-3B-Instruct on their system to confirm fixed FLOP count per prompt length (17-token prompts yield exactly 109,243,372,873 FLOPs regardless of content)
- Optimizer: N/A
- Results: The empirical FLOP measurement on Llama-3.2-3B-Instruct confirms the $\mathcal{O}(N^2 \cdot d)$ complexity model; code and computation walkthrough available at https://varinsikka.github.io/
