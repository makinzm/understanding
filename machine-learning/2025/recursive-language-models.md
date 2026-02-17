# Meta Information

- URL: [Recursive Language Models](https://arxiv.org/abs/2512.24601)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601.

# Overview

Recursive Language Models (RLMs) are an **inference paradigm** that enables LLMs to process inputs far exceeding their context window by treating the prompt as an external environment variable rather than a direct neural network input. Instead of passing $N$ tokens directly to the model, the RLM runs the LLM in a Python REPL where it writes code to programmatically decompose and query itself recursively over slices of the input.

> [!NOTE]
> "RLMs allow models to handle prompts up to two orders of magnitude beyond model context windows while remaining cost-competitive with alternative approaches."

## Applicability

- **Who**: Practitioners who need to process long documents (legal corpora, codebases, large retrieval sets) with fixed-context LLMs
- **When**: At inference time, requiring no model fine-tuning
- **Where**: Any API-accessible LLM that can execute Python code via a code interpreter tool

# Core Architecture

## REPL Environment Setup

The RLM initializes a Python REPL with:

| Variable / Function | Description |
|---|---|
| `context` | Full input string (up to 10M+ characters) loaded as a Python variable |
| `llm_query(prompt, context_str)` | Recursive sub-LM call; handles ~500 K characters per invocation |
| `print(...)` | Allows iterative inspection and intermediate reasoning |

**Execution flow**:
1. The root LLM receives a task description and a reference to `context` (not the full text)
2. It writes Python code to inspect, filter, and chunk `context` programmatically
3. Each `llm_query()` call dispatches a sub-LM over a snippet (Input: `str`, Output: `str`)
4. The root LLM aggregates sub-call results via code (string concatenation, dict accumulation, etc.)
5. Final answer is printed to stdout and returned

**Input/Output specification**:

| Component | Input | Output |
|---|---|---|
| Root LLM | Task + variable name `context` (small prompt) | Python code + `print(answer)` |
| Sub-LM (`llm_query`) | Snippet $s_i \in \text{str}$, sub-task prompt | Answer string |
| REPL | Python source code | Stdout (final answer) |

## Pseudocode

```
Algorithm: RLM Inference
Input: long_input L, task T, base_model M, sub_model M_sub
Output: answer A

1. Initialize REPL with:
     context = L
     def llm_query(prompt, ctx): return M_sub.complete(prompt + ctx)

2. system_prompt = build_system_prompt(T, examples)
3. code = M.complete(system_prompt + "context is loaded as variable `context`")

4. Execute code in REPL:
     for each llm_query(p, s) call in code:
         result = M_sub.complete(p + s)   # recursive sub-call
     collect results into variables

5. A = stdout of REPL execution
6. Return A
```

## Emergent Decomposition Strategies

Models spontaneously develop the following strategies (without explicit training):

1. **Regex filtering**: Use `re.findall()` on `context` based on model priors, then sub-query only matching segments
2. **Uniform chunking**: Split `context` into equal-sized chunks $[s_1, s_2, \ldots, s_k]$ and issue parallel sub-calls
3. **Verification loops**: Sub-LM sub-calls validate answers, sometimes redundantly
4. **Variable-based output construction**: For long outputs, progressively accumulate results: `answer += llm_query(...)`

# Experiments

- **Datasets**:
  - **S-NIAH** (Single Needle-In-A-Haystack): Constant-complexity single fact retrieval
  - **BrowseComp-Plus**: Multi-hop QA over 1,000 web documents (6 M–11 M tokens); linear complexity
  - **OOLONG**: Semantic aggregation task over 131 K tokens; linear information density
  - **OOLONG-Pairs**: Pairwise reasoning over 32 K tokens; quadratic complexity
  - **LongBench-v2 CodeQA**: Repository-level code understanding (23 K–4.2 M tokens)
- **Hardware**: Not explicitly specified; all LM calls made via API
- **Models**:
  - GPT-5 (medium reasoning) as root + GPT-5-mini as sub-LM
  - Qwen3-Coder-480B-A35B (open-source alternative)
- **Baselines**:
  - Direct base model call (no RLM)
  - Summary agent (iterative lossy compression)
  - CodeAct + BM25 retriever
  - RLM without sub-calls (ablation)

## Key Quantitative Results

| Task | GPT-5 RLM | Qwen3-Coder RLM | Base GPT-5 |
|---|---|---|---|
| CodeQA | 62% | 56% | ~16% |
| BrowseComp-Plus | **91.33%** | 44.66% | 0% |
| OOLONG | 56.50% | 48% | ~24–44% |
| OOLONG-Pairs (F1) | **58%** | 23.11% | 0.04% |

> [!IMPORTANT]
> On BrowseComp-Plus (6–11M tokens), the base GPT-5 achieves 0% because the context exceeds its window. RLM achieves 91.33% by recursively querying sub-LMs over document chunks.

## Cost Analysis

- GPT-5 RLM average cost on BrowseComp-Plus: **$0.99** (vs. $1.31 for summary agent)
- RLM is up to **3× cheaper** than summarization baselines while outperforming them
- Cost scales proportionally to task complexity (constant < linear < quadratic)
- High variance in outlier trajectories (long recursive chains); median cost competitive with base model

# Differences from Similar Algorithms

| Method | Input Handling | Information Loss | Training Required |
|---|---|---|---|
| **RLM** (this work) | Programmatic sub-queries over full input | Lossless (selective access) | No |
| Long-context fine-tuning | Extend context window directly | None, but limited by window size | Yes |
| MemWalker / ReSum | Iterative lossy compression | Yes (summarization loses detail) | No |
| MemGPT | Memory management + retrieval | Partial (retrieval misses) | No |
| ViperGPT / THREAD | Task decomposition via code | N/A (focuses on task, not input) | No |
| RAG + BM25 | Retrieval-based chunking | Yes (lexical retrieval misses) | No |

> [!NOTE]
> Prior task-decomposition methods (ViperGPT, ReDel) decompose the *task* into sub-tasks. RLMs decompose the *input* into sub-contexts, a fundamentally different axis of scaling.

# Limitations

- **Synchronous execution**: Sub-calls block the root LLM; asynchronous implementation could reduce latency
- **Recursion depth = 1**: Only one level of recursive sub-calling explored; deeper recursion (sub-sub-LMs) untested
- **No training signal**: Models not explicitly trained as RLMs; training (e.g., bootstrapped from frontier models) could improve efficiency and reduce excessive sub-call overhead (Qwen3-Coder issue)
- **High-variance costs**: Outlier trajectories can be expensive; cost predictability is limited

# System Prompt Differences by Model

| Model | Prompt Guidance |
|---|---|
| GPT-5 | Flexible sub-call usage; extensive examples for filtering, chunking, verification |
| Qwen3-Coder | Explicit warning against excessive `llm_query` calls; recommends batching ~200 K characters per call |

This model-specific tuning is necessary because Qwen3-Coder tends to issue excessive sub-calls without guidance, inflating cost without accuracy gains.
