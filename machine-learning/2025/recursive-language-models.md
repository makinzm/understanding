# Meta Information

- URL: [Recursive Language Models](https://arxiv.org/abs/2512.24601)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv preprint arXiv:2512.24601.

# Recursive Language Models (RLMs)

## Overview

Recursive Language Models (RLMs) are an **inference-time strategy** that addresses the fundamental limitation of fixed context windows in LLMs. Instead of feeding a long prompt directly into the neural network, RLMs expose the prompt as an **external environment variable** inside a Python REPL. The model then writes code to inspect, decompose, and recursively invoke itself on selected subsets of the input.

RLMs are applicable when:
- Input length exceeds model context windows (up to two orders of magnitude longer)
- Tasks require dense processing of every document chunk (linear or quadratic complexity)
- The model has strong code generation capabilities

The key insight is that the context window is treated as an **architectural constraint on inference**, not on the data itself. By externalizing the context as a Python variable, the LLM acts as an orchestrator that manages which portions of data to process and when.

## System Architecture

### REPL Environment Setup

The system initializes a Python Read-Eval-Print Loop (REPL) that exposes the full input as a variable:

```python
# What the LLM sees inside the REPL
context = "<full input text, potentially millions of characters>"
context_length = len(context)  # metadata exposed to model
```

The model can then write arbitrary Python code to manipulate this variable: slicing it into chunks, applying regex filters, and calling a provided `llm_query()` function that dispatches sub-LLM calls.

- **Root LLM**: The orchestrator (e.g., GPT-5 or Qwen3-Coder-480B-A35B) that writes the decomposition code
- **Sub-LM**: A potentially smaller model (e.g., GPT-5-mini) that processes individual chunks; constrained to roughly 500K characters per call to fit its context window

### Input/Output Specification

- **Input to Root LLM**: A system prompt describing the REPL interface + a user query + a REPL state showing `context` and any buffer variables
- **Output from Root LLM**: A Python code block that queries sub-LMs and accumulates results
- **Input to Sub-LM**: A string of up to ~500K characters (one chunk of the original context) + a sub-query from the root
- **Output from Sub-LM**: A text answer to the sub-query, stored back into the REPL as a Python variable
- **Terminal condition**: The root LLM wraps its final answer in `FINAL(answer)` or `FINAL_VAR(variable_name)` tags

### Recursive Decomposition Algorithm

The core inference loop proceeds as follows:

```
Algorithm: Recursive Language Model Inference
---------------------------------------------------------------------------
Input:  query Q, long context C (|C| >> context_window_size)
Output: answer A

1. Initialize REPL with context variable: context = C
2. WHILE terminal condition not reached:
   a. LLM reads current REPL state (variables, prior outputs)
   b. LLM generates Python code snippet
   c. Execute code in REPL environment:
      - May slice/filter context: chunk = context[i:j]
      - May invoke sub-LM: result = llm_query(chunk, sub_query)
      - May store results: buffer.append(result)
   d. Update REPL state with execution outputs
3. When LLM outputs FINAL(...) or FINAL_VAR(...):
   A = extract_answer(terminal_expression)
4. Return A
```

> [!NOTE]
> "Code is executed as-is in a fresh Python environment. You must include all necessary imports, data definitions, and context within your code blocks." — System prompt constraint

### Emergent Decomposition Strategies

Without explicit training for this task, models independently developed three categories of behavior:

1. **Filtering via regex**: Before invoking sub-LMs, the root model uses regex queries on the context string to prune irrelevant sections, reducing sub-LM call overhead.
2. **Chunking and recursive sub-calling**: The root model splits the context uniformly, by keyword presence, or by Markdown headers, then calls sub-LMs on each chunk in sequence or conditionally.
3. **Verification loops**: Some trajectories include a second round of sub-LM calls to cross-check partial answers; while useful for accuracy, this inflates token costs.

Additionally, for long-output tasks the model stitches together sub-answers by concatenating sub-LM outputs as Python string variables before producing the final answer.

## Complexity Classes Addressed

RLMs are designed to handle different task complexities relative to input length $n$:

| Complexity | Task Type | Example |
|---|---|---|
| $O(1)$ | Finding a single fact anywhere in the input | Needle-in-a-haystack |
| $O(n)$ | Processing every document exactly once | Semantic aggregation |
| $O(n^2)$ | Pairwise comparisons across all documents | Document pair reasoning |

> [!IMPORTANT]
> Base LLMs degrade significantly beyond their context window for $O(n)$ and $O(n^2)$ tasks, while RLMs maintain performance scaling into the 10M+ token regime.

## Comparison with Similar Approaches

| Approach | Mechanism | Limitation |
|---|---|---|
| **RAG (BM25/dense retrieval)** | Retrieve top-k chunks by similarity | Fails on tasks requiring dense coverage of all documents |
| **Summary Agents** | Iteratively summarize sliding windows | Lossy compression; no recursive sub-calls |
| **CodeAct** | Code-based agent with tool use | No recursive LLM self-invocation |
| **Long-context LLMs** | Extend context window via architecture | Fixed upper bound; quadratic attention cost |
| **RLMs (this work)** | Externalize context to REPL; recursive self-calls | Requires strong coding ability; sequential latency |

The critical difference from RAG: RLMs guarantee **dense coverage** of the entire context because the model explicitly codes loops over all chunks. RAG retrieves only the top-k most similar chunks and may miss relevant information scattered throughout the corpus.

# Experiments

## Datasets

- **S-NIAH** (Single Needle In A Haystack): 50 tasks; finding a single fact buried in a very long distractor context. $O(1)$ complexity.
- **BrowseComp-Plus**: 150 multi-hop QA tasks; each question requires reasoning over 1,000 randomly sampled documents from a 100K-document corpus (6M–11M tokens total). $O(n)$ complexity.
- **OOLONG**: 50 tasks from the `trec_coarse` split; semantic aggregation requiring processing every document chunk. $O(n)$ complexity.
- **OOLONG-Pairs**: 20 custom queries requiring pairwise document comparisons. $O(n^2)$ complexity.
- **LongBench-v2 CodeQA**: Code repository understanding benchmark.

## Models and Baselines

- **Root LLMs**: GPT-5 (as orchestrator) with GPT-5-mini for sub-calls; Qwen3-Coder-480B-A35B (used as both root and sub-LM)
- **Baselines**:
  - Base model with full context (no REPL)
  - RLM without sub-calls (REPL only, model reads context directly)
  - Summary agent (iterative sliding-window summarization)
  - CodeAct + BM25 retrieval

## Key Results

- **OOLONG**: RLM-GPT-5 achieves 28.4% improvement over base GPT-5; RLM-Qwen3 achieves 33.3% improvement.
- **OOLONG-Pairs**: RLM-GPT-5 reaches **58.00% F1** vs. 0.04% for base GPT-5 — a 1450× improvement.
- **S-NIAH**: Both base and RLM achieve near-perfect scores, confirming RLMs do not hurt $O(1)$ tasks.
- **BrowseComp-Plus**: RLMs outperform retrieval-augmented baselines by processing all 1,000 documents rather than top-k.
- **Token cost**: RLMs maintain comparable or lower average token costs than base models, because the root model avoids processing the entire context in its own context window.

## Limitations

- **High variance in inference cost**: Trajectory length varies significantly across instances; some models generate excessive verification loops.
- **Brittle terminal conditions**: The `FINAL(...)` tag-based termination caused reliability issues in practice, particularly for Qwen3-Coder.
- **Sequential execution**: All sub-LM calls were synchronous; asynchronous parallel calls could reduce wall-clock time substantially.
- **Model dependence**: RLMs require models with strong coding and instruction-following capabilities; weaker models fail to generate coherent decomposition code.
- **Suboptimal context management**: Models were observed to be "inefficient decision makers over their context," suggesting that fine-tuning models specifically for RLM-style inference could yield further gains.
