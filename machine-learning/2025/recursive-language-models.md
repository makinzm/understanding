# Meta Information

- URL: [Recursive Language Models](https://arxiv.org/abs/2512.24601)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601.

# Recursive Language Models

## Overview

Recursive Language Models (RLMs) are an **inference-time strategy** that enables large language models (LLMs) to process inputs that are orders of magnitude longer than their context windows, without any additional training or fine-tuning. Instead of feeding a long prompt directly into the model's neural network, RLMs expose the input as an **external environment variable** (a string in a Python REPL) and let the model write code to examine, filter, and decompose the prompt—recursively calling itself on sub-segments as needed.

**Applicability:**
- Used by developers and researchers who need LLMs to reason over very long documents (millions of tokens) without requiring larger context windows
- Applicable when the input exceeds the model's native context window (e.g., codebases, legal corpora, large document collections)
- Requires a model with strong code-writing capabilities (Python)

## Problem: Context Window Bottleneck

LLMs have a fixed context window $C$ (typically $10^4$–$10^6$ tokens). Processing long inputs $x$ where $|x| \gg C$ is not natively supported. Existing approaches include:

| Approach | Mechanism | Limitation |
|---|---|---|
| Retrieval-Augmented Generation (RAG) | BM25 / dense retrieval selects relevant chunks | Retrieval quality depends on query; misses cross-chunk reasoning |
| Summarization Agents | Iteratively compress context | Lossy; discards fine-grained details |
| Extended Context Models | Train or fine-tune on longer inputs | Expensive; still bounded by hardware |
| **RLM** | REPL environment + recursive sub-calls | Requires strong coding LLM; high cost variance |

## Core Architecture: The RLM Framework

### Inputs and Outputs

- **Input:** Long prompt $x$ with $|x| \gg C$ tokens, and a task description $t$
- **Output:** Answer $a$ produced by the LLM after programmatic decomposition

### Environment Setup

The LLM is placed in a **Python REPL** initialized with:

```python
# Environment variables exposed to the LLM
prompt: str         # The full long input (NOT truncated into the context window)
llm_query: Callable # Function to make a recursive sub-LLM call on a shorter string
```

The LLM **never sees** `prompt` directly in its neural context. Instead, it observes **truncated previews** of `prompt` via REPL stdout (e.g., first $k$ characters), and must write Python code to systematically extract the information it needs.

### Recursive Call Semantics

A sub-LLM call is defined as:

$$\text{llm\_query}(s) \rightarrow r$$

where $s \in \mathbb{R}^{|s|}$ is a sub-prompt with $|s| \leq C$, and $r$ is the string response. This mirrors the base LLM interface, allowing recursive decomposition up to a configurable depth.

> [!NOTE]
> The paper uses a **maximum recursion depth of 1** in experiments; deeper hierarchies are left as future work.

### Algorithm: RLM Inference

```
INPUT:  long_prompt x (|x| >> C), task description t, context window C
OUTPUT: answer a

1. Initialize REPL environment:
     env.prompt = x
     env.llm_query = lambda s: base_llm(s)    # base call with |s| <= C

2. Construct system prompt:
     sys = "You have access to `prompt` (a long string) and `llm_query(s)`.
            Write Python code to answer the task: {t}"

3. Begin LLM inference loop:
     while not done:
         code = llm_generate(sys + repl_history)  # LLM writes Python code
         stdout = execute(code, env)               # REPL executes code
         repl_history += (code, stdout)            # Append (code, output) to context
         if llm_signals_done(stdout):
             break

4. Extract final answer a from repl_history

RETURN a
```

> [!IMPORTANT]
> The LLM's context window at each step contains: system prompt + REPL history (interleaved code and outputs). The long input `x` is **not** in the LLM context—only the parts the LLM explicitly reads via code execution appear in stdout.

## Emergent Decomposition Behaviors

Without any explicit training for RLM behavior, models spontaneously exhibit four strategies:

| Behavior | Description | Example |
|---|---|---|
| **Regex Filtering** | Uses `re.findall` or string search on `prompt` to locate relevant sections before calling `llm_query` | `hits = re.findall(r'target_keyword.*', prompt)` |
| **Recursive Chunking** | Splits `prompt` into equal-length or keyword-delimited chunks; calls `llm_query` on each | `chunks = [prompt[i:i+C] for i in range(0, len(prompt), C)]` |
| **Answer Verification** | Redundantly calls `llm_query` a second time to verify the extracted answer | Observed in ~20% of GPT-5 trajectories |
| **Long Output Construction** | Assembles multi-part outputs as REPL string variables, concatenating results | `output += llm_query(chunk)` in a loop |

> [!NOTE]
> GPT-5 tends to be **conservative** (few sub-LLM calls, prefers regex filtering), while Qwen3-Coder-480B-A35B is **liberal** (more sub-LLM calls, more chunking). Both outperform their respective base models.

## Comparison with Baselines

| Method | Mechanism | Key Difference from RLM |
|---|---|---|
| **Base LLM** | Truncates input to fit context window | No access to full prompt; misses distant information |
| **Summary Agent** | Iteratively summarizes chunks, then answers | Lossy compression; RLM preserves raw text via REPL |
| **CodeAct + BM25** | Agent uses BM25 retrieval + code execution | Retrieval is keyword-based, not LLM-guided; RLM uses model priors to filter |
| **RLM (no sub-calls)** | REPL only, no `llm_query` recursion | Can handle inputs beyond context, but no recursive decomposition |

> [!TIP]
> CodeAct ([Wang et al., 2024](https://arxiv.org/abs/2402.01030)) is an agent framework where LLMs interleave code execution with reasoning. RLMs extend this idea with **recursive self-calls** rather than tool calls.

# Experiments

- **Datasets:**
  - **S-NIAH (Synthetic Needle-in-a-Haystack):** 50 tasks; input lengths $2^{13}$–$2^{18}$ tokens ($\approx 8\text{K}$–$262\text{K}$ tokens); constant-complexity task (locate a single "needle" fact)
  - **BrowseComp-Plus:** 150 tasks; 6M–11M tokens per task; multi-hop reasoning across large document collections
  - **OOLONG (trec\_coarse):** 50 tasks; 131K tokens; linear-complexity aggregation (classify all documents in corpus)
  - **OOLONG-Pairs:** 20 tasks; 32K tokens; quadratic-complexity pairwise comparison across all document pairs
  - **LongBench-v2 CodeQA:** multi-choice; 23K–4.2M tokens; code understanding over large codebases

- **Models:** GPT-5 (with GPT-5-mini for sub-calls), Qwen3-Coder-480B-A35B (with Qwen3-Coder-480B-A35B for sub-calls)

- **Hardware:** Not specified (cloud API inference)

- **Key Quantitative Results:**

| Task | GPT-5 Base | GPT-5 RLM | Qwen3 Base | Qwen3 RLM |
|---|---|---|---|---|
| CodeQA | 24.00% | **62.00%** | 20.00% | **56.00%** |
| BrowseComp-Plus | 0.00% | **91.33%** | 0.00% | **44.66%** |
| OOLONG | 44.00% | **56.50%** | 36.00% | **48.00%** |
| OOLONG-Pairs | 0.04% | **58.00%** | 0.06% | **23.11%** |

- **Cost:** Median token cost for RLMs is comparable to base models; 95th-percentile cost is higher due to high variance in trajectory lengths (some tasks require many recursive sub-calls)

> [!NOTE]
> The dramatic gain on BrowseComp-Plus (0% → 91%) and OOLONG-Pairs (0.04% → 58%) demonstrates RLM's advantage on tasks where the base model's context window is wholly insufficient.

## Ablation: Effect of Recursive Sub-Calls

| Setting | OOLONG | OOLONG-Pairs | BrowseComp+ |
|---|---|---|---|
| REPL only (no sub-calls) | Lower | Much lower | Much lower |
| Full RLM (REPL + sub-calls) | Higher | Substantially higher | Substantially higher |

The REPL alone is necessary to extend beyond the context window. Recursive sub-calls provide additional gains, especially on information-dense tasks.
