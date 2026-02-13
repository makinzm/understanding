# Meta Information

- URL: [Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/abs/2512.14982)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Leviathan, Y., Kalman, M., & Matias, Y. (2025). Prompt Repetition Improves Non-Reasoning LLMs. arXiv preprint arXiv:2512.14982.

---

# Prompt Repetition Improves Non-Reasoning LLMs

## Overview

This paper proposes a zero-cost inference-time technique: **duplicating the input prompt** before generation to improve LLM output quality. Instead of passing `<QUERY>` to the model, the query is transformed to `<QUERY><QUERY>`. This exploits the bidirectional attention structure that emerges when a token in the second copy can attend to all tokens in the first copy (and vice versa), partially overcoming the causal masking limitation of autoregressive language models.

**Applicability:** Any practitioner using black-box LLM APIs (OpenAI, Anthropic, Google, etc.) can apply this technique without model access, fine-tuning, or extra output generation. It is especially effective for non-reasoning (standard) LLMs; reasoning models already learn to repeat prompts internally during chain-of-thought.

---

## Background: Causal Masking and Token Attention

Standard autoregressive LLMs use causal (unidirectional) attention: token at position $i$ can only attend to tokens $j \leq i$. Given a prompt of length $L$ followed by generated tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = -\infty$ if $j > i$ (causal mask), $Q, K, V \in \mathbb{R}^{L \times d_k}$.

**Problem:** For a prompt token at position $i$, it can only attend to tokens $1, \ldots, i$. The *first* token in the prompt, therefore, can only attend to itself — it has no broader context at prefill time. This is asymmetric: the *last* prompt token attends to all preceding tokens, gaining the richest representation.

> [!NOTE]
> "When given a query `q`, causal LLMs process it as a sequence of tokens and use the representations of these tokens to generate a response. Due to the causal nature of these models, each token representation is conditioned only on the tokens that came before it in the sequence."

---

## Method: Prompt Repetition

### Input Transformation

Given an original query $q = (t_1, t_2, \ldots, t_L)$ with $L$ tokens, the repeated prompt is:

$$q_{\text{rep}} = (t_1, \ldots, t_L, t_1, \ldots, t_L)$$

Total input length becomes $2L$. The model generates its response conditioned on $q_{\text{rep}}$.

**Effect on attention:** In the second copy, token $t_i$ (at position $L + i$) can now attend to all $2L$ tokens of both copies. In particular, $t_1$ in the second copy attends to $t_1, \ldots, t_L$ (the full first copy), giving it access to the entire prompt context before it begins attending to its own copy.

> [!IMPORTANT]
> The repetition occurs only in the **prefill stage** (processing the input prompt), which is fully parallelizable on modern hardware. No additional **decode** tokens are generated, so output length and latency are unaffected for typical prompt lengths.

### Algorithm

```
Input: query q (list of tokens of length L)
Output: LLM response r

1. q_rep ← concatenate(q, q)          # duplicate prompt, length 2L
2. r ← LLM.generate(q_rep)            # standard generation from repeated prompt
3. return r
```

### Variants

| Variant | Description |
|---|---|
| Prompt Repetition (×2) | Simple duplication: `<QUERY><QUERY>` |
| Prompt Repetition (Verbose) | Includes transition: `<QUERY> Let me repeat that: <QUERY>` |
| Prompt Repetition (×3) | Triple repetition: `<QUERY><QUERY><QUERY>` |
| Padding Control | Pad input to length $2L$ with semantically empty tokens (no repetition) |

The **Padding Control** variant is a critical ablation: it matches the input length of Prompt Repetition but uses filler tokens instead of semantic content. It shows no improvement, confirming that the gain comes from semantic repetition rather than mere length increase.

---

## Theoretical Motivation

The motivation draws an analogy to **encoder-decoder** or **bidirectional** architectures (e.g., BERT, T5 encoder), where every token attends to every other token. By repeating the prompt, the second copy functionally acts as a "bidirectionally-attended" segment: the second occurrence of $t_i$ can attend to all of $t_1, \ldots, t_L$ (first copy) and $t_1, \ldots, t_{i-1}$ (second copy so far), approaching full bidirectional context.

> [!NOTE]
> "Reasoning models trained with RL often learn to repeat (parts of) the user's request. Prompt repetition provides a way to externalize this behavior, performing it externally in the prefill stage."

This observation connects to why prompt repetition shows diminishing returns on reasoning models: they already learn to re-state the problem in their chain-of-thought.

---

## Experiments

### Datasets

| Benchmark | Type | Task |
|---|---|---|
| ARC (Challenge) | Multiple-choice QA | Science questions |
| OpenBookQA | Multiple-choice QA | Elementary science |
| GSM8K | Math word problems | Grade-school arithmetic |
| MMLU-Pro | Multiple-choice QA | Professional/academic knowledge |
| MATH | Competition math | Advanced mathematics |
| NameIndex (custom) | Retrieval from context | Locate name at given index in a long list |
| MiddleMatch (custom) | Retrieval from context | Match query to item in middle of a list |

- **NameIndex** and **MiddleMatch** are synthetic tasks specifically designed to stress-test "lost in the middle" failures, where LLMs underperform on information located in the middle of long contexts.

### Models Evaluated

- Gemini 2.0 Flash, Gemini 2.0 Flash-Lite (Google)
- GPT-4o, GPT-4o-mini (OpenAI)
- Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Haiku (Anthropic)
- DeepSeek V3

### Statistical Testing

Per-sample McNemar's test is used to compare model outputs with and without prompt repetition. This is a paired nonparametric test suitable for binary outcomes (correct/incorrect per sample).

### Results Summary

- **Non-reasoning models:** Prompt repetition wins 47 out of 70 benchmark-model combinations, with **0 losses** (remaining are ties at p < 0.05).
- **Reasoning models:** Neutral to slightly positive (5 wins, 1 loss, 22 ties).
- **Padding control:** No improvement over baseline, ruling out length as the causal factor.
- **Verbose variant:** Comparable or slightly better than simple repetition on some benchmarks.
- **×3 repetition:** Marginal additional gain over ×2 in some settings; not consistently better.

**Notable example:** Gemini 2.0 Flash-Lite on NameIndex improved from **21.33% → 97.33%** accuracy with prompt repetition, demonstrating that the "lost in the middle" failure is largely resolved when the prompt is repeated.

---

## Comparison with Related Methods

| Method | Mechanism | Requires model access? | Additional output tokens? | Extra latency? |
|---|---|---|---|---|
| **Prompt Repetition** (this work) | Duplicate input at API level | No | No | No (prefill only) |
| Chain-of-Thought (CoT) | Elicit reasoning steps in output | No | Yes | Yes (decode) |
| Reasoning models (o1, etc.) | RL-trained internal repetition | No (API) | Yes (internal) | Yes |
| Bidirectional LM (BERT) | Full attention at training time | Training change | N/A | N/A |
| Self-Consistency | Sample multiple outputs, majority vote | No | Yes (×N) | Yes (×N decode) |
| Prompt Engineering | Manually craft better prompts | No | Varies | Minimal |

> [!NOTE]
> Unlike CoT or self-consistency, prompt repetition adds no output tokens and incurs no additional decode latency, making it uniquely suited for **latency-sensitive or cost-sensitive** deployments.

---

## Limitations and Future Work

- **Long prompts:** For very long prompts ($L$ large), doubling input may increase **prefill time** noticeably; selective repetition of key sub-segments is proposed as future work.
- **Reasoning models:** The technique provides minimal additional benefit since RL training already induces internal repetition behavior.
- **Mechanism not fully understood:** The paper provides intuition via causal masking but does not analyze attention patterns empirically.
- **Future directions:** Fine-tuning on repeated-prompt data, multi-modal repetition, selective/partial repetition, and analysis of which attention heads benefit most.
