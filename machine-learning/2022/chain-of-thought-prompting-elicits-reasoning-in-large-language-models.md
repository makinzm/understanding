# Meta Information

- URL: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

---

# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

## Overview

Chain-of-thought (CoT) prompting is a few-shot prompting technique for large language models (LLMs) that augments each exemplar with a series of intermediate natural language reasoning steps leading to the final answer. Instead of a standard `<input, output>` pair, each demonstration follows the structure `<input, chain of thought, output>`. This method, proposed by Wei et al. (2022) from Google Research, enables LLMs to solve complex reasoning tasks—such as multi-step arithmetic, commonsense inference, and symbolic manipulation—that are difficult with standard prompting alone.

**Applicable when:** A practitioner has access to a sufficiently large language model (≥100B parameters) and wants to improve its reasoning over complex, multi-step tasks without any fine-tuning.

---

## 1. Introduction and Motivation

Standard few-shot prompting (Brown et al., 2020) provides `<question, answer>` pairs as context and relies on the model to implicitly infer the reasoning. For tasks requiring multiple sequential inference steps—such as solving math word problems or multi-hop questions—this approach fails because the model has no mechanism to decompose or externalize intermediate computation.

Chain-of-thought prompting addresses this by explicitly including the reasoning process in the prompt. The core insight is that "a chain of thought is a series of intermediate natural language reasoning steps that lead to the final output" (Wei et al., 2022). The reasoning chain appears before the final answer so the model conditions on it during generation.

> [!NOTE]
> "We explore the ability of language models to perform few-shot prompting for reasoning tasks, given a prompt that consists of triples: ⟨input, chain of thought, output⟩."

---

## 2. Method: Chain-of-Thought Prompting

### 2.1 Prompt Construction

**Input:** A few-shot prompt $P$ consisting of $k$ exemplars, a new question $q$.

**Output:** A chain of thought followed by the final answer $a$.

Each exemplar in the prompt takes the form:

$$P = [(q_1, c_1, a_1), (q_2, c_2, a_2), \ldots, (q_k, c_k, a_k), q_{\text{new}}]$$

where $q_i$ is the input question, $c_i$ is the chain of thought (intermediate reasoning steps in natural language), and $a_i$ is the final answer.

At inference time, the model generates $c_{\text{new}}$ (reasoning steps) followed by $a_{\text{new}}$ (final answer) for the new question $q_{\text{new}}$.

### 2.2 Example (Arithmetic Reasoning)

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

Chain of Thought:
Roger started with 5 balls. 2 cans × 3 balls = 6 balls.
5 + 6 = 11.

A: The answer is 11.
```

Contrast with standard prompting which would only provide `Q: ... A: 11` without the intermediate steps.

### 2.3 Implementation Details

- **Number of exemplars:** 8 per task (4 for multiple-choice tasks like AQuA); for BIG-bench tasks without training sets, the first 10 examples are used as exemplars.
- **Decoding:** Greedy decoding for all primary results.
- **No fine-tuning:** "No language models were finetuned in the process of writing this paper"—CoT is purely a prompting technique.
- **Reasoning chain authorship:** Reasoning chains were written manually by the authors; robustness experiments show other annotators' chains produce similar improvements.

---

## 3. Evaluated Reasoning Categories

### 3.1 Arithmetic Reasoning

Multi-step math word problems that require numerical computation and semantic understanding.

| Dataset | Description |
|---------|-------------|
| GSM8K | Grade school math problems requiring 2–8 reasoning steps |
| SVAMP | Arithmetic problems designed to test robustness to structural variation |
| ASDiv | Diverse arithmetic word problems across multiple types |
| AQuA | Algebraic word problems in multiple-choice format |
| MAWPS | Math word problems with subsets: SingleOp, SingleEq, AddSub, MultiArith |

### 3.2 Commonsense Reasoning

Questions requiring background world knowledge and multi-hop inference.

| Dataset | Description |
|---------|-------------|
| CommonsenseQA (CSQA) | Complex semantic questions requiring prior world knowledge |
| StrategyQA | Multi-hop binary questions requiring implicit reasoning strategy |
| Date Understanding | BIG-bench task: infer absolute dates from contextual clues |
| Sports Understanding | BIG-bench task: plausibility judgments about sports statements |
| SayCan | Map natural language instructions to robot action sequences (120 examples) |

### 3.3 Symbolic Reasoning

Tasks testing compositional generalization and length extrapolation.

| Dataset | Description |
|---------|-------------|
| Last Letter Concatenation | Concatenate last letters of words in a name (e.g., "John Smith" → "nh") |
| Coin Flip | Track coin state (heads/tails) after a sequence of flip/no-flip operations |

For symbolic reasoning, **out-of-domain (OOD)** evaluation tests generalization: models are prompted with 2-word / 2-flip examples and tested on 3–4 word / 3–4 flip examples.

---

## 4. Models Evaluated

| Model | Parameter Counts |
|-------|-----------------|
| GPT-3 (OpenAI) | 350M, 1.3B, 6.7B, 175B |
| LaMDA (Google) | 422M, 2B, 8B, 68B, 137B |
| PaLM (Google) | 8B, 62B, 540B |
| UL2 (Google) | 20B |
| Codex (OpenAI) | code-davinci-002 |

LaMDA experiments average over 5 random orderings of exemplars; other models use a single ordering.

---

## 5. Key Results

### 5.1 Arithmetic Reasoning (GSM8K)

GSM8K is the most challenging benchmark evaluated. Chain-of-thought prompting provides dramatic gains for large models:

| Model | Standard Prompting | CoT Prompting | Δ |
|-------|--------------------|---------------|---|
| UL2 20B | 4.1% | 4.4% | +0.3% |
| LaMDA 137B | 6.5% | 14.3% | +7.8% |
| GPT-3 175B | 15.6% | 46.9% | +31.3% |
| Codex | 19.7% | 63.1% | +43.4% |
| PaLM 540B | 17.9% | 56.9% | +39.0% |

Prior SOTA (Cobbe et al., 2021, fine-tuned GPT-3): **55%**. PaLM 540B + CoT achieves **56.9%** without any fine-tuning.

> [!IMPORTANT]
> Chain-of-thought prompting **hurts** performance for small models (≤8B parameters). Smaller models generate "fluent but illogical chains of thought," leading to worse accuracy than standard prompting. The benefit only emerges reliably at ≥100B parameters.

### 5.2 Commonsense Reasoning

PaLM 540B with CoT achieves new SOTA on StrategyQA (77.8% vs. prior SOTA 69.4%) and Sports Understanding (95.4%, exceeding unaided sports enthusiasts at 84%).

### 5.3 Symbolic Reasoning (OOD Generalization)

| Task | Model | Standard | CoT |
|------|-------|----------|-----|
| Last Letter (OOD, 4 words) | PaLM 540B | 0.0% | 63.0% |
| Coin Flip (OOD, 4 flips) | PaLM 540B | ~20% | 90.2% |

Small models (≤8B) achieve near 0% OOD, even with CoT. The reasoning structure provided by CoT enables large models to generalize compositionally to longer sequences.

---

## 6. Ablation Studies

Conducted primarily on LaMDA 137B and PaLM 540B on GSM8K to isolate what makes CoT effective.

### 6.1 Ablation Variants

| Variant | Description | GSM8K (LaMDA 137B) |
|---------|-------------|---------------------|
| Standard prompting | `<q, a>` pairs, no intermediate steps | 6.5% |
| **CoT prompting** | `<q, chain of thought, a>` | **14.3%** |
| Equation only | Output arithmetic equations, no natural language | 5.4% |
| Variable compute (dots) | Placeholder dots matching equation length | 6.4% |
| Reasoning after answer | Chain of thought provided post-answer | 6.1% |

### 6.2 Interpretation

- **Equation only** fails on complex problems requiring semantic step-by-step decomposition: equations cannot capture "she gave 2 to her mom" logic without natural language.
- **Variable compute** fails: more tokens alone do not help; the sequential logical structure is needed.
- **Reasoning after answer** performs at baseline: the chain of thought must causally precede the answer to be useful. Post-hoc explanations cannot guide inference.

> [!NOTE]
> "The sequential nature of the chain of thought is important—none of the ablation variants approach the performance of the full chain of thought method."

---

## 7. Robustness Analysis

CoT prompting is robust to several sources of variation:

| Factor | Finding |
|--------|---------|
| Different annotators | All three annotators' chains improve over baseline; variance exists but gains persist |
| Exemplar ordering | LaMDA 137B averaged over 5 orderings; ordering changes results but CoT consistently outperforms standard |
| Number of exemplars | Performance improves with more exemplars; gains visible even with 1 exemplar |
| Different source datasets | Using different exemplar pools produces consistent improvements |

---

## 8. Comparison with Related Methods

| Method | Key Difference |
|--------|---------------|
| Standard few-shot (Brown et al., 2020) | Uses `<q, a>` pairs only; no intermediate reasoning visible to the model |
| Scratchpad prompting (Nye et al., 2021) | Also uses intermediate steps but requires fine-tuning a model to produce scratchpads |
| Neuro-symbolic (Roy & Roth, 2015; Chiang & Chen, 2019) | Uses formal languages (equations, programs) for reasoning; less flexible than natural language chains |
| Fine-tuning with rationales (Ling et al., 2017; Cobbe et al., 2021) | Requires labeled rationale datasets for training; CoT requires no fine-tuning |
| Zero-shot CoT (Kojima et al., 2022) | Adds "Let's think step by step" to elicit reasoning; no exemplars needed, but few-shot CoT outperforms it |

> [!TIP]
> Zero-shot chain-of-thought prompting (Kojima et al., 2022) — simply appending "Let's think step by step" — was developed concurrently as a complementary approach.

---

## 9. Algorithm Summary (Inference Procedure)

```
Algorithm: Chain-of-Thought Prompted Inference
Input:
  - Pretrained LLM M (≥100B parameters recommended)
  - Few-shot exemplar set E = {(q_i, c_i, a_i) for i=1..k}
  - New question q_new
Output:
  - Answer a_new

Procedure:
1. Construct prompt P by concatenating exemplars and new question:
     P = format(q_1, c_1, a_1) + ... + format(q_k, c_k, a_k) + format(q_new)
2. Generate response with LLM:
     r = M.generate(P)  # greedy decoding
3. Parse r to extract chain of thought c_new and answer a_new:
     c_new, a_new = parse(r)
4. Return a_new
```

For arithmetic tasks, an optional post-processing step applies Python `eval()` on extracted numeric expressions to verify calculations.

---

## 10. Limitations

1. **Scale dependency:** CoT only benefits very large models (≥100B parameters). At smaller scales, fluent but logically incorrect chains emerge, hurting accuracy. This limits practical deployment as large models are expensive.

2. **No correctness guarantee:** "There is no guarantee of correct reasoning paths." The model may produce a chain of thought that is plausible-sounding but contains semantic or logical errors.

3. **Manual effort:** Creating high-quality chains of thought requires human effort; while robustness experiments show other annotators achieve similar gains, domain-specific tasks may require expertise.

4. **Error propagation:** Early errors in the reasoning chain tend to cascade. In LaMDA 137B error analysis, 46% of wrong answers had chains with minor errors that cascaded into incorrect final answers.

---

# Experiments

- **Datasets:** GSM8K, SVAMP, ASDiv, AQuA, MAWPS (arithmetic); CommonsenseQA, StrategyQA, Date Understanding, Sports Understanding, SayCan (commonsense); Last Letter Concatenation, Coin Flip (symbolic)
- **Hardware:** Not specified (inference only; no fine-tuning)
- **Optimizer:** Not applicable (prompting only)
- **Key results:**
  - PaLM 540B + CoT achieves 56.9% on GSM8K, surpassing fine-tuned GPT-3 (55%) with no training
  - GPT-3 175B improves from 15.6% → 46.9% on GSM8K with CoT
  - PaLM 540B + CoT achieves 63.0% OOD on Last Letter Concatenation (vs. 0.0% standard)
  - CoT hurts small models (≤8B parameters) across all arithmetic tasks
  - StrategyQA new SOTA: 77.8% (PaLM 540B + CoT) vs. prior SOTA 69.4%
