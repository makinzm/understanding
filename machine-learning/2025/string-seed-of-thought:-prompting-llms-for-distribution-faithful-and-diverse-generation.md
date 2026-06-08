# Meta Information

- URL: [String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](https://arxiv.org/abs/2510.21150)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Misaki, K., Akiba, T. (2025). String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation. Sakana AI. Accepted at ICLR 2026.

# String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation

## Problem and Motivation

Large language models (LLMs) systematically fail to reproduce target probability distributions even when they can describe those distributions accurately. Given the instruction "choose A with probability 0.3 and B with probability 0.7", models instead collapse toward a dominant choice or revert to pseudo-uniform sampling. This failure mode undermines applications that depend on stochastic outputs: human behavior simulation, content diversification, mixed-strategy game playing, and test-time scaling via diverse candidate generation.

The authors formalize two distinct tasks:

1. **Probabilistic Instruction Following (PIF)**: Given a predefined set of options $\mathcal{A} = \{a_1, \ldots, a_k\}$ with target distribution $P = (p_1, \ldots, p_k)$, generate responses such that the empirical distribution $\hat{P}$ over $n$ samples converges to $P$.
2. **Diversity-Aware Generation (DAG)**: For open-ended tasks without predefined answer sets, generate responses that are both diverse (covering distinct concepts) and high-quality.

SSoT (String Seed of Thought) solves both with a unified prompting strategy that requires no model fine-tuning, no external tools, and no temperature access — making it directly applicable to reasoning models and closed APIs alike.

## Core Methodology: String Seed of Thought

### Overview

SSoT instructs the model to:
1. **Generate a random string** (the "seed") to accumulate sufficient entropy
2. **Manipulate the string deterministically** (via arithmetic on character codes) to derive a sample index
3. **Map the index to an answer** according to the target distribution

This separates entropy collection from distribution sampling: the random string captures unpredictability, and the hash-based extraction converts that unpredictability into a calibrated selection.

### PIF Prompt Structure

```
Input: question + option list + target distribution P = (p_1, ..., p_k)
Instruction:
  1. Output a random string of length n.
  2. Compute a hash value h from the string characters.
  3. Map h to an answer index i using the CDF of P (i.e., select a_i where CDF(i-1) < (h mod N)/N ≤ CDF(i)).
  4. Output the selected answer a_i.
Output: selected answer a_i ∈ {a_1, ..., a_k}
```

### DAG Prompt Structure

```
Input: question or creative task
Instruction:
  1. Output a random string of length n.
  2. Manipulate the string to select a unique conceptual starting point or template.
  3. Generate a complete, high-quality response from that starting point.
Output: a single diverse response
```

The string is used as a seed for divergence: rather than mapping to a fixed-set index, the model uses the hash to select a topic cluster, a stylistic template, or a narrative seed before generating.

### String Manipulation Strategies

Models autonomously develop two primary arithmetic strategies:

| Strategy | Formula | Distribution Fit |
|---|---|---|
| **Sum-Mod** | $h = \left(\sum_{i=1}^{n} \text{ord}(c_i)\right) \bmod M$ | Uniform |
| **Rolling Hash** | $h = \left(\sum_{i=1}^{n} \text{ord}(c_i) \cdot B^{n-i}\right) \bmod M$ | Biased / Non-uniform |

where $c_i$ is the $i$-th character of the generated string, $\text{ord}(\cdot)$ is the ASCII code, $B$ is a polynomial base (e.g., 31 or 37), and $M$ is the modulus. Sum-Mod suffices for uniform distributions; rolling hash provides finer-grained control needed to approximate biased distributions.

## Theoretical Analysis

### Theorem 4.1 — Hash Function Bound

For a 2-universal hash family $\mathcal{H} = \{h: \Sigma^n \to [M]\}$ applied to strings drawn from a source with min-entropy $k$, the expected total variation (TV) distance between the induced output distribution and the uniform distribution over $[M]$ is bounded by:

```math
\begin{align}
  \mathbb{E}_{h \sim \mathcal{H}}\left[\text{TV}(h(X), \mathcal{U}_{[M]})\right]
  \;\leq\; \frac{1}{2}\sqrt{M \cdot 2^{-k}} + \frac{1}{2}\sqrt{\frac{1}{n}}
\end{align}
```

- First term: decreases as min-entropy $k$ grows (longer strings → more entropy → $k$ increases)
- Second term: decreases as $n$ (number of trials) grows

As string length increases, $k$ grows, driving the first term toward zero. The bound confirms that longer SSoT strings produce distributions closer to the target.

### Theorem 4.2 — Sum-Mod Convergence

For i.i.d. characters drawn from distribution $Q$ over $\Sigma$, the TV distance between the sum-mod output distribution and the uniform distribution over $[M]$ satisfies:

```math
\begin{align}
  \text{TV}\!\left(\sum_{i=1}^{n} X_i \bmod M,\; \mathcal{U}_{[M]}\right)
  \;\leq\; \left(\max_{j \neq 0} |\hat{Q}(j)|\right)^{n}
\end{align}
```

where $\hat{Q}(j)$ is the discrete Fourier coefficient of $Q$ at frequency $j$. The bound shrinks geometrically in $n$, so the sum-mod output converges exponentially fast to uniform as string length grows.

Both theorems jointly justify the design: generating a longer random string directly tightens the distributional fidelity guarantee.

## Experiments

### PIF Experiments

- **Models**: DeepSeek-V3, GPT-4o, O4-mini-high, QwQ-32B, DeepSeek-R1
- **Tasks**: 2-choice (uniform and biased), 3-choice, 9-choice distributions
- **Protocol**: 100 trials per configuration, 10 independent repetitions; results averaged
- **Metrics**: Jensen-Shannon (JS) divergence, Kullback-Leibler (KL) divergence, Total Variation (TV) distance

**Baselines**:
- High temperature (max available)
- Few-shot examples ($k \in \{3, 10, 50\}$ i.i.d. samples from target $P$)
- Prompt ensemble (50 semantically equivalent paraphrases, majority vote)
- Sequential sampling (condition each call on prior outputs)
- External randomness injection (provide pre-generated random numbers in context)

**Results**: SSoT reduced JS divergence by 85–99% relative to the no-method baseline across all models. Reasoning models (DeepSeek-R1, QwQ-32B) approached near-PRNG fidelity. SSoT outperformed all baselines, including few-shot with $k=50$ and external randomness provision, particularly on biased distributions where simple temperature scaling degrades quickly.

### Rock-Paper-Scissors Adversarial Test

An LLM agent using SSoT played rock-paper-scissors against a pattern-exploiting bot that scanned prior moves for exploitable frequencies. SSoT-equipped models maintained near-zero cumulative score (balanced wins and losses), while baseline agents were consistently exploited. This validates real-world applicability to game-theoretic mixed strategies.

### DAG Experiments

- **Benchmark**: NoveltyBench (two splits: curated tasks and WildChat prompts)
- **Metrics**: Distinct score (semantic diversity across repeated generations) and Utility score (single-response quality judged by an LLM evaluator)

**Results**: SSoT achieved the highest Distinct scores across task categories while maintaining or improving Utility scores. The gains were most pronounced on "Creativity" tasks, where baseline models without SSoT tended to regenerate similar phrasings or topics on repeated calls.

### Reasoning Length Scaling (s1.1-32B)

Using the s1.1-32B model with controlled thinking token budgets (from short to extended CoT), the authors showed:
- JS and KL divergence decrease monotonically as thinking tokens increase
- Cohen's $w$ effect sizes indicate practically significant improvement even at moderate token counts
- Lempel-Ziv complexity and zlib compression ratios of the generated strings increase with reasoning length — confirming that longer reasoning genuinely produces higher-entropy strings, even at temperature 0

This suggests the randomness is not injected externally but is a product of the reasoning process itself.

## Strategy Analysis

### How Models Implement SSoT

When given SSoT instructions, frontier LLMs autonomously choose between Sum-Mod and Rolling Hash based on task complexity:
- For uniform distributions, models typically default to Sum-Mod (simpler, sufficient)
- For biased distributions, models switch to Rolling Hash (captures non-uniformity)

This emergent strategy selection indicates that the models understand the mathematical properties of the strategies rather than pattern-matching from training data.

### DAG Strategy Taxonomy

For diversity-aware generation, two dimensions characterize model strategies:
- **Assembly method**: direct list sampling (choose a topic then write) vs. template filling (instantiate a structural pattern)
- **Sampling scope**: global (single random draw governs entire response) vs. local (repeated random draws for each sub-component)

Creative tasks benefit most from template-based decomposition with local sampling, which diversifies multiple independent structural slots simultaneously.

## Comparison with Related Methods

| Method | Requires Fine-tuning | Requires Temp Access | Works on Closed APIs | Theoretical Guarantee | Handles Biased Distributions |
|---|---|---|---|---|---|
| Temperature scaling | No | Yes | Partial | No | No |
| Few-shot sampling | No | No | Yes | No | Weak |
| Calibration methods | Yes | No | No | Partial | No |
| **SSoT (this work)** | **No** | **No** | **Yes** | **Yes** | **Yes** |

Calibration methods (e.g., RLHF-based alignment) adjust model confidence to match empirical accuracy, but operate on a per-question basis rather than producing a target mixture across answers. SSoT instead enables true sampling from a designer-specified distribution over multiple valid outputs — a fundamentally different goal.

## Limitations

1. **Reasoning capability dependency**: Models smaller than ~8B parameters fail to develop correct hash-based strategies, producing outputs that ignore the string entirely. The approach requires sufficient in-context reasoning capacity.
2. **Lazy strategy risk**: If a model only uses the first character of the generated string (ignoring the rest), the effective entropy is low and output distributions become biased. Mitigated by explicitly directing models toward robust strategies (rolling hash) in the system prompt.
3. **Single-answer task inapplicability**: SSoT is designed for tasks with multiple valid answers. Applying it to math problems or factual retrieval — where one answer is correct — introduces distraction without benefit and may degrade accuracy.

# Experiments

- **Dataset (PIF)**: Synthetic distributions over $\{2, 3, 9\}$ choices; 100 trials × 10 repetitions per configuration
- **Dataset (DAG)**: NoveltyBench (curated subset + WildChat subset); includes creativity, brainstorming, and writing tasks
- **Hardware**: Not specified
- **Optimizer**: Not applicable (prompting-only method; no training)
- **Results**: 85–99% JS divergence reduction vs. baseline on PIF; highest Distinct scores on NoveltyBench DAG while maintaining Utility scores; near-PRNG fidelity on DeepSeek-R1 and QwQ-32B
