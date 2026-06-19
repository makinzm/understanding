# Meta Information

- URL: [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., & Zeng, M. (2023). Automatic Prompt Optimization with "Gradient Descent" and Beam Search. arXiv:2305.03495.

---

# 1. Introduction

LLM performance on downstream tasks depends heavily on the phrasing of the prompt, but manual prompt engineering is expensive, inconsistent, and requires domain expertise.

ProTeGi (Prompt Optimization with Textual Gradients) proposes an automatic method for iteratively improving prompts using only API access to a black-box LLM. It mirrors gradient descent by: (1) computing a **textual gradient**—natural language feedback describing errors in the current prompt—then (2) editing the prompt in the opposite semantic direction. A beam search with bandit-based selection tracks the most promising candidates across iterations.

**Who benefits from this**: Practitioners who need high-quality task-specific prompts but cannot fine-tune model weights (API-only access), and researchers studying discrete optimization of LLM instructions.

# 2. Background and Related Work

| Method | Access Required | Mechanism | Limitation |
|---|---|---|---|
| Soft prompt tuning | Internal model weights | Gradient-based embedding updates | Inapplicable to black-box APIs |
| RL-based (RLPrompt) | Reward model | Token-level RL actions | Produces incoherent prompt text |
| Monte Carlo search | None | Random mutations | No semantic direction |
| Evolutionary search | None | Genetic operations on tokens | Computationally expensive, directionless |
| **ProTeGi** | **API only** | **Textual gradient + beam search** | **Rate-limited by LLM API** |

ProTeGi is distinguished by producing human-readable intermediate gradients and candidate prompts, making it interpretable and easy to audit.

# 3. Method: ProTeGi

## 3.1 Overview

ProTeGi treats prompt optimization as a search problem over the discrete space of natural-language strings. Rather than using numerical gradients, it uses an LLM to generate textual critiques, then uses another LLM call to rewrite the prompt accordingly.

**Input**: Initial prompt $p_0$, training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, budget $r$ (iterations), beam width $b$.

**Output**: An optimized prompt $p^*$ maximizing a task metric on $\mathcal{D}$.

## 3.2 Textual Gradient Generation

Given current prompt $p$ and a minibatch $\mathcal{B} \subset \mathcal{D}$ of size 64, ProTeGi runs the LLM on each $(x_i, y_i) \in \mathcal{B}$ using $p$ and collects errors. It then calls the LLM with:

```
System: You are an expert prompt engineer.
User: Here is a prompt: {p}
      Here are some errors the prompt made: {error_cases}
      Describe, in a few sentences, what is wrong with this prompt and how to fix it.
```

This produces a **textual gradient** $\nabla p$—a natural language description of the prompt's failures. Multiple gradients can be sampled from this call (e.g., 4 samples with temperature 1.0) to generate diverse feedback.

## 3.3 Prompt Editing

Given textual gradient $\nabla p$, ProTeGi rewrites the prompt using:

```
System: You are an expert prompt engineer.
User: Here is a prompt: {p}
      Here is feedback on what is wrong with this prompt: {∇p}
      Rewrite the prompt to fix these issues. Output only the new prompt.
```

This produces an edited candidate $p' = p + \delta$, analogous to a gradient descent step $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$. Multiple edits are produced per gradient, yielding a candidate pool.

Additionally, **paraphrasing** is used to further expand candidates: the same semantic content is reworded in multiple ways to improve coverage of the prompt space.

## 3.4 Beam Search with Bandit Selection

ProTeGi uses beam search (width $b = 4$) to maintain the top-$b$ prompt candidates across $r = 6$ iterations. At each step, candidates are expanded via textual gradients and editing, then the best $b$ candidates are selected using a bandit algorithm.

**Why bandits?** Evaluating a prompt requires running it on data (costly). The best arm identification problem—finding the best candidate with minimal evaluations—is well-suited to multi-armed bandit algorithms.

| Bandit Algorithm | Mechanism | Notes |
|---|---|---|
| UCB (Upper Confidence Bound) | Score = mean + $c\sqrt{\frac{\ln t}{n_i}}$ | Exploration-exploitation tradeoff |
| UCB-E | Higher exploration constant | Emphasizes uncertainty |
| Successive Rejects | Eliminates worst arm each round | Provably sample-optimal, no hyperparameters |
| Successive Halving | Eliminates bottom half each round | More aggressive pruning |

Each candidate prompt $p_i$ is evaluated on a subset of examples; its running mean score $\hat{\mu}_i$ and selection count $n_i$ inform the bandit's selection at each round.

## 3.5 Full Algorithm (Pseudocode)

```
Input: p_0 (initial prompt), D (dataset), r (iterations), b (beam width)
Output: p* (optimized prompt)

beam = [p_0]

for step in 1..r:
    candidates = []
    for p in beam:
        B = sample_minibatch(D, size=64)
        errors = evaluate(p, B)
        for gradient in sample_gradients(p, errors, n=4):
            for edit in sample_edits(p, gradient, n=4):
                candidates.append(edit)
            for paraphrase in paraphrase(p, n=2):
                candidates.append(paraphrase)
    beam = bandit_select(candidates ∪ beam, top_k=b, D)

p* = argmax_{p in beam} score(p, D)
return p*
```

`bandit_select` assigns each candidate an arm in a multi-armed bandit, runs rounds of evaluation, and returns the top $b$ arms.

# 4. Experiments

## 4.1 Datasets

| Dataset | Task | Size | Language |
|---|---|---|---|
| Jailbreak | Detect AI safeguard violations | 452 examples | Multilingual |
| Ethos | Hate speech detection | 997 examples | English |
| Liar | Fake news classification | ~4,000 examples | English |
| Sarcasm | Sarcasm detection | 10,000 examples | Arabic |

All tasks are binary classification, evaluated using F1 score (or accuracy where stated).

## 4.2 Configuration

- Model: GPT-3.5-turbo (January 2023 checkpoint)
- Minibatch size for gradient generation: 64
- Beam width: $b = 4$
- Optimization steps: $r = 6$
- Temperature: 0.0 for classification inference, 1.0 for generation steps
- Initial prompts: human-written baselines per task

## 4.3 Baselines

- **Human prompt**: The manually written initial prompt (lower bound reference)
- **Monte Carlo**: Random mutations to the prompt without semantic direction
- **RL prompt (RLPrompt)**: Reinforcement-learning approach operating at the token level
- **AutoGPT**: Agentic LLM loop with iterative self-improvement

## 4.4 Key Results

ProTeGi achieves:
- Up to **+31% absolute improvement** over the initial human prompt
- **+3.9%** over Monte Carlo random search
- **+8.2%** over RL-based prompt optimization (RLPrompt)
- **+15.2%** over AutoGPT self-improvement baseline

> [!NOTE]
> The largest improvements occur on the Jailbreak and Sarcasm tasks, where the initial prompt was weakly specified and there was more room for targeted improvement.

## 4.5 Ablation: Search Strategy

Beam search (width 4) consistently outperforms:
- **Greedy search** (beam width 1): gets trapped in local optima
- **Flat enumeration**: explores broadly but lacks memory of top candidates

Beam search balances exploration of diverse candidates with exploitation of high-performing ones.

## 4.6 Ablation: Bandit Algorithm

UCB-based bandits outperform Successive Rejects in practice, despite Successive Rejects being theoretically sample-optimal. This may be because:
- Training sets are small and noisy, making exact best-arm identification harder to realize empirically
- UCB's hyperparameter $c$ can be tuned to the problem scale

## 4.7 Learning Curve

Performance typically peaks around **3 optimization steps**, after which overfitting to the training minibatches may cause slight degradation. This suggests early stopping is beneficial.

# 5. Comparison to Similar Methods

| Feature | ProTeGi | DSPy | RLPrompt | Soft Prompt Tuning |
|---|---|---|---|---|
| Model access needed | API only | API or weights | API + reward model | Internal weights |
| Output readable | Yes | Yes | No (token-level) | No (embedding) |
| Handles multi-step pipelines | No | Yes | No | Partial |
| Optimization direction | Semantic (textual) | Bootstrapping + compilation | Numerical RL reward | Backprop gradient |
| Scales to large datasets | Partially (minibatches) | Yes | Yes | Yes |

> [!IMPORTANT]
> ProTeGi optimizes a **single prompt string** for a single LM call. It does not optimize multi-step pipelines. DSPy addresses this more general problem but requires more infrastructure and module decomposition.

# 6. Limitations

- **Speed**: A full 6-iteration optimization can exceed 1 hour due to LLM API rate limits.
- **Scope**: Only evaluated on 4 binary classification tasks; generalization to generation or complex reasoning tasks is unclear.
- **Gradient quality**: Textual gradients are sometimes tangential or generic ("be more specific") and do not always point toward the true error source.
- **Variance**: Some configurations exhibit high run-to-run variance due to LLM stochasticity.

# 7. Conclusion

ProTeGi demonstrates that the conceptual structure of gradient descent—compute a gradient, step in the direction that reduces loss—can be lifted to the discrete, natural-language domain using an LLM as the gradient estimator and editor. Combined with beam search and bandit-based candidate selection, this yields a practical and interpretable algorithm for automatic prompt optimization that outperforms prior black-box methods while requiring only API access.
