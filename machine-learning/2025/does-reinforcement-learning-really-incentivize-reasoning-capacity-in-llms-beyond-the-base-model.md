# Meta Information

- URL: [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., & Huang, G. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. arXiv:2504.13837.

# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

## Overview

This paper challenges the prevailing assumption that Reinforcement Learning with Verifiable Rewards (RLVR) genuinely expands the reasoning capabilities of large language models (LLMs). Using a pass@k analysis across large k values, the authors demonstrate that RLVR does not grant LLMs reasoning abilities beyond what their base models already possess — it merely redistributes the probability mass of the existing solution space toward higher-reward outputs.

> [!IMPORTANT]
> The central claim: RLVR improves **sampling efficiency** (pass@1), but **shrinks** the model's overall **reasoning boundary** (pass@k for large k). Base models, when sampled many times, consistently outperform their RL-fine-tuned counterparts in terms of the total set of solvable problems.

## Background: RLVR for LLM Reasoning

RLVR fine-tunes a policy $\pi_\theta$ by optimizing expected reward from a verifiable reward function $r$ (e.g., exact-match for math, unit tests for code):

```math
\begin{align}
  J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [r(y, x)] \right]
\end{align}
```

Popular algorithms instantiating this objective include:

| Algorithm | Key Mechanism |
|-----------|--------------|
| PPO | Clipped surrogate loss with value baseline |
| GRPO | Group relative policy optimization (no value model) |
| RLOO | Leave-one-out baseline for variance reduction |
| Reinforce++ | Reinforce with reward normalization improvements |
| DAPO | Decoupled clipping and dynamic sampling |

The PPO clipped objective is:

```math
\begin{align}
  \mathcal{L}_{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t\right)\right]
\end{align}
```

where $r_t(\theta) = \pi_\theta(y_t|x, y_{<t}) / \pi_{\theta_{\text{old}}}(y_t|x, y_{<t})$ is the probability ratio and $A_t$ is the advantage estimate.

## Core Methodology: pass@k Evaluation

### pass@k Metric

To evaluate the **upper bound** of a model's reasoning capacity, the authors use an unbiased estimator of pass@k: for a given problem $x_i$, sample $n$ responses ($n \geq k$) and count $c_i$ correct ones:

```math
\begin{align}
  \text{pass@}k := \mathbb{E}_{x_i \sim \mathcal{D}}\left[1 - \frac{\binom{n - c_i}{k}}{\binom{n}{k}}\right]
\end{align}
```

- **pass@1**: Measures greedy/single-sample accuracy — the standard metric used to claim RLVR improvements.
- **pass@k (large k)**: Measures whether **any** of many diverse samples solves the problem — a proxy for the model's reasoning boundary.

> [!NOTE]
> All chain-of-thought responses are **manually validated** to exclude solution hacking (accidentally correct answers arising from flawed reasoning chains), ensuring that pass@k values reflect genuine problem-solving ability.

### Sampling Efficiency Gap (SE Gap)

The SE gap quantifies how much additional exploration (more samples) can recover for a model:

```math
\begin{align}
  \Delta_{\text{SE}} = \text{pass@}k_{\max} - \text{pass@}1
\end{align}
```

A large SE gap indicates that the model can solve many more problems when given more chances, but low single-sample probability for each. RLVR-trained models show consistently high SE gaps (42–44 points), meaning their pass@1 gains come at the cost of diversity.

## Key Findings

### 1. RLVR Narrows the Reasoning Boundary

Across math (Minerva, AIME24, AMC23, MATH500), code (LiveCodeBench, HumanEval+, MBPP+), and vision (MathVista, MathVision) benchmarks, the pattern is consistent:

- **pass@1 improves** after RLVR training (the commonly reported metric).
- **pass@k (large k) degrades** relative to the base model, meaning the RL model can solve *fewer distinct* problems under exhaustive sampling.

For example, on LiveCodeBench with Qwen-2.5-7B:

- Base model: pass@1 ≈ 23.8%, pass@128 ≈ 50%
- RL-trained: pass@1 ≈ 28.1%, pass@128 ≈ 42.8%

The RL model gains ~4 points at k=1 but loses ~7 points at k=128, indicating a shrunken solution boundary.

### 2. RL Does Not Introduce Novel Reasoning Paths

To confirm that RL solutions are not qualitatively new, the authors measure perplexity of RL-generated responses under the base model:

```math
\begin{align}
  \text{PPL}_m(Y|x) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_m(y_t \mid x, y_1, \ldots, y_{t-1})\right)
\end{align}
```

RL-generated correct solutions have **low perplexity under the base model**, meaning the base model already assigns non-trivial probability to these reasoning chains. RLVR simply up-weights paths that already existed.

### 3. Result Holds Across Model Families and RL Algorithms

The narrowed boundary is not an artifact of a specific model or algorithm:

| Setting | SE Gap (RL) |
|---------|------------|
| PPO on Qwen-2.5-7B | ~42.6 pts |
| GRPO on Qwen-2.5-7B | ~43.1 pts |
| RLOO on Qwen-2.5-7B | ~43.9 pts |
| Reinforce++ on Qwen-2.5-7B | ~43.2 pts |

All algorithms produce similarly large SE gaps, and all show pass@k degradation at large k relative to the base model.

### 4. Distillation vs. RLVR

Unlike RLVR, knowledge distillation from a stronger teacher model (e.g., DeepSeek-R1) genuinely expands the reasoning boundary:

- After distillation, **both** pass@1 and pass@k (large k) improve over the base model.
- Distillation introduces reasoning patterns the base model had **zero or near-zero probability** of generating, as confirmed by high perplexity of distilled solutions under the original base model.

> [!IMPORTANT]
> RLVR and distillation are **fundamentally different** in what they teach. RLVR reshapes the distribution within the base model's existing solution space; distillation injects genuinely new reasoning trajectories from a stronger model.

## Input / Output Specification

| Component | Input | Output |
|-----------|-------|--------|
| Base LLM $\pi_{\text{base}}$ | Problem prompt $x \in \mathcal{X}$ | Response distribution $\pi_{\text{base}}(\cdot|x)$ |
| RLVR policy $\pi_\theta$ | Problem prompt $x$ | Biased response distribution $\pi_\theta(\cdot|x)$ with up-weighted reward-maximizing paths |
| pass@k evaluator | $n$ sampled responses $\{y_1, \ldots, y_n\}$, correctness labels | Scalar $\text{pass@}k \in [0, 1]$ |
| Perplexity scorer | Response $Y = (y_1, \ldots, y_T)$, reference model $m$ | Scalar $\text{PPL}_m(Y|x)$ |

## Comparison with Related Work

| Method | pass@1 | pass@k (large k) | Novel Reasoning? |
|--------|--------|-----------------|-----------------|
| RLVR (PPO/GRPO/etc.) | ↑ Improves | ↓ Degrades vs. base | No — redistributes existing paths |
| SFT on model's own outputs | ↑ Improves | ↓ Degrades | No — same distribution narrowing |
| Distillation from stronger model | ↑ Improves | ↑ Improves | Yes — introduces genuinely new paths |
| Base model (no fine-tuning) | Baseline | Higher at large k | N/A (reference) |

> [!NOTE]
> The paper directly contradicts concurrent work claiming RLVR "unlocks" new reasoning abilities. The key methodological distinction is that prior work only reports pass@1, which masks the boundary-shrinking effect visible only at large k.

# Experiments

- **Datasets (Math)**: GSM8K, MATH500, Minerva Math, Olympiad Bench, AIME24 (90 problems), AMC23 (40 problems)
- **Datasets (Code)**: LiveCodeBench (880 problems, May 2023 – January 2025), HumanEval+, MBPP+
- **Datasets (Vision)**: MathVista-TestMini (460 problems after filtering unsupported types), MathVision-TestMini (114 problems)
- **Models**: Qwen-2.5-7B, Qwen-2.5-14B, Qwen-2.5-32B, LLaMA-3.1-8B, Qwen-2.5-VL-7B (vision)
- **RL Training Frameworks**: SimpleRLZoo (GRPO on GSM8K+MATH), Code-R1 (GRPO on 12K LeetCode/TACO, 832 steps)
- **Sampling**: Temperature 0.6, top-p 0.95, max 16,384 tokens; $n = 200$ samples per problem for pass@k estimation
- **Key Result**: Across all benchmarks, model families, and RL algorithms, RLVR increases pass@1 but reduces pass@k at large k relative to the base model. Distillation is the only evaluated method that improves both simultaneously.
