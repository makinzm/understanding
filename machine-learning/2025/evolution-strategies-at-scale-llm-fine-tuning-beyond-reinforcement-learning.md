# Meta Information

- URL: [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/abs/2509.24372)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Qiu, X., Gan, Y., Hayes, C. F., Liang, Q., Meyerson, E., Hodjat, B., & Miikkulainen, R. (2025). Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning. arXiv:2509.24372.
- Affiliations: Cognizant AI Lab, MIT, UT Austin.

> [!NOTE]
> This summary is based on a structured extraction of the paper (ar5iv), not a line-by-line reading. Numbers below are as reported in that extraction; verify against the paper before relying on them.

# Overview

The paper scales Evolution Strategies (ES), a gradient-free, population-based optimizer, to full-parameter fine-tuning of billion-parameter LLMs, and compares it with RL fine-tuning (PPO, GRPO). Claimed advantages of ES: better sample efficiency, tolerance of long-horizon / outcome-only rewards, robustness across base models, less reward hacking, and more stable results across runs. No backpropagation is needed; only forward passes (inference) and a scalar reward per candidate.

**Applicability**: practitioners who fine-tune LLMs against a verifiable, outcome-level reward (e.g., task accuracy) and want an alternative to PPO/GRPO, especially when memory for gradients/optimizer states or run-to-run instability is a concern. Less applicable when dense per-token supervision is available (SFT is then cheaper).

# Method

## Problem setup

- Parameters $\theta \in \mathbb{R}^{d}$ ($d$ in the billions): all LLM weights.
- Reward $R(\theta) \in \mathbb{R}$: scalar score of the model's generated output(s) for a prompt/task (decoding is greedy, so the reward is deterministic given $\theta$).
- Output: updated $\theta$ that maximizes expected reward.

ES optimizes the Gaussian-smoothed objective:

```math
\begin{align}
  J(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I_d)}\left[ R(\theta + \sigma \epsilon) \right]
\end{align}
```

whose gradient can be estimated without backpropagation through the model:

```math
\begin{align}
  \nabla_\theta J(\theta) \approx \frac{1}{N \sigma} \sum_{n=1}^{N} R_n \, \epsilon_n
\end{align}
```

where $N$ is the population size, $\sigma$ the noise scale, $\epsilon_n \in \mathbb{R}^d$ the $n$-th noise vector, and $R_n = R(\theta + \sigma\epsilon_n)$.

## Algorithm (simplified natural evolution strategy, OpenAI-ES style)

Calculation order per iteration:

```
Input: pretrained weights θ, population size N, noise scale σ, learning rate α
for each iteration t:
    for n = 1..N (evaluated in parallel across GPUs):
        sample seed s_n
        ε_n ~ N(0, I) regenerated from s_n   # layer by layer, in place
        perturb θ ← θ + σ ε_n
        R_n ← reward of greedy-decoded outputs
        restore θ ← θ − σ ε_n                # regenerate noise from s_n
    Z-score normalize: R_n ← (R_n − mean(R)) / std(R)
    θ ← θ + α · (1/N) Σ_n R_n ε_n            # noise regenerated again from seeds
```

Implementation tricks that make billion-parameter ES feasible:
1. **Seed-based noise regeneration**: only the random seed is stored/communicated per candidate, never the $d$-dimensional noise.
2. **In-place, layer-wise perturbation**: noise is generated and added one layer at a time and then undone, so memory overhead stays near one layer.
3. **Parallel evaluation** of population members across GPUs.
4. **Z-score reward normalization** so the update is invariant to reward scale.
5. **Greedy decoding** so reward differences come from parameter perturbation rather than sampling noise.

Hyperparameters reported: $N = 30$, $\sigma = 0.001$, $\alpha = 5\times10^{-4}$.

## Difference from similar methods

| Method | Gradient | Exploration in | Notes |
|---|---|---|---|
| PPO / GRPO | Backprop through policy log-probs | Action (token) space via sampling | Needs credit assignment over tokens, value model (PPO) or group baseline (GRPO); KL penalty often used |
| ES (this paper) | Finite-difference estimate from scalar rewards | Parameter space | No backprop; one scalar reward per perturbed model; population of 30 |
| MeZO (Malladi et al., 2023) | Zeroth-order SPSA estimate | Parameter space | Used for supervised fine-tuning losses with few perturbations; ES here uses a larger population and a reward objective |
| OpenAI ES (Salimans et al., 2017) | Same estimator | Parameter space | Small policy networks; this paper scales it to LLMs |

# Experiments

- **Countdown task**: build an arithmetic expression from given numbers to reach a target; verifiable reward; sampled train and test questions.
- **Conciseness task**: reward short answers; 2 training prompts, 8 evaluation prompts with verifiable short answers (appendix).
- **Models**: Qwen-2.5 (0.5B, 7B) and LLaMA-3.1-8B among others.
- **Baselines**: PPO and GRPO (group size 8).
- **Results (Countdown accuracy %)**: Qwen-2.5-0.5B: base 0.1, PPO 0.5, ES 14.4. Qwen-2.5-7B: base 31.2, best RL 55.1, ES 54.2. LLaMA-3.1-8B: base 8.1, best RL 42.8, ES 49.9. Average relative improvement over base: ES 36.4%, PPO 17.9%, GRPO 21.3%.
- **Conciseness**: ES reaches a comparable or better reward-vs-KL tradeoff than GRPO without a KL penalty; reward std across 4 seeds is 0.004–0.008 for ES vs 0.041–0.062 for GRPO.
- Hardware / optimizer: not captured in the extraction; ES uses the plain update rule above.

# Discussion

Proposed explanations (hypotheses, not proven): rollouts perturb the whole model once and are evaluated with greedy decoding, giving lower-variance reward signals than token-level sampling; Gaussian smoothing flattens jagged reward landscapes; outcome-only rewards need no token-level credit assignment. Open directions include theory for why ES works at this dimensionality, and extending to more tasks.

> [!CAUTION]
> The ES-vs-RL comparison covers few tasks (mainly Countdown, plus conciseness); generality to open-ended tasks is untested.
