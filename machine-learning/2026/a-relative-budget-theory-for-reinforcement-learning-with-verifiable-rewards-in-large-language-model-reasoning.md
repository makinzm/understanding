# Meta Information

- URL: [A Relative-Budget Theory for Reinforcement Learning with Verifiable Rewards in Large Language Model Reasoning](https://arxiv.org/abs/2602.01523)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html); Paper content under [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Wachi, A., Kinoshita, H., Takakura, S., Higuchi, R., & Suzuki, T. (2026). A Relative-Budget Theory for Reinforcement Learning with Verifiable Rewards in Large Language Model Reasoning. arXiv:2602.01523.

---

# A Relative-Budget Theory for RL with Verifiable Rewards in LLM Reasoning

## Overview

This paper provides a theoretical framework explaining **why reinforcement learning (RL) effectiveness varies across tasks and compute budgets** when training large language models (LLMs) on reasoning tasks with verifiable rewards. The central contribution is the **relative budget** metric $\xi$, which quantifies how the token generation capacity relates to the expected difficulty of the task. The framework identifies three distinct operational regimes and provides finite-sample guarantees for online RL algorithms (e.g., GRPO), empirically validated on GSM8K and MATH-500 with Llama-3.2-3B, Phi-4-mini, and Qwen3-4B.

**Who benefits:** Researchers and practitioners designing RL training pipelines for LLM reasoning, particularly those selecting token budgets, filtering training data by difficulty, or analyzing why RL succeeds or fails on specific tasks.

---

## Problem Setup

### Setting

- A language model (base policy $\pi_b$) generates reasoning trajectories $\tau = (a_1, a_2, \ldots, a_H)$ over a token generation horizon $H$.
- The model is trained on tasks $x$ drawn from a distribution; each task has a verifiable correct answer.
- The **reward** is a length-penalized binary signal:

$$
R(\tau) = \begin{cases} H - T(\tau) + 1 & \text{if } T(\tau) \leq H \\ 0 & \text{otherwise} \end{cases}
$$

where $T(\tau)$ is the position of the first correct solution token in trajectory $\tau$ (i.e., the number of tokens until correctness is first achieved). A trajectory receives reward only if the solution appears within the budget; longer solutions receive lower rewards.

- **Input:** A reasoning task $x$ (e.g., a math problem).
- **Output:** A trajectory $\tau \in \mathcal{A}^H$ (a sequence of at most $H$ tokens), together with a scalar reward $R(\tau) \in [0, H]$.

### Key Quantities

| Symbol | Definition |
|--------|-----------|
| $H$ | Generation horizon (token budget) |
| $T(\tau)$ | Number of tokens until first correct solution in trajectory $\tau$ |
| $\mu_x = \mathbb{E}_{\tau \sim \pi_b(\cdot \mid x)}[T(\tau)]$ | Expected tokens to solution under base policy for task $x$ |
| $\sigma_{b,x}^2$ | Variance of $R(\tau)$ under base policy for task $x$ |
| $\mathcal{R}$ | Reward function class; $\lvert \mathcal{R} \rvert$ is its cardinality |

---

## Core Concept: Relative Budget

**Definition 4.1 (Relative Budget):**

$$
\xi_x \coloneqq \frac{H}{\mu_x} = \frac{H}{\mathbb{E}_{\tau \sim \pi_b(\cdot \mid x)}[T(\tau)]}
$$

The relative budget $\xi_x \in (0, \infty)$ is the ratio of the token budget to the expected number of tokens the base policy needs to reach a correct solution. It is a **dimensionless difficulty-normalized budget indicator**:

- $\xi_x \ll 1$: The budget is insufficient — the model rarely produces a correct solution within the allotted tokens.
- $\xi_x = \Theta(1)$: The budget is commensurate with task difficulty — balanced regime.
- $\xi_x \gg 1$: The budget greatly exceeds task difficulty — ample regime.

> [!IMPORTANT]
> The relative budget $\xi$ is computed with respect to the **base policy** (before RL training), not the current policy. As training progresses, $\xi$ changes because $\mu_x$ decreases as the model improves.

---

## Anti-Concentration Coefficient

**Definition 3.1 (Anti-Concentration Coefficient):** For $\varepsilon > 0$ and task $x$,

$$
c_x(\varepsilon) \coloneqq \mathbb{P}_{\tau \sim \pi_b(\cdot \mid x)}\!\left[R(\tau) \geq \mathbb{E}[R(\tau)] + \sigma_{b,x} \sqrt{\varepsilon}\right]
$$

This measures how likely a trajectory is to be **significantly above-average** in reward — i.e., how often the model produces an informative training signal. A larger $c_x(\varepsilon)$ means more frequent high-quality positive examples, yielding better sample efficiency.

The **regime-averaged anti-concentration coefficient** $c_0(\xi; \kappa)$ is defined over a threshold $\kappa > 0$ and governs the theoretical sample complexity bounds.

> [!NOTE]
> The anti-concentration coefficient $c_0$ directly controls the sub-optimality gap in Theorem 5.2: larger $c_0$ → lower sample complexity → faster learning.

---

## Three Regimes

The relative budget $\xi$ partitions the learning dynamics into three qualitatively distinct regimes:

### 1. Deficient Regime ($\xi \to 0$)

- High-reward trajectories are exponentially rare: the probability $\mathbb{P}[T(\tau) \leq H]$ approaches 0 as $\xi \to 0$.
- Anti-concentration: $c_0(\xi; \kappa) = \Theta(f(\xi))$, where $f(\xi) \to 0$ as $\xi \to 0$ (governed by the left-tail behavior of $T$).
- Sample complexity **diverges**: learning is theoretically prohibitive because informative reward signals are nearly absent.
- Supervised fine-tuning (SFT) is also problematic here since there are few positive examples to learn from.

### 2. Balanced Regime ($\xi = \Theta(1)$)

- Anti-concentration: $c_0(\xi; \kappa) = \Theta(1)$ — constant and maximized.
- RL achieves **maximal sample efficiency**: informative trajectories appear with non-negligible probability.
- SFT suffers from **peak solution heterogeneity**: multiple valid reasoning paths exist with different lengths, making it hard for SFT to identify a consistent training signal.
- This is the regime where RL has the clearest advantage over SFT.

### 3. Ample Regime ($\xi \to \infty$)

- Anti-concentration: $c_0(\xi; \kappa) = \Theta(1)$ — stable and bounded.
- Learning remains reliable but **per-iteration gains diminish**: the model already solves the problem often, so additional samples yield marginal improvement.
- Empirically, excessive budget wastes compute since most trajectories are already correct.

---

## Theoretical Results

### Assumptions

**Assumption 4.2 (Variance Scaling):** $v_x = \mathbb{V}[T(\tau) \mid T(\tau) \leq H] = \Theta(\mu_x^2)$ — conditional variance scales quadratically with the mean, ensuring non-trivial reward spread.

**Assumption 4.3 (Left-Tail Regularity):** For $t \leq \mu_x$ with $z_x = t / \mu_x \in [0, z_0]$:

$$
c_- f(z_x) \leq \mathbb{P}[T(\tau) \leq t] \leq c_+ f(z_x)
$$

where $f$ satisfies a polynomial doubling condition: $f(2z) \leq c_f \cdot f(z)$. This assumption controls how quickly the left tail of $T$ decays (i.e., how often the model produces very short correct solutions).

**Assumption 4.4 (Balanced Non-Degeneracy):** Within $\xi_x \in [\xi_{\min}, \xi_{\max}]$, success probabilities satisfy $q_{\min} \leq q_x \leq 1 - q_{\min}$ and $\mathbb{V}[T(\tau) \mid T(\tau) \leq H] \geq c_v \cdot v_x$.

### Theorem 5.2 (Offline RL Sub-Optimality Gap)

For the offline RL estimator $\hat{\pi}^n_{\mathrm{RL}}$ with $n$ i.i.d. rollouts, the sub-optimality gap satisfies:

$$
J(\bar{\pi}_\kappa) - J(\hat{\pi}^n_{\mathrm{RL}}) \lesssim \frac{H \cdot \log(\lvert \mathcal{R} \rvert / \delta)}{c_0(\xi; \kappa) \cdot n}
$$

with probability $\geq 1 - \delta$. Here $\bar{\pi}_\kappa$ is the optimal policy in class $\mathcal{R}$ and $J(\pi) = \mathbb{E}[R(\tau)]$.

**Regime implications:**
- Deficient ($\xi \to 0$): $c_0 \to 0$, so $n \to \infty$ samples required.
- Balanced ($\xi = \Theta(1)$): $c_0 = \Theta(1)$, minimal sample complexity.
- Ample ($\xi \to \infty$): $c_0 = \Theta(1)$, but policy improvement is small since $J(\bar{\pi}_\kappa) - J(\hat{\pi}^n_{\mathrm{RL}}) \to 0$ anyway.

### Theorem 6.2 (One-Step Online Improvement)

For online RL iteration $i$ with $n_i$ samples satisfying:

$$
n_i \geq \frac{2CH \cdot \log(\lvert \mathcal{R} \rvert / \delta)}{\sqrt{\kappa_i} \cdot c_0(\xi_i; \kappa_i) \cdot \sigma(\pi^{(i)})}
$$

the policy $\pi^{(i+1)}$ satisfies:

$$
J(\pi^{(i+1)}) - J(\pi^{(i)}) \geq \frac{1}{2} \sqrt{\kappa_i} \cdot \sigma(\pi^{(i)})
$$

The improvement magnitude $\frac{1}{2}\sqrt{\kappa_i} \cdot \sigma(\pi^{(i)})$ is proportional to the reward standard deviation $\sigma(\pi^{(i)})$ — a larger spread of rewards yields larger per-step gains.

### Theorem 6.3 (Online RL Sample Complexity by Regime)

The required samples per iteration $n_i$ scale as:

| Regime | Required $n_i$ |
|--------|----------------|
| Deficient ($\xi \to 0$) | $\tilde{\Omega}\!\left(\kappa_i^{-1/2} (f(\xi_i))^{-3/2}\right)$ |
| Balanced ($\xi = \Theta(1)$) | $\tilde{\Omega}\!\left(1/\sqrt{\kappa_i}\right)$ |
| Ample ($\xi \to \infty$) | $\tilde{\Omega}\!\left(\xi_i / \sqrt{\kappa_i}\right)$ |

The balanced regime minimizes sample requirements; the deficient regime is worst-case.

### Theorem 6.4 (Linear Budget Growth under Gamma Model)

Under the concrete model $T(\tau) \mid x \sim \mathrm{Gamma}(K, p)$ (with $K$ shape and $p$ rate), the relative budget grows linearly across online RL iterations:

$$
\xi_i - \xi_0 \geq \frac{i}{2K} - \mathcal{O}_K(1)
$$

Required rollouts satisfy $n_i = \tilde{\Theta}(K \xi_i^2)$. The budget begins in the balanced regime and transitions to ample as training continues, explaining observed empirical saturation.

---

## Algorithm: Group Relative Policy Optimization (GRPO)

The paper uses GRPO as the concrete online RL algorithm for empirical validation. GRPO operates as follows:

**Pseudocode:**
```
Input: base policy π_b, task distribution p(x), rollouts G per task, horizon H
Initialize: π^(0) ← π_b, iteration i ← 0

Repeat until convergence:
  1. Sample batch of tasks {x_1, ..., x_B} from p(x)
  2. For each task x_j:
       a. Sample G trajectories {τ_{j,1}, ..., τ_{j,G}} ~ π^(i)(· | x_j) with max length H
       b. Compute reward r_{j,g} = R(τ_{j,g}) for each trajectory
       c. Compute group mean: μ_j = (1/G) Σ_g r_{j,g}
       d. Compute group std: σ_j = std({r_{j,g}})
       e. Normalize: ĝ_{j,g} = (r_{j,g} - μ_j) / (σ_j + ε)  [group-relative reward]
  3. Update policy:
       π^(i+1) ← argmax_π Σ_{j,g} ĝ_{j,g} · log π(τ_{j,g} | x_j)
                  subject to KL(π || π^(i)) ≤ δ_KL
  i ← i + 1
```

The group-relative normalization $\hat{g}_{j,g}$ serves as a variance-reduction baseline, making GRPO equivalent to a self-normalized importance-weighted policy gradient.

---

## Comparison with Supervised Fine-Tuning (SFT)

| Aspect | RL (GRPO) | Supervised Fine-Tuning (SFT) |
|--------|-----------|------------------------------|
| Deficient regime ($\xi \to 0$) | Prohibitively expensive (rare rewards) | Also difficult (few positive examples) |
| Balanced regime ($\xi = \Theta(1)$) | **Maximum efficiency** | **Worst case** (high solution heterogeneity) |
| Ample regime ($\xi \to \infty$) | Stable but diminishing returns | Efficient (abundant, consistent solutions) |
| Training signal | Relative reward contrast within group | Imitation of demonstrated solutions |
| Data requirement | Unlabeled problems with verifier | Labeled (problem, solution) pairs |

> [!IMPORTANT]
> The balanced regime ($\xi \approx 1$–$2$) is exactly where RL has the **largest advantage over SFT**: RL can extract learning signal from heterogeneous trajectories via relative comparison, while SFT struggles because multiple valid reasoning paths of different lengths create conflicting supervision signals.

---

## Relationship to Difficulty Filtering

The relative budget framework provides a theoretical justification for **difficulty-based data filtering** in RL training:

- Tasks with $\xi \ll 1$ (too hard): exclude because informative trajectories are too rare.
- Tasks with $\xi \gg 1$ (too easy): deprioritize because policy improvement is marginal.
- Tasks with $\xi \approx 1$–$2$ (just right): include because they yield maximum learning signal.

This aligns with the empirical finding (Bae et al., 2025) that filtering for tasks where the model's success rate is around 50% maximizes training efficiency — such tasks correspond precisely to the balanced regime.

---

## Experiments

- **Datasets:** GSM8K (grade school math, 8,500 training / 1,319 test problems), MATH-500 (500 competition math problems across 5 difficulty levels)
- **Models:** Llama-3.2-3B, Phi-4-mini, Qwen3-4B
- **Training algorithm:** GRPO with group size $G = 8$, horizon $H \in \{512, 1024, 2048\}$
- **Optimizer:** AdamW
- **Evaluation metric:** Pass@1 accuracy on test sets
- **Hardware:** Not specified in the abstract page (see full paper for details)

**Key results:**
- Phase transition in learning signal quality observed near $\xi \approx 1.0$ across all model-dataset combinations.
- Peak improvement rates measured at $\xi \in [1.5, 2.0]$, consistent with the theoretical balanced regime prediction.
- Both deficient ($\xi < 0.5$) and ample ($\xi > 5$) regimes show substantially lower per-iteration improvement, matching Theorems 6.3 and 6.4.
- Gamma distribution model fits empirical $T(\tau)$ distributions closely (confirmed via goodness-of-fit tests), validating Theorem 6.4's assumptions.

---

## Related Work

| Work | Relationship |
|------|-------------|
| GRPO (Shao et al., 2024) | The RL algorithm used in empirical validation |
| DeepSeek-R1 (DeepSeek-AI, 2025) | Motivated the interest in RL for LLM reasoning |
| Bae et al. (2025) | Empirical finding that ~50% success rate is optimal; this paper provides the theoretical explanation |
| Zhang et al. (2025) | "Edge of competence" framing; consistent with balanced regime |
| RLVR (Ziegler et al., 2019) | Foundational work on RL with verifiable rewards |
| PAC learning theory | The finite-sample guarantees build on PAC-style analysis |

---

## Practical Guidelines

Based on the theoretical framework, practitioners designing RL training pipelines for LLM reasoning should:

1. **Estimate $\xi$ before training**: Measure the base policy's success rate on each task to estimate $\mu_x$, then compute $\xi_x = H / \mu_x$.
2. **Filter training tasks to $\xi \in [1.5, 2.0]$**: This ensures maximum learning signal per rollout.
3. **Avoid $\xi < 0.5$**: Tasks where the model almost never solves within budget produce negligible reward variance and waste compute.
4. **Monitor $\xi$ during training**: As the model improves, $\xi$ increases. Once $\xi \gg 2$, consider increasing task difficulty or reducing $H$ to stay in the balanced regime.
5. **Prefer RL over SFT in the balanced regime**: When $\xi \approx 1$–$2$, the solution heterogeneity that hurts SFT is precisely what RL exploits through relative reward comparison.
