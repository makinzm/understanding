# Meta Information

- URL: [Influential Bandits: Pulling an Arm May Change the Environment](https://arxiv.org/abs/2504.08200)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sato, R., & Ito, S. (2025). Influential Bandits: Pulling an Arm May Change the Environment. Transactions on Machine Learning Research (TMLR).

# Influential Bandits: Pulling an Arm May Change the Environment

## Overview

This paper introduces the **influential bandit** problem, a multi-armed bandit variant where the act of pulling an arm permanently changes the reward distributions of other arms through an unknown interaction/influence matrix. Unlike classical stochastic bandits (independent, stationary rewards) or rotting bandits (only the pulled arm's reward degrades), influential bandits model the realistic scenario where an agent's actions reshape the entire reward landscape.

**Applicability:** Practitioners building recommendation systems, clinical trial designers, and anyone operating a sequential decision process where choices have cross-item spillover effects. For example, recommending one product changes user preferences, thus altering the appeal of all other products.

## Problem Formulation

### Setup

The learner interacts with $K$ arms over $T$ rounds. At each round $t \in [T]$:

1. The learner selects arm $a_t \in [K]$
2. Observes reward $r_t = \mu_{a_t}(t) + \eta_t$ where $\eta_t$ is sub-Gaussian noise
3. The mean reward vector **evolves** due to the pull

### Influence Model

Let $N_i(t) \in \mathbb{Z}_{\geq 0}$ denote the number of times arm $i$ has been pulled by the end of round $t$. The mean reward vector $\mu(t) \in \mathbb{R}^K$ evolves as:

$$\mu(t) = \mu(0) + M \cdot N(t)$$

where:
- $\mu(0) \in \mathbb{R}^K$ — initial mean rewards (unknown)
- $M \in \mathbb{R}^{K \times K}$ — **influence matrix** (unknown), with $M_{ij}$ representing the per-pull effect of arm $j$ on the mean reward of arm $i$
- $N(t) = (N_1(t), \ldots, N_K(t))^\top \in \mathbb{Z}_{\geq 0}^K$ — cumulative pull count vector

The influence matrix $M$ is assumed to be **symmetric** ($M = M^\top$) and to satisfy a **spectral radius** condition $\rho(M) < 1$ to ensure that rewards do not grow without bound:

$$\|\mu(t)\|_\infty \leq C \quad \text{for all } t$$

> [!IMPORTANT]
> The symmetry assumption $M = M^\top$ means that the influence is bidirectional: pulling arm $j$ affects arm $i$ by the same amount that pulling arm $i$ affects arm $j$. This is a key structural constraint enabling efficient learning.

### Comparison with Related Models

| Model | Reward Stationarity | Influence Scope | Known $M$? |
|---|---|---|---|
| Classical stochastic bandit | Stationary, i.i.d. | None | N/A |
| Rotting bandit | Decreasing (pulled arm only) | Self-influence only | N/A |
| Restless bandit | Markovian per-arm | Per-arm, independent | Partially |
| **Influential bandit (this work)** | Non-stationary, coupled | Cross-arm interactions | No |

## Regret Definition

The cumulative pseudo-regret is defined with respect to the **best arm in hindsight** under the influence dynamics. Because the optimal arm can change over time as the mean vector $\mu(t)$ shifts, the regret target is the expected total reward of the offline oracle that knows $M$ and $\mu(0)$:

$$\mathcal{R}(T) = \mathbb{E}\left[\sum_{t=1}^T \mu^*(t) - r_t\right]$$

where $\mu^*(t) = \max_i \mu_i(t)$ is the instantaneous best arm's mean at time $t$.

## Theoretical Results

### Lower Bounds (Theorem 2.2 and Theorem 2.3)

**Theorem 2.2 (Superlinear lower bound for naive algorithms):** Any algorithm that ignores the influence structure and treats each arm as stationary must incur regret at least $\Omega(T^{3/2} / K)$ in the worst case over all influence matrices with $\rho(M) \leq c < 1$.

> [!NOTE]
> This result shows that simply applying UCB or Thompson Sampling without accounting for cross-arm effects leads to superlinear (i.e., linear or worse) regret — catastrophically bad performance.

**Theorem 2.3 (Algorithm-independent lower bound):** Even the best possible algorithm must incur regret at least $\Omega(K \log T / (1 - \rho(M)))$ when $M$ is unknown. This establishes that the $\log T$ dependence on the time horizon is optimal.

### Upper Bound — LCB Algorithm (Theorem 2.4 and Theorem 2.6)

The paper proposes an **LCB (Lower Confidence Bound)** based algorithm that accounts for influence effects.

**Theorem 2.4 (Known $M$):** When $M$ is known, the LCB algorithm achieves:

$$\mathcal{R}(T) \leq O\!\left(\frac{K \log T}{1 - \rho(M)}\right)$$

matching the lower bound up to constants.

**Theorem 2.6 (Unknown $M$):** When $M$ must be estimated from data, the algorithm achieves:

$$\mathcal{R}(T) \leq \tilde{O}\!\left(K^2 \sqrt{T \log T}\right)$$

where $\tilde{O}$ hides poly-logarithmic factors. This reflects the additional cost of learning the $K^2$ entries of the influence matrix alongside decision-making.

## Algorithm: LCB for Influential Bandits

### Pseudocode

**Input:** $K$ arms, $T$ rounds, confidence parameter $\delta$

**Initialization:**
- Pull each arm once; initialize empirical mean $\hat{\mu}_i$, pull count $N_i \leftarrow 1$
- Initialize influence matrix estimate $\hat{M} \leftarrow 0 \in \mathbb{R}^{K \times K}$

**For each round $t = K+1, \ldots, T$:**

1. Compute current estimated mean:
   $$\hat{\mu}(t) = \hat{\mu}(0) + \hat{M} \cdot N(t-1)$$

2. Compute confidence radius for arm $i$:
   $$\beta_i(t) = \sqrt{\frac{2 \log(1/\delta)}{N_i(t-1)}} + \|\hat{M} - M\|_\infty \cdot \|N(t-1)\|_1$$

3. Compute **lower confidence bound**:
   $$\text{LCB}_i(t) = \hat{\mu}_i(t) - \beta_i(t)$$

4. Select arm:
   $$a_t = \arg\max_i \text{LCB}_i(t)$$

5. Observe reward $r_t$; update $\hat{\mu}$, $N$, and $\hat{M}$ via ordinary least squares

**Output:** Sequence of arm selections $a_1, \ldots, a_T$

> [!IMPORTANT]
> Unlike UCB (Upper Confidence Bound) which adds the confidence term, this algorithm uses LCB (Lower Confidence Bound). The intuition is that when $M$ has positive entries, arms that have been pulled a lot have artificially inflated means, so pessimism about current rewards is appropriate — the algorithm prefers arms whose future influence potential is high, not those with optimistically inflated current means.

### Why LCB Instead of UCB?

In standard (stationary) bandits, UCB works because reward uncertainty is symmetric. In influential bandits, pulling an arm may have raised competing arms' means via positive off-diagonal $M$ entries. UCB would over-exploit arms whose means were inflated by past pulls. LCB corrects for this by being pessimistic about arms that appear currently optimal but whose advantage may be transient.

## Comparison with Similar Algorithms

| Algorithm | Setting | Regret | Handles Cross-Arm Influence? |
|---|---|---|---|
| UCB1 (Auer et al., 2002) | Stationary | $O(K \log T / \Delta)$ | No — $\Omega(T^{3/2})$ in influential setting |
| Thompson Sampling | Stationary | $O(K \log T / \Delta)$ | No |
| Rotting Bandit (Levine et al., 2017) | Self-decaying arms | $O(\sqrt{KT})$ | No |
| Whittle Index (Whittle, 1988) | Restless, Markovian | PSPACE | Per-arm Markov only |
| **Influential LCB (this work, known $M$)** | Cross-arm influence | $O(K \log T)$ | Yes |
| **Influential LCB (this work, unknown $M$)** | Cross-arm influence | $\tilde{O}(K^2 \sqrt{T})$ | Yes |

# Experiments

- **Datasets:**
  - **MovieLens** (standard recommendation benchmark) — used to simulate the influence of recommending item $j$ on user preference for item $i$; the influence matrix $M$ is estimated from rating history
  - **Synthetic data** — generated from known influence matrices $M$ with varying spectral radii $\rho(M) \in \{0.3, 0.6, 0.9\}$ to validate that regret scaling matches theoretical predictions
- **Baselines:** UCB1, Thompson Sampling, and a "naive" stationary bandit algorithm that ignores influence
- **Hardware:** Not specified
- **Optimizer:** N/A (bandit algorithm, not neural network training)
- **Key results:**
  - On synthetic data, the LCB algorithm's empirical regret matches $O(\log T)$ for known $M$, while UCB1 exhibits $O(T^{3/2})$ regret — confirming the theoretical gap
  - On MovieLens, the influential LCB algorithm achieves 20–40% lower cumulative regret compared to UCB1 depending on $\rho(M)$, demonstrating practical gains from modeling cross-item influence
  - Regret degrades gracefully as $\rho(M) \to 1$: the $(1 - \rho(M))^{-1}$ factor in the bound correctly predicts increasing difficulty
