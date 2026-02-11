# Meta Information

- URL: [Influential Bandits: Pulling an Arm May Change the Environment](https://arxiv.org/abs/2504.08200)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sato, R., & Ito, S. (2025). Influential Bandits: Pulling an Arm May Change the Environment. Transactions on Machine Learning Research (TMLR).

# Influential Bandits: Pulling an Arm May Change the Environment

## Problem Setting

Classical multi-armed bandit (MAB) problems assume that pulling arm $i$ produces a reward sampled i.i.d. from a fixed distribution $\mathcal{D}_i$, independent of which other arms were pulled previously. The influential bandit problem relaxes this assumption: **pulling one arm may change the loss (or reward) distribution of other arms**.

The setting uses a **loss-based formulation** (minimize cumulative loss, equivalently maximize negative reward). At each round $t \in [T]$, the learner selects arm $a_t \in [K]$ and suffers loss $\ell_{t, a_t}$.

**The interaction model** defines how arm pulls affect future losses via an unknown symmetric positive semi-definite (PSD) interaction matrix $M \in \mathbb{R}^{K \times K}$. The expected loss of arm $i$ at round $t$ is:

$$\mu_{t,i} = \mu_{0,i} + \sum_{s < t} M_{i, a_s}$$

where:
- $\mu_{0,i} \in \mathbb{R}$ is the initial expected loss of arm $i$
- $M_{i,j} \geq 0$ captures how pulling arm $j$ increases the expected loss of arm $i$ (arms become "worse" the more frequently related arms are pulled)
- $M$ is symmetric ($M_{i,j} = M_{j,i}$) and positive semi-definite

**Input/Output per round:**
- Input: $K$ arms; at time $t$, $t-1$ previous arm selections $a_1, \ldots, a_{t-1}$
- Output: arm selection $a_t \in [K]$ to minimize total cumulative loss $\sum_{t=1}^{T} \ell_{t, a_t}$

> [!NOTE]
> The PSD constraint on $M$ ensures that the influence structure is self-consistent: arm pulls can only increase expected losses (or leave them unchanged), never decrease them arbitrarily. This models saturation effects in recommender systems, where repeatedly recommending similar items reduces user engagement across all similar items.

## Comparison with Related Bandit Models

| Model | Stationarity | Inter-arm influence | Key assumption |
|---|---|---|---|
| Standard MAB | Stationary | None | i.i.d. rewards per arm |
| Rotting bandits | Non-stationary | None | Each arm independently degrades |
| Restless bandits | Non-stationary | None | Arms evolve independently via Markov chain |
| **Influential bandits** | **Non-stationary** | **Yes (via $M$)** | **PSD interaction matrix governs coupling** |

The crucial distinction: in rotting and restless bandits, arm dynamics are independent. In influential bandits, pulling arm $j$ explicitly changes the loss of arm $i$ through $M_{i,j}$. This makes the optimal arm selection problem a **coupled optimization** across all arms.

## Regret Analysis

**Regret definition**: The learner's cumulative regret relative to the best single-arm policy (hindsight optimal) over $T$ rounds:

$$R_T = \sum_{t=1}^T \ell_{t, a_t} - \min_{i \in [K]} \sum_{t=1}^T \ell_{t,i}$$

> [!IMPORTANT]
> The baseline here is the best **fixed** arm in hindsight, not the best adaptive policy. This is weaker than full adaptivity, but more tractable for analysis.

### Lower Bounds

**Theorem (Superlinear lower bound for LCB)**: Standard Lower Confidence Bound (LCB) algorithms, which were designed for stationary or rotting bandits, suffer regret at least $\Omega(T^2 / \log^2 T)$ in the influential bandit setting. This superlinear rate arises because LCB underestimates arms that have been pulled frequently (they look "cheap" despite causing degradation in others).

**Theorem (Algorithm-independent lower bound)**: For the influential bandit problem, any algorithm must suffer at least $\Omega(T)$ regret in the worst case over the unknown interaction matrix $M$.

### Upper Bound

**Proposed algorithm**: A modified LCB estimator designed specifically for the influential bandit's loss dynamics. The key modification accounts for the accumulated influence of past arm pulls when computing confidence bounds.

**Algorithm (Influential-LCB) — pseudocode:**

```
Input: K arms, T rounds, confidence parameter δ
Initialize: pull each arm once; maintain empirical loss estimates ℓ̂_{t,i}

For t = K+1 to T:
  1. Compute adjusted loss estimate for arm i:
       ℓ̃_{t,i} = ℓ̂_{t,i} - (influence correction based on M̂ and pull counts)
  2. Compute confidence radius: r_{t,i} = √(log(T/δ) / N_{t,i})
     where N_{t,i} = number of times arm i has been pulled up to t
  3. Select arm: a_t = argmin_i (ℓ̃_{t,i} - r_{t,i})
  4. Observe loss ℓ_{t,a_t} and update estimates
```

**Theorem (Upper bound)**: The Influential-LCB algorithm achieves:

$$R_T = O(KT \log T)$$

with high probability over $T$ rounds with $K$ arms. The $K$ factor reflects the cost of estimating the $K \times K$ interaction matrix $M$, and the $\log T$ factor comes from standard confidence interval arguments.

> [!CAUTION]
> The exact form of the influence correction in Influential-LCB depends on the matrix $M$, which is unknown. The algorithm estimates $M$ on-the-fly using observed loss changes, adding a learning-within-learning structure. The pseudocode above is a high-level description; precise implementation details require careful handling of the estimation error for $M$.

## Gap Between Upper and Lower Bounds

The algorithm-independent lower bound is $\Omega(T)$ and the proposed algorithm achieves $O(KT \log T)$. A gap of factor $K \log T$ remains, and it is an open question whether a tighter $O(T \log T)$ bound is achievable or whether a stronger lower bound of $\Omega(KT)$ holds.

## Applications and Scope

**Who uses this**: Practitioners deploying sequential decision systems where actions have externalities, including:
- **Recommender systems**: Recommending item $i$ repeatedly lowers engagement with similar items $j$ (content fatigue)
- **Clinical trials**: Administering treatment $i$ may alter biomarkers relevant to treatment $j$
- **Online advertising**: Showing ad $i$ repeatedly may reduce click-through for related ads

**When applicable**: When the environment is non-stationary and the non-stationarity is caused by the learner's own actions (endogenous dynamics), and the influence structure is approximately PSD.

**Where not applicable**: When arms influence each other in non-PSD ways (e.g., adversarial or oscillatory interactions), or when influence is asymmetric in ways that violate the symmetric $M$ assumption.

# Experiments

- **Datasets**: Synthetic environments with controlled influence patterns; MovieLens dataset (real-world recommender system scenario with inter-item influence)
- **Baselines**: Standard LCB/UCB algorithms, Thompson Sampling, and rotting bandit algorithms
- **Hardware**: Not specified
- **Optimizer**: Not applicable (online learning setting, no gradient-based optimization)
- **Results**:
  - Standard LCB confirms the $\Omega(T^2 / \log^2 T)$ superlinear regret empirically on synthetic data
  - Influential-LCB achieves $O(KT \log T)$ regret, consistent with theoretical guarantees
  - On the MovieLens dataset, the influential bandit model better captures inter-item dependencies compared to standard bandit baselines, demonstrating that real-world recommendation data exhibits the inter-arm influence structure assumed by the model
