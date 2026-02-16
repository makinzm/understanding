# Meta Information

- URL: [Dynamic Causal Effects Evaluation in A/B Testing with a Reinforcement Learning Framework](https://arxiv.org/abs/2002.01711)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shi, C., Wang, X., Luo, S., Zhu, H., Ye, J., & Song, R. (2020). Dynamic Causal Effects Evaluation in A/B Testing with a Reinforcement Learning Framework. arXiv:2002.01711.

---

# Dynamic Causal Effects Evaluation in A/B Testing with a Reinforcement Learning Framework

## Problem Setting and Motivation

Standard A/B testing assumes the Stable Unit Treatment Value Assumption (SUTVA), meaning that a unit's outcome depends only on its current treatment and not on past treatments or outcomes. In two-sided marketplace platforms such as ride-sharing services, this assumption fails because:

- A driver's income today depends on the dispatching strategy applied in previous time steps (carryover effects).
- Sequential treatments alter the platform state (e.g., demand/supply balance), which in turn influences future rewards.
- Classical t-tests and double machine learning (DML) methods cannot detect these dynamic treatment-outcome interactions and consequently have near-zero power in these scenarios.

This paper frames the problem as evaluating the Average Treatment Effect (ATE) between two sequential policies in a Markov Decision Process (MDP), and develops the first sequential (group-sequential) testing procedure valid in this RL setting.

**Applicability:** Platform companies and researchers running online controlled experiments with sequential treatments, where early stopping is valuable and carryover effects are present (e.g., recommendation systems, pricing algorithms, dispatch systems).

---

## Framework: MDP for A/B Testing

### State-Treatment-Outcome Triplets

At each time step $t$, observe a triplet $(S_t, A_t, Y_t)$:

- $S_t \in \mathcal{S}$: state (e.g., demand, supply, equilibrium metric on a ride-sharing platform).
- $A_t \in \{0, 1\}$: treatment assignment (0 = control policy $\pi^{(0)}$, 1 = treatment policy $\pi^{(1)}$).
- $Y_t \in \mathbb{R}$: immediate reward (e.g., driver income per time unit).

The data-generating process satisfies the Markov property:

$$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = \mathcal{P}(S_{t+1} \mid S_t, A_t)$$

### Value Function and Causal Estimand

The discounted value function under policy $\pi$ starting from state $s$ is:

$$V(\pi; s) = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}\{Y_t^*(\pi) \mid S_0 = s\}$$

where $Y_t^*(\pi)$ is the potential outcome under policy $\pi$ and $\gamma \in [0,1)$ is the discount factor.

The **Average Treatment Effect (ATE)** — the primary causal estimand — is:

$$\tau = \int_{\mathcal{S}} \left[ V(\pi^{(1)}; s) - V(\pi^{(0)}; s) \right] \nu(ds)$$

where $\nu$ is the initial state distribution. Testing $H_0: \tau \leq 0$ vs. $H_1: \tau > 0$.

---

## Identification Assumptions

Four assumptions are required for identification:

1. **Consistency:** $Y_t = Y_t^*(A_t, A_{t-1}, \ldots) = Y_t^*(\pi)$ when treatments follow $\pi$.
2. **Sequential Randomization:** $A_t \perp \{Y_t^*(\bar{a}) : \bar{a}\} \mid S_t, A_{t-1}, S_{t-1}, \ldots$ (no unmeasured confounders).
3. **Markov Assumption:** $(S_t, A_t)$ is sufficient for the future.
4. **Conditional Mean Independence:** $\mathbb{E}(Y_t \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = r(A_t, S_t)$ for some function $r$.

Under these assumptions, the Q-function $Q(a; s, a') = r(a, s) + \gamma \mathbb{E}\{V(\pi; S_{t+1}) \mid S_t = s, A_t = a\}$ satisfies the **Bellman equation**:

$$\mathbb{E}\left[\left\{Q(a'; A_t, S_t) - Y_t - \gamma Q(a'; a', S_{t+1})\right\} \phi(S_t, A_t)\right] = 0$$

for any basis function $\phi(S_t, A_t)$, where $a' \in \{0,1\}$ denotes the policy being evaluated.

---

## Estimation via Temporal-Difference Learning

### Basis Function Approximation (Sieve Estimator)

Let $\Psi(s) \in \mathbb{R}^q$ be a vector of basis functions of the state (e.g., polynomial basis of degree $J$). The Q-function is approximated as:

$$Q(a'; s, a) \approx \beta_{a'}^\top \Psi(s) \cdot \mathbf{1}(a = a') + \text{(other terms)}$$

The full parameter vector $\beta \in \mathbb{R}^{2q}$ stacks coefficients for both policies $a' \in \{0,1\}$.

### Estimating Equations (Temporal-Difference Residuals)

At time $t$, with cumulative data up to time $T_k$ at interim stage $k$, form:

$$\hat{\Sigma}(T_k) = \frac{1}{T_k} \sum_{t=1}^{T_k} \Phi_t \Phi_t^\top - \gamma \Phi_t \tilde{\Phi}_{t+1}^\top \quad \in \mathbb{R}^{2q \times 2q}$$

$$\hat{\eta}(T_k) = \frac{1}{T_k} \sum_{t=1}^{T_k} Y_t \Phi_t \quad \in \mathbb{R}^{2q}$$

where $\Phi_t$ encodes the basis evaluated at $(A_t, S_t)$ and $\tilde{\Phi}_{t+1}$ at the next state under the target policy.

The **parameter estimate** is:

$$\hat{\beta}(T_k) = \hat{\Sigma}^{-1}(T_k) \hat{\eta}(T_k)$$

### ATE Estimator

Using a contrast vector $U \in \mathbb{R}^{2q}$ that computes $V(\pi^{(1)};s) - V(\pi^{(0)};s)$ integrated over $\nu$:

$$\hat{\tau}(T_k) = U^\top \hat{\beta}(T_k)$$

Variance estimate:

$$\hat{\sigma}^2(T_k) = U^\top \hat{\Sigma}^{-1}(T_k) \hat{\Omega}(T_k) \left(\hat{\Sigma}^{-1}(T_k)\right)^\top U$$

where $\hat{\Omega}(T_k)$ estimates the asymptotic sandwich variance of the temporal-difference residuals.

---

## Three Treatment Design Types

The framework supports three distinct assignment mechanisms:

| Design | Description | Positivity Required |
|--------|-------------|---------------------|
| **D1 (Markov)** | $P(A_t=1 \mid S_t) = b^{(0)}(S_t)$, propensity bounded away from 0 and 1 | Yes |
| **D2 (Alternating)** | $A_{2j}=0$, $A_{2j+1}=1$ deterministically | No |
| **D3 (Adaptive)** | Policy updated at interim stages using $\hat{\beta}$ to assign optimal actions | Depends on stage |

D2 is practically convenient for platforms that alternate strategies in fixed time windows (e.g., 30-minute intervals). D3 enables combining experimental learning with operational improvement.

---

## Sequential Testing Procedure

### Group-Sequential Test Statistic

At each interim stage $k = 1, \ldots, K$, after observing $T_k$ time steps, compute:

$$Z_k = \frac{\sqrt{T_k} \, \hat{\tau}(T_k)}{\hat{\sigma}(T_k)}$$

Reject $H_0$ at stage $k$ if $Z_k > b_k$, where $\{b_k\}$ are stopping thresholds satisfying an $\alpha$-spending function $\alpha(t)$ (e.g., O'Brien-Fleming: $\alpha(t) = 2[1 - \Phi(z_{\alpha/2}/\sqrt{t})]$).

### Bootstrap for Threshold Determination

Rather than numerically integrating multivariate normal probabilities (which requires knowing the full joint distribution of $\{Z_k\}$), a multiplier bootstrap is used:

**Algorithm:**

```
Input: Accumulated data {(S_t, A_t, Y_t)} up to stage k, B bootstrap samples

For each interim stage k:
  1. For b = 1, ..., B:
     a. Sample i.i.d. multipliers ζ_1^b, ..., ζ_{T_k}^b ~ N(0,1)
     b. Compute bootstrap parameter:
        β̂^{MB}(T_k) = Σ̂^{-1}(T_k) · (1/T_k) Σ_t ζ_t^b · Φ_t(Y_t - Φ̃_{t+1}^T β̂)
     c. Compute bootstrap statistic:
        Z_k^{MB,b} = √T_k · U^T β̂^{MB}(T_k) / σ̂(T_k)

  2. Determine threshold b̂_k:
     Solve for b̂_k such that:
       P*(Z_k^{MB} > b̂_k | Z_{k-1}^{MB} ≤ b̂_{k-1}, ...) = [α(T_k) - α(T_{k-1})] / [1 - α(T_{k-1})]
     using empirical quantiles of bootstrap samples

  3. Reject H_0 at stage k if Z_k > b̂_k; otherwise continue
```

**Key advantage:** Multiplier bootstrap statistics $Z_k^{MB}$ are conditionally i.i.d. N(0,1) given the data, enabling the conditional rejection probabilities to be approximated from $B$ samples without storing historical trajectories.

**Computational complexity:**
- Bootstrap: $O(BKq^2 + Tq^2 + Kq^3)$
- Classical wild bootstrap (requiring full history): $\Omega(BTq^2)$

Online updating of $\hat{\Sigma}$ and $\hat{\eta}$ is performed incrementally, so only current-stage summaries need to be retained.

---

## Theoretical Guarantees

### Theorem 1 (Joint Asymptotic Normality)

Under geometric ergodicity of the MDP and mild regularity conditions (bounded basis functions, consistent variance estimation), the joint distribution of test statistics converges:

$$\left(Z_1, \ldots, Z_K\right) \xrightarrow{d} \mathcal{N}(\mu, \Sigma^*)$$

where $\mu_k \leq 0$ under $H_0$ and $\Sigma^*$ is consistently estimated. This result holds for all three design types D1–D3.

> [!NOTE]
> The approximation error from basis function estimation only needs to be $o(T^{-1/4})$, not $o(T^{-1/2})$ as in classical nonparametric testing, because the estimating equations are idempotent under sieve approximation (similar to projection operators). This removes the need for undersmoothing.

### Theorem 2 (Type I Error Control)

$$P\left(\bigcup_{j=1}^{k} \{Z_j > \hat{b}_j\}\right) \leq \alpha(T_k) + o(1)$$

The bootstrap-generated thresholds asymptotically control the familywise error rate at level $\alpha(T_k)$.

### Theorem 3 (Power)

- Against **fixed alternatives** ($\tau > 0$): power $\to 1$ as $T \to \infty$.
- Against **local alternatives** ($\tau_0 = T^{-1/2} h$ for $h > 0$): power exceeds a non-negligible constant.

---

## Experiments

- **Dataset:** Simulated MDP with 2-dimensional state; real ride-sharing platform data (December 2016, 2-week experiment).
- **Hardware:** Not specified.
- **Optimizer:** Not applicable (estimating equations, not gradient-based).
- **Simulation setup:** 3 interim stages, $T \in \{100, 500\}$, discount $\gamma \in \{0.6, 0.8\}$, polynomial basis of degree $J \in \{3,4,5\}$, effect sizes $\delta \in \{0, 0.05, 0.1, 0.15, 0.2\}$.
- **Competing methods:** classical t-test, DML-based test, V-learning (Luckett et al., 2020), crossover trial adaptation.

**Key results:**

| Method | Power (δ=0.15, D1) | Type-I Error |
|--------|-------------------|--------------|
| Proposed | ~70–80% | ≈0.05 |
| t-test | ~5% | ≈0.05 |
| DML | ~5% | ≈0.05 |
| V-learning | Fails (positivity) | — |

In the real data application (ride-sharing):
- **A/A test:** $p$-value above threshold (sanity check passed).
- **A/B test:** Rejected $H_0$ at an interim stage, confirming the new dispatch strategy significantly improved drivers' income per unit time.

---

## Comparison with Related Methods

| Method | Carryover Effect | Sequential Testing | Positivity Required | Online Update |
|--------|-----------------|-------------------|---------------------|---------------|
| **Proposed** | Yes (via MDP) | Yes (group-sequential) | Only for D1 | Yes |
| Classical t-test | No | Possible | No | Yes |
| DML | Partial | No | No | Partial |
| V-learning | Yes | No | Yes | No |
| Crossover trials | Partial | No | No | No |

> [!IMPORTANT]
> The proposed method does not require an inverse propensity weighting (IPW) step, so it avoids variance explosion from near-extreme propensity scores. This is especially advantageous for D2 (alternating design) where propensities are 0 or 1 and IPW-based methods completely fail.

---

## High-Dimensional and Other Extensions

- **High-dimensional states:** Replace sieve basis with Dantzig selector + decorrelated score for sparse $\beta$.
- **Dynamic (state-dependent) policies:** The value function framework extends to $\pi^{(a)}(s)$ rather than constant $\{0,1\}$ policies.
- **Basis selection:** Cross-validation using prediction loss for TD residuals, analogous to model selection in regression.
- **Unequal interim intervals:** Framework handles varying $T_k - T_{k-1}$ between stages without modification.
