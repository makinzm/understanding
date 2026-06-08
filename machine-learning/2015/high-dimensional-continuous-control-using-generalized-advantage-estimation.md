# Meta Information

- URL: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016.

# High-Dimensional Continuous Control Using Generalized Advantage Estimation

## Overview

This paper addresses two central challenges in policy gradient reinforcement learning: **high sample complexity** and **instability of updates**. The authors introduce **Generalized Advantage Estimation (GAE)**, an exponentially-weighted estimator of the advantage function parameterized by $\lambda \in [0, 1]$ that smoothly interpolates between high-bias/low-variance and low-bias/high-variance estimates. GAE is combined with Trust Region Policy Optimization (TRPO) and a trust-region–based value function optimizer to learn neural network locomotion policies for simulated 3D robots.

**Who uses it / when / where**: Researchers and practitioners working on continuous-action policy gradient RL who need to balance variance reduction (for stable learning) with bias introduction (to avoid requiring a perfect value function). Applicable to any episodic or discounted-reward MDP where a learned value function $V_\phi$ is available as a baseline.

## Background: Policy Gradient Methods

The standard policy gradient objective maximizes expected discounted return:

```math
\begin{align}
  g = \mathbb{E}\!\left[\sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
\end{align}
```

where $\Psi_t$ can be instantiated as several quantities (empirical return, advantage estimate, TD residual, etc.). The key trade-off: using the full return $\sum_{t'=t}^{\infty} \gamma^{t'-t} r_{t'}$ is unbiased but high-variance; using the one-step TD residual $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ is low-variance but biased if $V \neq V^\pi$.

The **advantage function** $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$ measures how much better action $a_t$ is compared to the average action under $\pi$ at state $s_t$. Using $A^\pi$ as $\Psi_t$ yields the minimum-variance unbiased estimator when the exact advantage is known.

## Generalized Advantage Estimation (GAE)

### TD Residuals as Building Blocks

Define the $k$-step TD residual:

```math
\begin{align}
  \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
\end{align}
```

Its expectation given exact $V = V^\pi$ equals the exact advantage: $\mathbb{E}[\delta_t^V] = A^\pi(s_t, a_t)$.

The $k$-step advantage estimator sums $k$ TD residuals:

```math
\begin{align}
  \hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k})
\end{align}
```

- $k=1$: $\hat{A}_t^{(1)} = \delta_t^V$ — lowest variance, highest bias (relies heavily on $V$).
- $k \to \infty$: $\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ — unbiased when $V = V^\pi$, but highest variance.

### GAE Definition

GAE takes an exponentially-weighted average over all $k$-step estimators:

```math
\begin{align}
  \hat{A}_t^{\text{GAE}(\gamma,\lambda)}
  &= (1 - \lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)} \\
  &= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
\end{align}
```

- **Parameters**: $\gamma \in (0,1]$ is the discount factor (controls effective horizon); $\lambda \in [0,1]$ is the GAE decay (controls bias–variance trade-off).
- **Special cases**:
  - $\lambda = 0$: $\hat{A}_t^{\text{GAE}(\gamma,0)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ — one-step TD, lowest variance, highest bias.
  - $\lambda = 1$: $\hat{A}_t^{\text{GAE}(\gamma,1)} = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ — full Monte Carlo return minus baseline, unbiased but highest variance.

> [!NOTE]
> "The main insight is that $\lambda$ controls the trade-off between bias and variance: small $\lambda$ reduces variance at the cost of more bias; large $\lambda$ reduces bias at the cost of more variance."

### Reward Shaping Interpretation

GAE has an elegant reward-shaping interpretation. Define shaped rewards:

```math
\begin{align}
  \tilde{r}_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\end{align}
```

Then the discounted sum of shaped rewards with effective discount $\gamma\lambda$ recovers GAE:

```math
\begin{align}
  \hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \tilde{r}_{t+l}
\end{align}
```

Shaping with $V^\pi$ removes long-range temporal dependencies from the reward signal; the steeper discount $\gamma\lambda$ further suppresses noise from remaining distant transitions. This justifies using intermediate $\lambda < 1$ even when $V = V^\pi$.

### Response Function

The **response function** $\chi(l; s_t, a_t) = \mathbb{E}[r_{t+l} \mid s_t, a_t] - \mathbb{E}[r_{t+l} \mid s_t]$ measures how much action $a_t$ influences reward $l$ steps later. Advantage decomposes as:

```math
\begin{align}
  A^\pi(s_t, a_t) = \sum_{l=0}^{\infty} \gamma^l \chi(l; s_t, a_t)
\end{align}
```

If $\chi$ decays quickly (short-delay credit assignment), then even $\lambda = 0$ (single TD step) provides a low-bias estimate. If $\chi$ decays slowly (long-delay credit), larger $\lambda$ is needed to capture the full advantage.

## Algorithm: Policy Optimization with GAE

### Value Function Optimization via Trust Region

The value function $V_\phi(s)$ is trained to minimize squared error. A trust region constraint prevents overfitting:

```math
\begin{align}
  \min_\phi \sum_{n=1}^{N} \| V_\phi(s_n) - \hat{V}_n \|^2 \quad \text{s.t.} \quad \frac{1}{N}\sum_{n=1}^{N} \frac{(V_\phi(s_n) - V_{\phi_{\text{old}}}(s_n))^2}{2\sigma^2} \leq \epsilon
\end{align}
```

where $\hat{V}_n$ is the empirical discounted return, $\sigma^2 = \frac{1}{N}\sum_n (V_{\phi_{\text{old}}}(s_n) - \hat{V}_n)^2$ is the previous prediction variance. This is solved as a constrained optimization via conjugate gradient (same approach as TRPO for the policy).

### Policy Optimization via TRPO

TRPO updates the policy by maximizing a surrogate objective with a KL divergence constraint:

```math
\begin{align}
  \max_\theta \sum_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t^{\text{GAE}} \quad \text{s.t.} \quad \frac{1}{T} \sum_t D_\text{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \| \pi_\theta(\cdot \mid s_t)) \leq \delta
\end{align}
```

This ensures the new policy does not move too far from the old policy, preventing instabilities.

### Full Algorithm Pseudocode

```
Initialize policy π_θ and value function V_φ
for iteration = 1, 2, ... do
    Collect trajectories {τ_i} using π_{θ_old}
    for each timestep t in trajectories do
        Compute δ_t^V = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
    end
    Compute GAE estimates:
        Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}^V
    Compute target values:
        V̂_t = Σ_{l=0}^{T-t} γ^l r_{t+l}  (empirical return)
    Update V_φ via trust-region constrained least squares:
        min_φ ||V_φ(s_t) - V̂_t||^2   s.t. KL(V_φ, V_{φ_old}) ≤ ε_V
    Update π_θ via TRPO:
        max_θ Σ_t [π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)] Â_t
        s.t.  (1/T) Σ_t KL(π_{θ_old}(·|s_t) || π_θ(·|s_t)) ≤ δ
    θ_old ← θ
end
```

### Input / Output Specification

| Component | Input | Output |
|---|---|---|
| Policy $\pi_\theta$ | $s_t \in \mathbb{R}^{d_s}$ (state vector) | $\mu_t \in \mathbb{R}^{d_a}$, $\sigma_t \in \mathbb{R}^{d_a}$ (Gaussian action parameters) |
| Value function $V_\phi$ | $s_t \in \mathbb{R}^{d_s}$ | $\hat{V}(s_t) \in \mathbb{R}$ (scalar value estimate) |
| GAE estimator | trajectory $\{(s_t, a_t, r_t)\}_{t=0}^{T}$, $\gamma$, $\lambda$ | $\hat{A}_t \in \mathbb{R}$ for each $t$ |

## Network Architecture

For 3D locomotion tasks:
- **Shared body**: Feedforward network with three hidden layers of 100, 50, and 25 units, `tanh` activations.
- **Policy head**: Linear output layer mapping to action mean $\mu \in \mathbb{R}^{d_a}$; a separate log-standard-deviation parameter (not state-dependent) outputs $\log \sigma$.
- **Value head**: A separate scalar linear output head $V_\phi(s) \in \mathbb{R}$.
- Policy and value function networks do **not** share weights.

## Comparison with Similar Methods

| Method | Advantage Estimator | Bias | Variance | Value Function |
|---|---|---|---|---|
| REINFORCE (Monte Carlo) | $\sum_{t'=t}^T \gamma^{t'-t} r_{t'} - b(s_t)$ | 0 (no VF error) | Very high | Optional baseline |
| Actor-Critic (TD) | $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ | High (if $V \neq V^\pi$) | Low | Required |
| $n$-step returns | $\hat{A}_t^{(n)}$ | Medium | Medium | Required |
| **GAE (this work)** | $\sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$ | Tunable via $\lambda$ | Tunable via $\lambda$ | Required |
| Compatible features | $\nabla_\theta \log \pi_\theta \cdot w$ | Depends on parameterization | Depends | Not required |

> [!IMPORTANT]
> Compatible function approximation (Konda & Tsitsiklis) is an orthogonal concept: it ensures the value function lies in the span of policy gradient features to avoid bias. GAE instead parameterizes the temporal extent of advantage estimation and does not constrain the function class of $V$.

> [!TIP]
> GAE generalizes $TD(\lambda)$ from prediction to control: just as $TD(\lambda)$ uses eligibility traces for value prediction, GAE uses the same exponential weighting for advantage estimation in policy optimization.

## Experiments

### Datasets / Environments

All experiments use simulated physics environments (MuJoCo):

| Task | Description | State dim | Action dim |
|---|---|---|---|
| Cart-pole | Classic balancing task | ~4 | 1 |
| 3D biped locomotion | Forward walking gait | ~41 | ~10 |
| 3D quadruped locomotion | Forward walking gait | ~29 | ~8 |
| 3D biped stand-up | Rise from ground to standing | ~41 | ~10 |
| 3D swimmer/hopper/ant (comparison) | Various locomotion | varies | varies |

- **Simulation time**: Bipedal locomotion required experience equivalent to approximately 5.8 days of simulated real time.
- **Hardware**: Not explicitly specified.
- **Optimizer**: TRPO (conjugate gradient + line search) for policy; same solver for value function trust region update.

### Key Results

- Intermediate $\lambda \in [0.92, 0.99]$ consistently outperforms both $\lambda=0$ (too biased) and $\lambda=1$ (too high variance) across all tasks.
- Optimal $\gamma \in [0.96, 0.995]$; smaller $\gamma$ provides implicit variance reduction but truncates the effective horizon.
- Combined TRPO + GAE successfully learns stable 3D bipedal and quadrupedal locomotion gaits from scratch using raw kinematics → joint torques, without manually engineered reward shaping.
- The trust-region value function update is more robust than simple gradient descent for fitting $V_\phi$, especially when the policy changes rapidly between iterations.

## Summary of Contributions

1. **GAE formula**: A principled, single-parameter ($\lambda$) family of advantage estimators bridging Monte Carlo and one-step TD.
2. **Reward shaping link**: Formal connection between GAE and potential-based reward shaping, providing theoretical justification for intermediate $\lambda$.
3. **Trust region VF training**: Robust neural network value function training using the same conjugate gradient solver as TRPO.
4. **Empirical validation**: Demonstrated on challenging high-dimensional 3D locomotion tasks, establishing a practical recipe for continuous control with deep RL.
