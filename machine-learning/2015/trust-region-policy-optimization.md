# Meta Information

- URL: [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). Trust Region Policy Optimization. *Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)*.

---

# Trust Region Policy Optimization (TRPO)

TRPO is a policy gradient algorithm for reinforcement learning that provides a **monotonic improvement guarantee** at each update step. It is applicable to large nonlinear policy classes (e.g., neural networks) for continuous control and discrete action tasks alike. Practitioners use TRPO when they need stable, sample-efficient policy optimization without hand-tuning step sizes.

---

## Background: Policy Optimization Objective

**Input:** A Markov Decision Process $(S, A, P, r, \rho_0, \gamma)$ where:
- $S$: state space
- $A$: action space
- $P(s' \mid s, a)$: transition distribution
- $r(s)$: reward function
- $\rho_0$: initial state distribution
- $\gamma \in [0,1)$: discount factor

**Goal:** Maximize expected discounted return:

$$\eta(\pi) = \mathbb{E}_{s_0, a_0, \ldots}\!\left[\sum_{t=0}^{\infty} \gamma^t r(s_t)\right]$$

where $s_0 \sim \rho_0$, $a_t \sim \pi(\cdot \mid s_t)$, $s_{t+1} \sim P(\cdot \mid s_t, a_t)$.

Define:
- **State-action value:** $Q_\pi(s,a) = \mathbb{E}\!\left[\sum_{t=0}^\infty \gamma^t r(s_t) \mid s_0=s, a_0=a\right]$
- **Value function:** $V_\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[Q_\pi(s,a)\right]$
- **Advantage function:** $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$
- **Discounted state visitation:** $\rho_\pi(s) = \sum_{t=0}^\infty \gamma^t \Pr(s_t = s)$

---

## Theoretical Foundation

### Improvement Bound (Kakade & Langford, extended)

The expected return of a new policy $\tilde\pi$ can be decomposed as:

$$\eta(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\tilde\pi}(s) \sum_a \tilde\pi(a \mid s) A_\pi(s,a)$$

This is exact but intractable because $\rho_{\tilde\pi}$ depends on the new policy. Define the **local surrogate objective** using the *old* policy's visitation instead:

$$L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde\pi(a \mid s) A_\pi(s,a)$$

$L_\pi$ matches $\eta$ to first order: $\nabla_\theta L_{\pi_\theta}(\pi_{\theta'}) \big|_{\theta'=\theta} = \nabla_\theta \eta(\pi_\theta)$.

### Monotonic Improvement Guarantee (Theorem 1)

Let $\alpha = \max_s D_\mathrm{TV}(\pi \,\|\, \tilde\pi)[s]$ (maximum total variation divergence). Then:

$$\eta(\tilde\pi) \geq L_\pi(\tilde\pi) - \frac{4\varepsilon\gamma}{(1-\gamma)^2}\alpha^2$$

where $\varepsilon = \max_{s,a} |A_\pi(s,a)|$. Using $D_\mathrm{TV}(p\|q)^2 \leq D_\mathrm{KL}(p\|q)$, the bound becomes:

$$\eta(\tilde\pi) \geq L_\pi(\tilde\pi) - \frac{4\varepsilon\gamma}{(1-\gamma)^2} D_\mathrm{KL}^\mathrm{max}(\pi, \tilde\pi)$$

where $D_\mathrm{KL}^\mathrm{max}(\pi,\tilde\pi) = \max_s D_\mathrm{KL}(\pi(\cdot\mid s) \| \tilde\pi(\cdot\mid s))$.

> [!IMPORTANT]
> Maximizing the right-hand side at each iteration guarantees $\eta(\pi_{i+1}) \geq \eta(\pi_i)$ — a **monotonic improvement** property.

---

## TRPO Algorithm

### Optimization Problem

Instead of the hard penalty form, TRPO uses a **trust region constraint**:

$$\max_\theta \; L_{\theta_\mathrm{old}}(\theta) \quad \text{subject to} \quad \bar{D}_\mathrm{KL}^{\rho_{\theta_\mathrm{old}}}(\theta_\mathrm{old}, \theta) \leq \delta$$

where $\bar{D}_\mathrm{KL}^\rho(\theta_1, \theta_2) = \mathbb{E}_{s \sim \rho}\!\left[D_\mathrm{KL}(\pi_{\theta_1}(\cdot\mid s) \| \pi_{\theta_2}(\cdot\mid s))\right]$.

> [!NOTE]
> The authors empirically find the constraint form is "more robust" than a penalty form because choosing the Lagrange multiplier adaptively (via $\delta$) avoids sensitivity to its scale.

### Surrogate Objective (Importance-Weighted Form)

The expected advantage is rewritten in importance-sampling form over old-policy trajectories:

$$L_{\theta_\mathrm{old}}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_\mathrm{old}},\; a \sim \pi_{\theta_\mathrm{old}}}\!\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_\mathrm{old}}(a\mid s)} Q_{\theta_\mathrm{old}}(s,a)\right]$$

**Input:** old policy parameters $\theta_\mathrm{old}$, collected trajectories $(s_t, a_t, r_t)$
**Output:** updated policy parameters $\theta$

### Pseudocode

```
Algorithm: TRPO
Input: initial policy parameters θ₀, KL threshold δ
for k = 0, 1, 2, ... do
    1. Collect trajectories {τᵢ} by running π_θₖ in environment
    2. Estimate advantages Â(s,a) using Monte Carlo returns or GAE
    3. Compute gradient g = ∇_θ L_θₖ(θ)|_{θ=θₖ}   [policy gradient]
    4. Compute Fisher-vector product: F·v = ∇²_θ D̄_KL · v  [analytically]
    5. Solve F·x = g using conjugate gradient  → search direction x
    6. Line search along x to find largest step satisfying:
       (a) D̄_KL(θₖ, θₖ + α·x) ≤ δ  (constraint)
       (b) L_θₖ(θₖ + α·x) improves    (objective)
    7. Set θₖ₊₁ = θₖ + α·x
end for
```

### Computational Details: Fisher-Vector Product

The key computational challenge is solving the constrained quadratic subproblem. TRPO avoids explicitly forming the $|\theta|\times|\theta|$ Fisher information matrix $\mathbf{F}$ by using **Fisher-vector products**:

$$\mathbf{F}\mathbf{v} = \nabla_\theta \!\left[(\nabla_\theta \bar{D}_\mathrm{KL})^\top \mathbf{v}\right]$$

This costs one additional backward pass per conjugate gradient step — roughly the same as a standard gradient computation.

> [!IMPORTANT]
> The authors compute the Fisher information matrix **analytically** (integrating over actions) rather than empirically from samples, reducing variance and memory cost.

---

## Data Collection: Two Sampling Schemes

### 1. Single-Path Method

- Execute current policy $\pi_\theta$ to collect full trajectories $\tau = (s_0, a_0, r_0, \ldots, s_T)$
- Estimate $Q$ values via Monte Carlo: $\hat Q(s_t, a_t) = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$
- **Suitable for:** physical systems or environments where state resets are expensive/impossible

### 2. Vine Method

1. Execute the policy to generate "trunk" rollouts, forming a set of candidate states $\mathcal{S}$
2. From each $s \in \mathcal{S}$, branch by sampling $K$ actions $a_1,\ldots,a_K \sim \pi_\theta(\cdot\mid s)$
3. Perform a short rollout (length $m$) from each $(s, a_k)$ using **common random numbers** (shared environment randomness across actions)
4. Estimate $Q(s, a_k)$ from rollout returns

> [!NOTE]
> Common random numbers make rollouts from the same state comparable, dramatically reducing variance of advantage estimates. This is only possible with simulator access (resettable environments).

**Output:** A set of $(s, a, \hat Q(s,a))$ tuples used to form the surrogate objective and KL constraint.

---

## Connection to Related Methods

| Method | Approximation to η | Constraint / Penalty |
|---|---|---|
| **Policy Gradient** | $L_\pi$ with gradient step | None (fixed step size) |
| **Natural Policy Gradient** | $L_\pi$ linear approx | Quadratic KL approximation |
| **Policy Iteration** | $L_\pi$ unconstrained max | None |
| **TRPO** | $L_\pi$ with importance weights | Hard KL trust region $\leq\delta$ |
| **REPS** | KL constraint on $p(s,a)$ marginals | Constrain state-action distribution |

> [!NOTE]
> REPS constrains the **joint** state-action distribution $p(s,a)$ rather than the **conditional** action distribution $\pi(a\mid s)$, and requires a costly inner-loop nonlinear optimization not needed in TRPO.

---

# Experiments

- **Datasets / Environments:**
  - **MuJoCo Locomotion** (3 tasks):
    - Swimmer (10-dimensional state, 2-dimensional action)
    - Hopper (12-dimensional state, 3-dimensional action, underactuated)
    - Walker (18-dimensional state, 6-dimensional action, contact dynamics)
  - **Atari 2600** (7 games): Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders — using raw 128×128 grayscale image input
- **Hardware:** Not specified
- **Policy Architecture (Locomotion):** Gaussian MLP with state-dependent mean; two hidden layers of 100 and 50 units (RBF activations)
- **Policy Architecture (Atari):** Convolutional network (~33,500 parameters) outputting a softmax distribution over actions
- **Optimizer:** Conjugate gradient + backtracking line search (TRPO); compared against natural gradient, CEM, CMA, and deep Q-learning
- **KL Budget:** $\delta = 0.01$

**Key Results:**
- TRPO (vine) solved all three MuJoCo locomotion tasks, outperforming natural gradient, CEM, and CMA across the board
- TRPO (single path) matched vine on simpler tasks and was competitive on harder ones
- On Atari, TRPO achieved "consistent reasonable scores" on 7 games, outperforming the cross-entropy method on most and remaining competitive with Deep Q-learning (though not consistently exceeding it)
- Both sampling variants (single path and vine) performed similarly on Atari, suggesting single-path is sufficient for high-dimensional discrete action spaces

---

## Applicability

- **Who:** RL practitioners who need stable policy optimization for continuous control (robotics, locomotion) or high-dimensional discrete tasks (video games)
- **When:** When training neural network policies where standard policy gradient diverges or requires painful step-size tuning
- **Where:** Any environment with a differentiable policy and episodic or step-based reward signal; vine method additionally requires a resettable simulator

> [!TIP]
> TRPO was later succeeded by **PPO (Proximal Policy Optimization, arXiv:1707.06347)**, which achieves similar monotonic improvement guarantees using a clipped surrogate objective and multiple gradient steps per batch — much simpler to implement and equally effective in practice.
