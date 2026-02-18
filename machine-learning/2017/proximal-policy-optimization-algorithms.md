# Meta Information

- URL: [[1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

# Overview

PPO (Proximal Policy Optimization) is an on-policy policy gradient method developed at OpenAI that improves training stability by constraining the step size of policy updates. It alternates between collecting trajectory data by running the current policy in the environment, and optimizing a clipped surrogate objective via minibatch stochastic gradient ascent over multiple epochs. PPO is applicable to any task modeled as an MDP where a policy gradient method is appropriate: continuous control robotics, discrete action game playing, and fine-tuning large language models with human feedback (RLHF).

> [!NOTE]
> "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a 'surrogate' objective function using stochastic gradient ascent."

# Terminologies

| Term | Symbol | Description |
| --- | --- | --- |
| Policy | $\pi_\theta(a \mid s)$ | Parameterized distribution over actions given state $s \in \mathcal{S}$, $a \in \mathcal{A}$ |
| Old policy | $\pi_{\theta_\text{old}}(a \mid s)$ | Policy used for data collection before the gradient update |
| Probability ratio | $r_t(\theta)$ | $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ — how much more likely the new policy takes action $a_t$ vs the old |
| Advantage estimate | $\hat{A}_t \in \mathbb{R}$ | Estimated advantage at time $t$; measures how much better action $a_t$ is than average |
| Return | $R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ | Discounted cumulative reward from time $t$ |
| Value function | $V_\phi(s)$ | Learned baseline; parameterized by $\phi$, outputs scalar estimate of $\mathbb{E}[R_t \mid s_t = s]$ |
| KL divergence | $D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta)$ | Measure of how much the new policy has changed from the old policy |
| Horizon | $T$ | Number of timesteps in a rollout segment |
| Mini-batch size | $M$ | Number of transitions sampled from the rollout buffer per gradient step |
| Epochs | $K$ | Number of passes over the rollout buffer per policy update |

# Background: Policy Gradient Methods

## Vanilla Policy Gradient (REINFORCE)

The basic policy gradient estimator computes the gradient of expected return with respect to policy parameters:

```math
\hat{g} = \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t\right]
```

- Input: a batch of $(s_t, a_t, r_t, s_{t+1})$ tuples
- Output: a gradient estimate $\hat{g}$ used to update $\theta$ via gradient ascent

The corresponding surrogate objective (whose gradient equals $\hat{g}$) is:

```math
L^\text{PG}(\theta) = \mathbb{E}_t\left[\log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t\right]
```

**Problem**: Using this objective with multiple update steps per batch of data leads to destructively large policy updates. The policy can change so drastically that performance collapses.

## Trust Region Policy Optimization (TRPO)

TRPO (Schulman et al., 2015) addresses policy collapse by solving a constrained optimization problem:

```math
\begin{align}
&\max_\theta \; \mathbb{E}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} \hat{A}_t\right] \\
&\text{subject to} \; \mathbb{E}_t\left[D_\text{KL}\left(\pi_{\theta_\text{old}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t)\right)\right] \leq \delta
\end{align}
```

**Problem**: TRPO requires computing second-order derivatives or conjugate gradient steps to enforce the KL constraint, making it computationally expensive and complex to implement.

> [!TIP]
> [Trust Region Policy Optimization (arXiv:1502.05477)](https://arxiv.org/abs/1502.05477)

# PPO: Core Idea

PPO achieves the stability benefits of TRPO using first-order optimization only, by clipping the probability ratio $r_t(\theta)$ so that it cannot deviate too far from 1.0. This removes the need for second-order optimization, constraint handling, or line searches.

## Clipped Surrogate Objective (PPO-Clip)

Define the probability ratio:

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
```

Note: $r_t(\theta_\text{old}) = 1$ by definition.

The clipped surrogate objective is:

```math
L^\text{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat{A}_t\right)\right]
```

where $\epsilon$ is a hyperparameter (typically $0.1$ or $0.2$).

**How the clip works**:

| Case | $\hat{A}_t > 0$ (good action) | $\hat{A}_t < 0$ (bad action) |
| --- | --- | --- |
| $r_t < 1-\epsilon$ | Ratio capped at $1-\epsilon$; gradient pushes to increase $r_t$ | Ratio capped; gradient not pushed lower (loss already pessimistic) |
| $1-\epsilon \le r_t \le 1+\epsilon$ | Normal gradient update | Normal gradient update |
| $r_t > 1+\epsilon$ | Ratio capped at $1+\epsilon$; prevents over-exploitation | Ratio capped; gradient not pushed higher |

The `min` operator ensures the objective is a lower bound (pessimistic estimate) on unclipped performance. When $\hat{A}_t > 0$, the update is capped so the new policy cannot become more than $(1+\epsilon)$ times as likely to take $a_t$. When $\hat{A}_t < 0$, the update is capped so the policy cannot become less than $(1-\epsilon)$ times as likely.

> [!IMPORTANT]
> The clipping acts as a regularizer. It does not prevent the policy from changing — it prevents the surrogate objective from giving credit for large changes. Without clipping, the surrogate objective can grow arbitrarily as $r_t \to \infty$ even if the true performance has already saturated.

## Adaptive KL Penalty (PPO-KL)

An alternative formulation uses an unconstrained objective with an adaptive KL penalty coefficient $\beta$:

```math
L^\text{KLPEN}(\theta) = \mathbb{E}_t\left[r_t(\theta)\hat{A}_t - \beta \cdot D_\text{KL}\left(\pi_{\theta_\text{old}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t)\right)\right]
```

$\beta$ is adjusted after each policy update:
- If $D_\text{KL} < d_\text{targ} / 1.5$: halve $\beta$
- If $D_\text{KL} > d_\text{targ} \times 1.5$: double $\beta$

In practice, PPO-Clip outperforms PPO-KL in most benchmarks.

# Full PPO Algorithm

## Input / Output

- **Input**: environment that provides $(s_t, a_t, r_t, s_{t+1}, \text{done})$ tuples; initial policy parameters $\theta$; initial value function parameters $\phi$
- **Output**: trained policy $\pi_\theta$ that maximizes expected return

## Combined Objective

The full training objective combines the policy surrogate, a value function loss, and an entropy bonus:

```math
L^\text{total}(\theta, \phi) = \mathbb{E}_t\left[L^\text{CLIP}(\theta) - c_1 \cdot L^\text{VF}(\phi) + c_2 \cdot S[\pi_\theta](s_t)\right]
```

where:
- $L^\text{VF}(\phi) = \left(V_\phi(s_t) - V_t^\text{target}\right)^2$ — value function mean squared error
- $S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$ — entropy bonus to encourage exploration
- $c_1, c_2$ — coefficients (hyperparameters)

When the policy network and value network share parameters, all three terms are optimized jointly.

## Pseudocode

```
Initialize policy parameters θ, value parameters φ
for iteration = 1, 2, ... do
    # Collect rollouts
    for actor = 1, ..., N do
        Run π_{θ_old} in environment for T timesteps
        Store transitions (s_t, a_t, r_t, s_{t+1}) in buffer B
    end for

    # Compute advantage estimates from buffer B
    for t = T, T-1, ..., 1 do
        δ_t = r_t + γ * V_φ(s_{t+1}) - V_φ(s_t)         # TD residual
        Â_t = δ_t + (γλ) * δ_{t+1} + ... + (γλ)^{T-t-1} * δ_{T-1}  # GAE
    end for
    Compute returns V_t^target = Â_t + V_φ(s_t)

    # Optimize objective for K epochs with minibatches of size M
    θ_old ← θ
    for epoch = 1, ..., K do
        for minibatch in shuffle(B) do
            Compute r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
            L^CLIP = E_t[min(r_t(θ)*Â_t, clip(r_t(θ), 1-ε, 1+ε)*Â_t)]
            L^VF   = E_t[(V_φ(s_t) - V_t^target)^2]
            L^S    = E_t[entropy of π_θ(·|s_t)]
            L^total = L^CLIP - c1*L^VF + c2*L^S
            Update θ, φ via gradient ascent on L^total
        end for
    end for
end for
```

## Generalized Advantage Estimation (GAE)

PPO uses GAE (Schulman et al., 2016) to estimate advantages with controllable bias-variance trade-off:

```math
\hat{A}_t^\text{GAE}(\gamma, \lambda) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V
```

where $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

- $\lambda = 0$: high bias, low variance (equivalent to one-step TD)
- $\lambda = 1$: low bias, high variance (equivalent to Monte Carlo returns)
- Typical value: $\lambda = 0.95$

> [!TIP]
> [High-Dimensional Continuous Control Using Generalized Advantage Estimation (arXiv:1506.02438)](https://arxiv.org/abs/1506.02438)

# Comparison with Similar Algorithms

| Method | Update Constraint | Complexity | Epochs/Sample | Use Cases |
| --- | --- | --- | --- | --- |
| REINFORCE | None | Low | 1 | Simple tasks with discrete actions |
| Actor-Critic (A3C) | None | Medium | 1 | Continuous control, parallel actors |
| TRPO | Hard KL constraint (2nd order) | High | 1 | When stability is critical; small batches |
| PPO-Clip | Soft clip on ratio (1st order) | Low | $K > 1$ | General purpose; scales to LLMs |
| PPO-KL | Adaptive KL penalty (1st order) | Low | $K > 1$ | When KL monitoring is desired |
| GRPO | Group relative, no value net | Low | $K > 1$ | LLM fine-tuning; memory-efficient |

**Key differences from TRPO**:
1. PPO uses first-order gradients only; TRPO uses conjugate gradient + Fisher information matrix
2. PPO can reuse each rollout for $K$ gradient steps; TRPO is effectively 1 step per rollout
3. PPO's clipping is implicit; TRPO's KL constraint is explicit and enforced exactly
4. PPO is significantly simpler to implement (~50 lines of PyTorch vs. ~500 for TRPO)

**Key differences from vanilla Actor-Critic**:
1. Vanilla A-C performs one gradient step per environment step; PPO performs $K$ epochs over $NT$ timesteps before collecting new data
2. PPO's clipping prevents catastrophic policy degradation that can occur with multiple unconstrained gradient steps in A-C

# Hyperparameters

| Hyperparameter | Typical Value | Role |
| --- | --- | --- |
| Clip ratio $\epsilon$ | 0.1 or 0.2 | Controls maximum policy change per update |
| Discount $\gamma$ | 0.99 | Temporal discount for future rewards |
| GAE parameter $\lambda$ | 0.95 | Bias-variance trade-off in advantage estimation |
| Epochs per update $K$ | 3–10 | Number of gradient passes over each rollout batch |
| Mini-batch size $M$ | 64–4096 | Transitions per gradient step |
| Rollout length $T$ | 128–2048 | Timesteps collected before each update |
| Number of actors $N$ | 1–64 | Parallel environments for data collection |
| Value loss coefficient $c_1$ | 0.5 | Scales value function loss relative to policy loss |
| Entropy coefficient $c_2$ | 0.01 | Encourages exploration by rewarding high-entropy policies |

# Experiments

## Datasets / Benchmarks

- **MuJoCo continuous control**: HalfCheetah-v1, Hopper-v1, Swimmer-v1, Walker2d-v1, Ant-v1, Humanoid-v1 — physics simulation environments with continuous action spaces $a \in \mathbb{R}^d$
- **Atari**: 49 games from the Arcade Learning Environment (ALE) — discrete action spaces, raw pixel inputs $s \in \mathbb{R}^{210 \times 160 \times 3}$ (processed to $84 \times 84 \times 4$ frame stacks)
- **Roboschool**: Additional simulated locomotion tasks

## Key Results

- On MuJoCo benchmarks, PPO-Clip achieves higher final performance and better sample efficiency than TRPO, A2C (synchronous A3C), and CEM (cross-entropy method) on most tasks
- On Atari, PPO matches or exceeds A2C and ACER (sample-efficient actor-critic) on most of the 49 games when trained for 40M frames
- PPO-Clip outperforms PPO-KL consistently, making the clipping variant the recommended default

## Hardware

- 8 parallel actors (workers) collecting data simultaneously
- No GPU requirement mentioned; single-machine CPU training reported

## Optimizer

- Adam optimizer, learning rate $3 \times 10^{-4}$ (annealed to 0 over training in some experiments)

# Applications

PPO is the dominant policy gradient algorithm for:
1. **Robotics simulation**: Training locomotion and manipulation policies in MuJoCo, Isaac Gym, etc.
2. **Game AI**: Training agents on Atari, StarCraft, Dota 2 (OpenAI Five used PPO)
3. **RLHF for LLMs**: PPO is the RL component in InstructGPT, ChatGPT, and subsequent RLHF pipelines — the reward model provides $r_t$, the language model is $\pi_\theta$, and the KL penalty from the reference model is incorporated into the reward
4. **Custom environments**: Any MDP with a reward signal, whether discrete or continuous action space

> [!IMPORTANT]
> In the RLHF context, a KL penalty term $-\beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ is typically added to the per-token reward to prevent the policy from deviating too far from the supervised fine-tuned reference policy. This is separate from the PPO clipping and is applied at the reward level, not the objective level.
