# Meta Information

- URL: [A Survey on Contextual Multi-armed Bandits](https://arxiv.org/abs/1508.03326)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhou, L. (2015). A Survey on Contextual Multi-armed Bandits. arXiv:1508.03326.

# A Survey on Contextual Multi-armed Bandits

## Overview

Contextual multi-armed bandits (CMB) formalize sequential decision-making under uncertainty, where an agent observes **contextual side information** before selecting an action and then receives a partial reward signal. Unlike supervised learning, feedback is only observed for the chosen action, not for unchosen alternatives — this is known as the **bandit feedback** problem.

The survey is aimed at researchers and practitioners who need to:
- Personalize recommendations (e.g., news articles, ads, products)
- Run adaptive clinical trials
- Optimize online advertising bidding

### Decision-Making Taxonomy

| Setting | Outcome Model Unknown | Outcome Model Known |
|---|---|---|
| **Stochastic** | Contextual Bandits (survey focus) | Supervised learning |
| **Adversarial** | EXP4 family | Full information games |

The contextual bandit assumption is that "actions taken won't change the state of the world" — i.e., there is no temporal dependency between rounds (unlike reinforcement learning).

## Problem Formulation

**Notation:**

- $\mathcal{X}$: context space
- $\mathcal{A} = \{1, \ldots, K\}$: arm (action) set, $K$ arms total
- $\mathcal{H}$: policy/hypothesis space (mapping contexts to arms)
- $T$: time horizon (total number of rounds)

**Protocol for one round $t$:**

1. Environment reveals context $x_t \in \mathcal{X}$
2. Agent selects arm $a_t \in \mathcal{A}$ according to policy $h_t \in \mathcal{H}$
3. Agent observes reward $r_{t, a_t} \in [0, 1]$ only for the chosen arm

**Expected reward of policy $h$:**

$$R(h) = \mathbb{E}_{x, r}[r_{h(x)}]$$

**Empirical estimate:**

$$\hat{R}(h) = \frac{1}{T} \sum_{t=1}^{T} \hat{r}_{t, h(x_t)}$$

**Cumulative regret** measures the gap between the optimal policy $h^*$ and the agent's choices:

$$\text{Regret}(T) = T \cdot R(h^*) - \sum_{t=1}^{T} r_{t, a_t}$$

The goal is to minimize regret sub-linearly in $T$.

## Unbiased Reward Estimation

Since only the chosen arm's reward is observed, naive empirical estimates are biased. The **Inverse Propensity Scoring (IPS)** technique produces unbiased estimates:

$$\hat{r}_{t, a} = \frac{r_{t, a'} \cdot \mathbb{1}[a' = a]}{p_a}$$

where $a'$ is the arm chosen by the logging policy and $p_a$ is the probability that arm $a$ was chosen. This converts partial feedback into an unbiased estimator usable for offline policy evaluation.

> [!NOTE]
> IPS requires that the logging policy had non-zero probability of selecting every arm ($p_a > 0$), otherwise certain arms cannot be evaluated offline.

## Stochastic Contextual Bandits

In the stochastic setting, rewards are drawn i.i.d. from an unknown distribution parameterized by the context.

### Linear Payoff Model

Assumes $\mathbb{E}[r_{t,a} | x_{t,a}] = x_{t,a}^\top \theta_a^*$ where $x_{t,a} \in \mathbb{R}^d$ is the feature vector for arm $a$ at time $t$, and $\theta_a^* \in \mathbb{R}^d$ is an unknown weight vector.

**Input:** Feature vector $x_{t,a} \in \mathbb{R}^d$ per arm.
**Output:** Selected arm $a_t$, received reward $r_t \in [0,1]$.

#### LinUCB Algorithm

Uses ridge regression to estimate $\theta_a$ and constructs an upper confidence bound (UCB):

```
For each round t = 1, ..., T:
  Observe context x_{t,a} for each arm a
  For each arm a:
    Compute A_a = D_a^T D_a + I_d        # ridge regression matrix (d×d)
    Compute b_a = D_a^T r_a              # reward vector (d×1)
    Compute θ̂_a = A_a^{-1} b_a          # parameter estimate (d×1)
    Compute UCB_a = x_{t,a}^T θ̂_a + α √(x_{t,a}^T A_a^{-1} x_{t,a})
  Select a_t = argmax_a UCB_a
  Observe r_{t,a_t}, update D_a, b_a
```

- $D_a \in \mathbb{R}^{m \times d}$: design matrix of past contexts for arm $a$ ($m$ past observations)
- $\alpha > 0$: exploration parameter (width of confidence ellipsoid)

**Regret bound:** $O(d\sqrt{T} \ln((1+T)/\delta))$ with probability $1-\delta$.

#### SupLinUCB

A variant that partitions time into epochs and maintains a shared feature space across arms:

$$\hat{\theta} = (D^\top D + I_d)^{-1} D^\top r \in \mathbb{R}^d$$

Achieves $\tilde{O}(\sqrt{dT})$ regret, matching minimax optimal rates.

#### LinREL / SupLinREL

Instead of ridge regression, uses **eigenvalue truncation** of the covariance matrix to regularize:

$$A_{\text{trunc}} = \sum_{i: \sigma_i \geq \tau} \sigma_i u_i u_i^\top$$

where $\sigma_i, u_i$ are eigenvalue-eigenvector pairs. Truncating small eigenvalues removes directions with high uncertainty. Achieves similar regret as LinUCB.

#### Thompson Sampling (Linear)

A Bayesian approach. Maintains a Gaussian posterior over $\theta$:

$$\theta | D \sim \mathcal{N}(\hat{\theta}, \sigma^2 A^{-1})$$

At each round:
1. Sample $\tilde{\theta} \sim \mathcal{N}(\hat{\theta}, \sigma^2 A^{-1})$
2. Select $a_t = \arg\max_a x_{t,a}^\top \tilde{\theta}$

**Regret bound:** $\tilde{O}(d\sqrt{T})$ with high probability.

> [!TIP]
> Thompson Sampling is often preferred in practice due to lower regret empirically and easy implementation, despite having slightly weaker theoretical guarantees than UCB.

### Kernelized Methods

When the linear assumption is too restrictive, kernel methods extend bandits to non-linear reward functions via the **kernel trick** $k(x, x') = \phi(x)^\top \phi(x')$.

#### GP-UCB (Gaussian Process UCB)

Places a Gaussian process prior over reward functions $f \sim \mathcal{GP}(0, k)$:

- Posterior mean: $\mu_t(x) = k_t(x)^\top (K_t + \sigma^2 I)^{-1} r_t$
- Posterior variance: $\sigma_t^2(x) = k(x,x) - k_t(x)^\top (K_t + \sigma^2 I)^{-1} k_t(x)$

**UCB selection:** $a_t = \arg\max_a \mu_{t-1}(x_{t,a}) + \beta_t^{1/2} \sigma_{t-1}(x_{t,a})$

- $k_t(x) \in \mathbb{R}^t$: kernel vector between $x$ and all past contexts
- $K_t \in \mathbb{R}^{t \times t}$: kernel matrix of past contexts

**Regret bound:** $O(\sqrt{T \gamma_T})$ where $\gamma_T$ is the maximum information gain of the kernel.

#### CGP-UCB

Extension of GP-UCB that uses **contextual features** to share information across arms. The context $(x, a)$ pairs encode both user features and arm identity.

#### KernelUCB

A frequentist alternative using kernelized ridge regression instead of GP priors:

$$\hat{f}(x) = k_t(x)^\top (K_t + \lambda I)^{-1} r_t$$

Confidence bounds are derived from Reproducing Kernel Hilbert Space (RKHS) norms.

### Arbitrary Policy Sets

When the reward function is unknown and non-parametric, algorithms operate directly over a finite policy class $\mathcal{H}$ of size $N$.

#### Epoch-Greedy

Separates exploration (uniform random arm selection) and exploitation (greedy policy):

```
Partition T rounds into epochs of increasing length
For each epoch:
  Exploration phase:
    Collect data by selecting arms uniformly at random
    Use IPS estimator to compute R̂(h) for all h ∈ H
  Exploitation phase:
    Select ĥ = argmax_h R̂(h)
    Play ĥ for the remaining rounds in epoch
```

**Regret bound:** $O((K \ln(N/\delta))^{1/3} T^{2/3})$

> [!NOTE]
> Epoch-Greedy reduces the bandit problem to a weighted classification task, enabling use of any off-the-shelf classifier as the oracle.

#### RandomizedUCB

Uses a **non-uniform distribution** over policies to balance exploration and exploitation:

$$\pi_t(h) \propto \exp\left(\eta \hat{R}_t(h) - \text{UCB penalty}\right)$$

Achieves $\tilde{O}(\sqrt{KT \ln N})$ regret with polylogarithmic runtime per round.

#### ILOVETOCONBANDITS

Extension that handles continuous policy classes by maintaining a version space of consistent hypotheses. Achieves $\tilde{O}(\sqrt{T})$ regret with access to an optimization oracle.

## Adversarial Contextual Bandits

In adversarial settings, rewards can be chosen by an adversary to maximize regret. There is no distributional assumption.

### EXP4 Algorithm

Maintains a distribution over $N$ experts (policies):

```
Initialize: w_1(h) = 1/N for all h ∈ H
For t = 1, ..., T:
  Observe context x_t
  Compute mixture policy: π_t(a) = Σ_h w_t(h) h(a | x_t)
  Sample arm a_t ~ π_t
  Observe reward r_{t,a_t}
  Compute IPS estimate: r̂_{t,a} = r_{t,a_t} · 1[a_t = a] / π_t(a)
  Update: w_{t+1}(h) ∝ w_t(h) · exp(η Σ_a h(a|x_t) r̂_{t,a})
```

**Regret bound:** $O(\sqrt{TK \ln N})$ against any fixed policy.

### EXP4.P

High-probability variant of EXP4 with improved confidence bounds:

$$\text{Regret} \leq O\left(\sqrt{TK \ln(N/\delta)}\right) \text{ with probability } 1-\delta$$

Uses additional regularization to stabilize the exponential weight updates.

> [!IMPORTANT]
> EXP4 requires knowing the logging policy probabilities $\pi_t(a)$ to form unbiased IPS estimates. When policies are deterministic, additional randomization (mixing with uniform) is required.

## Comparison of Algorithms

| Algorithm | Reward Model | Regret Bound | Handles Adversarial | Infinite Policies |
|---|---|---|---|---|
| LinUCB | Linear | $O(d\sqrt{T} \ln T)$ | No | N/A |
| SupLinUCB | Linear | $\tilde{O}(\sqrt{dT})$ | No | N/A |
| Thompson Sampling | Linear (Bayesian) | $\tilde{O}(d\sqrt{T})$ | No | N/A |
| GP-UCB | Kernelized | $O(\sqrt{T\gamma_T})$ | No | N/A |
| Epoch-Greedy | Arbitrary | $O(T^{2/3})$ | No | No |
| RandomizedUCB | Arbitrary | $\tilde{O}(\sqrt{KT \ln N})$ | No | No |
| ILOVETOCONBANDITS | Arbitrary | $\tilde{O}(\sqrt{T})$ | No | Yes |
| EXP4 | Arbitrary | $O(\sqrt{TK \ln N})$ | Yes | No |
| EXP4.P | Arbitrary | $O(\sqrt{TK \ln(N/\delta)})$ | Yes | No |

### Key Differences from Standard Multi-armed Bandits

| Property | Standard MAB | Contextual MAB |
|---|---|---|
| Side information | None | Context $x_t$ observed each round |
| Optimal policy | Fixed arm | Mapping $h: \mathcal{X} \to \mathcal{A}$ |
| Reward model | Per-arm distribution | Context-dependent distribution |
| Sample complexity | $O(K)$ per round | $O(d)$ or $O(N)$ per round |

### Key Differences from Reinforcement Learning

| Property | Contextual Bandits | RL |
|---|---|---|
| State transitions | None (actions don't change world state) | Yes |
| Temporal credit assignment | Single-step | Multi-step |
| Complexity | Lower | Higher |

## Practical Applications

- **News personalization**: Arms = articles, contexts = user/article features
- **Online advertising**: Arms = ads, contexts = user demographics + page content
- **Clinical trials**: Arms = treatments, contexts = patient covariates
- **Search ranking**: Arms = ranked lists, contexts = query features

The linear payoff model (LinUCB) is most commonly deployed in industry due to its computational efficiency and interpretability, while GP-UCB methods are suited for smaller-scale problems with highly non-linear reward functions.

# Experiments

- Dataset: The survey is theoretical; no dedicated empirical datasets are introduced. Practical applications use web-scale datasets (e.g., Yahoo! news click-through data mentioned in citations)
- Hardware: Not mentioned (survey paper)
- Results: Regret bounds are the primary evaluation metric; see the algorithm comparison table above
