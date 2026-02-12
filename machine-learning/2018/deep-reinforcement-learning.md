# Meta Information

- URL: [Deep Reinforcement Learning](https://arxiv.org/abs/1810.06339)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, Y. (2018). Deep Reinforcement Learning. arXiv:1810.06339. (Work in progress, Synthesis Lectures in AI and ML, Morgan & Claypool)

---

# Terminologies

| Term | Symbol | Description |
| --- | --- | --- |
| State | $s_t \in \mathcal{S}$ | Representation of the current situation of the agent in the environment at timestep $t$ |
| Action | $a_t \in \mathcal{A}$ | The action chosen by the agent at timestep $t$ |
| Reward | $r_t \in \mathbb{R}$ | Scalar feedback signal indicating the immediate benefit of the agent's action |
| Transition probability | $p(s'\|s,a)$ | Probability of transitioning to $s'$ given current state $s$ and action $a$ |
| Discount factor | $\gamma \in (0,1]$ | Weights the importance of future rewards; $\gamma=0$ is myopic, $\gamma \to 1$ is far-sighted |
| Policy | $\pi(a\|s)$ | Probability distribution over actions given state; maps states to action probabilities |
| State-value function | $v_\pi(s)$ | Expected cumulative discounted reward starting from $s$ and following policy $\pi$ |
| Action-value function | $q_\pi(s,a)$ | Expected cumulative discounted reward starting from $(s,a)$ and thereafter following $\pi$ |
| Advantage function | $A(s,a)$ | $Q(s,a) - V(s)$; measures how much better action $a$ is relative to the average |
| Return | $R_t$ | $\sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$; cumulative discounted reward from timestep $t$ |

---

# 1. Introduction

Deep RL (Deep Reinforcement Learning) combines representation learning via deep neural networks with sequential decision-making via RL algorithms. David Silver's conjecture frames the field: *Artificial Intelligence = Reinforcement Learning + Deep Learning*. The key advantage of deep RL over "shallow" RL is that neural networks automatically extract hierarchical features from raw inputs (pixels, text, audio), eliminating hand-crafted feature engineering.

This survey is aimed at graduate students and researchers who want a broad entry point into deep RL and its applications. It covers six core elements (value function, policy, reward, model, exploration, representation), six mechanisms (attention/memory, unsupervised learning, hierarchical RL, multi-agent RL, relational RL, learning to learn), and twelve application domains.

---

# 2. Background

## 2.1. Markov Decision Process (MDP)

An MDP is defined by the 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **Input (at each step):** Current state $s_t \in \mathcal{S}$
- **Output (at each step):** Action $a_t \in \mathcal{A}$, next state $s_{t+1}$, reward $r_t$

The agent-environment interaction loop:

$$s_0 \xrightarrow{a_0} (r_0, s_1) \xrightarrow{a_1} (r_1, s_2) \xrightarrow{a_2} \cdots$$

The **Markov property** holds when $p(s_{t+1} | s_0, a_0, \ldots, s_t, a_t) = p(s_{t+1} | s_t, a_t)$; the future depends only on the present, not the history.

## 2.2. Value Functions and Bellman Equations

**State-value function:**
$$v_\pi(s) = \mathbb{E}_\pi[R_t \mid s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]$$

**Action-value function:**
$$q_\pi(s,a) = \mathbb{E}_\pi[R_t \mid s_t = s, a_t = a]$$

**Bellman expectation equation** for $v_\pi$:
$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s'|s,a)\left[r(s,a,s') + \gamma v_\pi(s')\right]$$

**Bellman optimality equations:**
$$v_*(s) = \max_{a} \sum_{s'} p(s'|s,a)\left[r + \gamma v_*(s')\right]$$
$$q_*(s,a) = \sum_{s'} p(s'|s,a)\left[r + \gamma \max_{a'} q_*(s',a')\right]$$

The **Bellman operator** $\mathcal{T}^\pi$ is a contraction mapping with fixed point $v_\pi$, guaranteeing convergence of iterative policy evaluation.

## 2.3. Dynamic Programming

DP requires a complete model $p(s'|s,a)$ and performs **full backups** over all successor states.

**Policy Iteration** alternates:
1. **Policy Evaluation:** Iterate $v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)[r + \gamma v_k(s')]$ until convergence
2. **Policy Improvement:** $\pi'(s) = \arg\max_a q_\pi(s,a)$ (greedy update)

**Value Iteration** collapses to a single sweep:
$$v_{k+1}(s) = \max_{a} \sum_{s'} p(s'|s,a)\left[r + \gamma v_k(s')\right]$$

| Method | Model Required | Backup Type | Convergence |
|---|---|---|---|
| Dynamic Programming | Yes (full model) | Full backup over all states | Guaranteed |
| Monte Carlo | No | Complete episode backup | Requires episode completion |
| Temporal Difference | No | One-step bootstrap | Online, incomplete episodes |

## 2.4. Monte Carlo Methods

**Input:** Complete episode $s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T$

MC estimates value by averaging observed returns: $V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]$ where $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$.

**Advantage:** Unbiased; effective in non-Markov environments (does not bootstrap).
**Disadvantage:** High variance; requires complete episodes; cannot handle continuing tasks.

## 2.5. Temporal Difference Learning

**Input:** Single transition $(s_t, a_t, r_t, s_{t+1})$

**TD(0) update:**
$$V(s_t) \leftarrow V(s_t) + \alpha \underbrace{[r_t + \gamma V(s_{t+1}) - V(s_t)]}_{\text{TD error } \delta_t}$$

The **TD error** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the difference between the bootstrapped target and the current estimate.

**SARSA** (on-policy control):
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**Q-learning** (off-policy control):
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

Q-learning uses the greedy max over the next state, making it off-policy (learns $q_*$ independently of the behavior policy).

---

# 3. Value Function: DQN and Extensions

## 3.1. Deep Q-Network (DQN)

**Problem:** Combining off-policy learning, nonlinear function approximation, and bootstrapping creates the **deadly triad** — instability and divergence.

**DQN Solution:** Two stabilization mechanisms applied to Q-learning with a neural network $\hat{q}(s,a;\theta)$:

**Input:** Raw pixels $s \in \mathbb{R}^{H \times W \times C}$ (e.g., $84 \times 84 \times 4$ stacked Atari frames)
**Output:** $Q$-values for all actions $\in \mathbb{R}^{|\mathcal{A}|}$

**Algorithm: DQN**
```
Initialize replay buffer D with capacity N
Initialize online network q̂(s,a;θ) with random weights θ
Initialize target network q̂(s,a;θ⁻) with θ⁻ ← θ
for each episode:
  Observe initial state s₁
  for each step t:
    Choose a_t = argmax_a q̂(s_t,a;θ) with ε-greedy exploration
    Execute a_t, observe r_t and s_{t+1}
    Store (s_t, a_t, r_t, s_{t+1}) in D
    Sample minibatch {(sⱼ, aⱼ, rⱼ, s'ⱼ)} from D
    Set target: yⱼ = rⱼ + γ max_a' q̂(s'ⱼ, a'; θ⁻)
    Update θ by minimizing: L(θ) = E[(yⱼ - q̂(sⱼ,aⱼ;θ))²]
    Every C steps: θ⁻ ← θ  (update target network)
```

**Experience Replay:** Stores transitions in buffer $\mathcal{D}$; uniformly samples minibatches. This breaks temporal correlations between consecutive samples and allows reuse of past experience.

**Target Network:** A frozen copy of $\theta^-$ used only for computing bootstrap targets, updated every $C$ steps. This prevents the target from shifting every step, reducing oscillations.

## 3.2. DQN Extensions

| Variant | Key Improvement |
|---|---|
| Double DQN | Decouples action selection and evaluation: $y = r + \gamma q̂(s', \arg\max_{a'} q̂(s',a';\theta); \theta^-)$ — reduces overestimation bias |
| Prioritized Replay | Samples transitions with probability $\propto |\delta_t|^\alpha$; uses importance-sampling weights $w_i = (N \cdot P(i))^{-\beta}$ to correct bias |
| Dueling DQN | Decomposes $Q(s,a;\theta) = V(s;\theta) + A(s,a;\theta) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\theta)$ — separates state value from action advantage |
| Rainbow | Combines Double DQN + Prioritized Replay + Dueling + Multi-step + Distributional + NoisyNet |
| Distributional RL | Models full return distribution $Z(s,a)$ rather than scalar $Q(s,a) = \mathbb{E}[Z(s,a)]$ |

---

# 4. Policy Gradient Methods

## 4.1. Policy Gradient Theorem

**Input:** State $s \in \mathcal{S}$
**Output:** Action distribution $\pi(a|s;\theta) \in \Delta^{|\mathcal{A}|}$ (probability simplex over actions)

The **policy gradient theorem** gives the gradient of the objective $J(\theta) = v_\pi(s_0)$:
$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi\left[\nabla_\theta \log \pi(a|s;\theta) \cdot Q^\pi(s,a)\right]$$

The term $\nabla_\theta \log \pi(a|s;\theta)$ is the **score function** (likelihood ratio gradient), which does not require knowledge of the environment dynamics.

**REINFORCE Algorithm:**
```
Initialize policy parameters θ
for each episode:
  Generate trajectory τ = (s₀,a₀,r₀, ..., s_T)
  for each timestep t:
    Compute return G_t = Σ_{k=0}^{T-t} γᵏ r_{t+k}
    θ ← θ + α ∇_θ log π(a_t|s_t;θ) (G_t - b(s_t))
```

The **baseline** $b(s_t)$ (commonly $V(s_t)$) reduces variance without introducing bias because $\mathbb{E}[\nabla_\theta \log \pi(a|s;\theta) b(s)] = 0$.

## 4.2. Actor-Critic

Replaces Monte Carlo return $G_t$ with a bootstrapped TD estimate to reduce variance:

**Critic:** Estimates $V(s;\phi)$, updated by TD error $\delta = r + \gamma V(s';\phi) - V(s;\phi)$
**Actor:** Updates $\theta \leftarrow \theta + \alpha \delta \nabla_\theta \log \pi(a|s;\theta)$

**Advantage Actor-Critic (A2C/A3C):** Uses advantage $A(s,a) = Q(s,a) - V(s)$ instead of raw $Q$; reduces variance further. A3C uses asynchronous parallel workers to decorrelate gradient updates.

**Proximal Policy Optimization (PPO):**
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio. Clipping prevents excessively large policy updates.

**Comparison: Value-Based vs. Policy Gradient**

| Property | Value-Based (DQN) | Policy Gradient (PG) |
|---|---|---|
| Action space | Discrete (naturally) | Discrete or continuous |
| Policy | Implicit (greedy) | Explicit (parameterized) |
| Convergence | Can oscillate | Local optimum guarantee |
| Variance | Low | High (reduced with baseline) |
| Sample efficiency | Higher (off-policy) | Lower (typically on-policy) |

---

# 5. Reward Design

Reward shaping modifies the reward to guide learning: $r'(s,a,s') = r(s,a,s') + F(s,a,s')$ where $F$ is the shaping function. **Potential-based shaping** $F(s,a,s') = \gamma \Phi(s') - \Phi(s)$ guarantees that the optimal policy is unchanged.

**Inverse Reinforcement Learning (IRL):** Infers a reward function from expert demonstrations, then derives a policy. Applicable when the reward is unknown but expert behavior is observable (e.g., autonomous driving from human demonstrations).

---

# 6. Model-Based RL

Model-based RL learns a transition model $\hat{p}(s'|s,a)$ and/or reward model $\hat{r}(s,a)$ to generate synthetic experience for planning.

**Dyna-Q Algorithm:**
```
Initialize Q(s,a), model M(s,a)
for each real step:
  Execute a_t, observe r_t, s_{t+1}
  Update Q(s_t,a_t) using real transition (TD update)
  Update model: M(s_t,a_t) ← r_t, s_{t+1}
  Repeat n times:
    Sample random s, a from visited pairs
    r, s' ← M(s,a)  (simulated step)
    Update Q(s,a) using simulated transition
```

**Advantage:** More sample-efficient by reusing real data via simulation.
**Disadvantage:** Model errors compound; "model exploitation" can lead to poor performance if the model is inaccurate.

---

# 7. Exploration vs. Exploitation

| Strategy | Description |
|---|---|
| $\epsilon$-greedy | With probability $\epsilon$ choose random action; otherwise choose $\arg\max_a Q(s,a)$ |
| UCB (Upper Confidence Bound) | $a = \arg\max_a \left[Q(s,a) + c\sqrt{\frac{\ln t}{N(s,a)}}\right]$; balances optimism with visit count |
| Thompson Sampling | Sample $Q$ from posterior distribution; act greedily on the sample |
| Intrinsic Motivation | Add count-based or prediction-error bonus $r_{\text{int}}$ to encourage visiting novel states |
| NoisyNet | Add learnable noise to network weights $\epsilon \sim \mathcal{N}(0, \sigma^2)$; noise is state-dependent exploration |

---

# 8. Representation Learning

Neural networks for deep RL use the following common architectures:

| Architecture | Input | Use Case |
|---|---|---|
| CNN | $H \times W \times C$ pixels | Atari games, visual control |
| RNN/LSTM | Sequential $x_t \in \mathbb{R}^d$ | Partial observability (POMDP) |
| Transformer | Set of tokens | Attention-based memory |

**Convolutional feature extraction** (e.g., DQN): Input $84 \times 84 \times 4$ → Conv → Conv → Conv → FC → $Q$-values $\in \mathbb{R}^{|\mathcal{A}|}$

Distributed representations exploit compositional structure: $2^n$ configurations expressible with $n$ binary features, combating the curse of dimensionality.

---

# 9. Important Mechanisms

## 9.1. Hierarchical RL

Decomposes long-horizon tasks into sub-goals. The **options framework** defines $\langle I, \pi, \beta \rangle$ where $I$ is the initiation set, $\pi$ is the intra-option policy, and $\beta$ is the termination condition. High-level policy selects options; low-level policy executes them.

## 9.2. Multi-Agent RL

Multiple agents interact in a shared environment. Challenges: non-stationarity (other agents' policies change the effective MDP), credit assignment in cooperative settings, and equilibrium computation in competitive settings.

## 9.3. Learning to Learn (Meta-RL)

The goal is to learn a policy that quickly adapts to new tasks using few interactions. MAML (Model-Agnostic Meta-Learning) finds an initial parameter $\theta$ such that a few gradient steps on task-specific data yield good performance.

---

# 10. Landmark Systems

## 10.1. Deep Q-Network (DQN) — Mnih et al., 2015

First demonstration that a single architecture with fixed hyperparameters achieves superhuman performance on 49 Atari games from raw pixels. Key result: 29 of 49 games at or above human level.

## 10.2. AlphaGo — Silver et al., 2016

Combines:
1. Supervised learning on human expert moves (initialization)
2. Self-play policy gradient RL (improvement)
3. Monte Carlo Tree Search (MCTS) for planning at inference time
4. Value network to evaluate board positions

Defeated world champion Lee Sedol 4-1, demonstrating combination of search, learning, and self-play.

## 10.3. AlphaGo Zero / AlphaZero

Removes human data entirely; learns from self-play only. Achieves stronger performance with fewer computational resources. AlphaZero generalizes to Chess and Shogi.

---

# Experiments

- **Datasets / Environments:**
  - Arcade Learning Environment (ALE): 49 Atari 2600 games; input $84 \times 84 \times 4$ grayscale frames; 18 discrete actions
  - MuJoCo: Continuous control tasks (HalfCheetah, Hopper, Ant, Humanoid); state $\in \mathbb{R}^{17\text{–}376}$, action $\in \mathbb{R}^{6\text{–}17}$
  - OpenAI Gym: Standardized RL benchmark suite
  - DeepMind Lab: 3D first-person navigation tasks
  - Board games: Go ($19 \times 19$ grid), Chess, Shogi
- **Hardware:** GPU-based training; A3C uses 16 CPU workers
- **Optimizer:** RMSProp (DQN), Adam (many policy gradient methods)
- **Key Results:**
  - DQN: 8 of 49 Atari games surpassed previous best algorithms; 29 surpassed human-level
  - AlphaGo: First AI to defeat professional human Go player (Fan Hui), then world champion (Lee Sedol)
  - A3C: Matches DQN performance in half the training time using asynchronous CPU workers

---

# Summary: Comparison with Classical RL

| Property | Classical (Tabular) RL | Deep RL |
|---|---|---|
| State representation | Explicit table $Q[s,a]$ | Neural network $\hat{q}(s,a;\theta)$ |
| Scalability | Limited to small, discrete state spaces | Scales to continuous, high-dimensional spaces |
| Feature engineering | Manual | Automatic (end-to-end) |
| Stability | Convergence guarantees exist | Deadly triad requires mitigation (experience replay, target networks) |
| Sample efficiency | Typically more efficient per sample | Typically less efficient; requires millions of steps |

> [!IMPORTANT]
> The **deadly triad** — combining (1) off-policy updates, (2) function approximation, (3) bootstrapping — can cause divergence. DQN addresses this via experience replay (decorrelating samples) and target networks (stabilizing the bootstrap target), but does not fully solve the theoretical problem.

> [!TIP]
> Foundational resources: Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.); David Silver's UCL RL course (10 lectures); Sergey Levine's UC Berkeley CS285 Deep RL course.
