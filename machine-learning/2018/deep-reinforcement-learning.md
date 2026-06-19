# Meta Information

- URL: [Deep Reinforcement Learning](https://arxiv.org/abs/1810.06339)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, Y. (2018). Deep Reinforcement Learning. arXiv:1810.06339.

# Overview

This survey covers deep reinforcement learning (deep RL) through six core elements (value function, policy, reward, model, exploration vs. exploitation, representation), six important mechanisms (attention/memory, unsupervised learning, hierarchical RL, multi-agent RL, relational RL, meta-learning), and twelve application domains. It is intended for researchers and practitioners who want a structured, unified view of the field, spanning foundational RL theory, DQN-era breakthroughs, policy gradient methods, and emerging frontiers.

> [!NOTE]
> The paper's central conjecture is: "artificial intelligence ≈ reinforcement learning + deep learning."

# Terminologies

| Term | Symbol | Description |
|------|--------|-------------|
| State / State space | $s_t \in S$ | A representation of the agent's current situation in the environment |
| Action / Action space | $a_t \in A$ | All possible actions the agent can take |
| Reward | $r_t$ | Scalar feedback from the environment indicating immediate benefit |
| State transition probability | $p(s_{t+1} \mid s_t, a_t)$ | Probability of moving to $s_{t+1}$ from $(s_t, a_t)$ |
| Discount factor | $\gamma \in [0, 1)$ | Weights importance of future rewards |
| Policy | $\pi(a_t \mid s_t)$ | Probability distribution over actions given a state |
| State-value function | $v_\pi(s)$ | Expected discounted return from $s$ following $\pi$ |
| Action-value function | $q_\pi(s, a)$ | Expected discounted return from $(s, a)$ following $\pi$ |
| Optimal value functions | $v_*, q_*$ | Maximum over all policies of $v_\pi$, $q_\pi$ |
| Advantage function | $A(s, a)$ | $q(s, a) - v(s)$; measures how much better $a$ is than average |

# 2. Background

## 2.1. Machine Learning Paradigms

Three paradigms:

- **Supervised learning**: labeled dataset, minimize prediction loss
- **Unsupervised learning**: discover structure without labels
- **Reinforcement learning**: agent learns by interacting with an environment to maximize cumulative reward

A learning algorithm generally consists of: a dataset (train / validation / test split), a loss function, an optimization procedure, and a model.

## 2.3. Deep Learning

- **CNNs**: exploit local connectivity and weight sharing for image-like inputs
- **RNNs / LSTMs**: handle sequential dependencies; LSTM gates (input, forget, output) prevent vanishing gradients
- **Backpropagation**: computes $\partial L / \partial \theta$ via chain rule; weights updated by gradient descent $\theta \leftarrow \theta - \alpha \nabla_\theta L$
- Deep networks learn hierarchical feature representations, removing the need for hand-crafted features

## 2.4. Reinforcement Learning

### 2.4.1. Problem Setup

The agent–environment interaction is modeled as a **Markov Decision Process (MDP)** $(S, A, P, R, \gamma)$.

At each timestep $t$: agent observes $s_t$, selects $a_t \sim \pi(\cdot \mid s_t)$, receives $r_t$, and transitions to $s_{t+1}$.

Goal: maximize the discounted return:

```math
\begin{align}
  G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
\end{align}
```

### 2.4.2. Value Functions

```math
\begin{align}
  v_\pi(s) &= \mathbb{E}_\pi\bigl[G_t \mid s_t = s\bigr] \\
  q_\pi(s, a) &= \mathbb{E}_\pi\bigl[G_t \mid s_t = s,\, a_t = a\bigr]
\end{align}
```

**Bellman expectation equations** (used in policy evaluation):

```math
\begin{align}
  v_\pi(s) &= \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_\pi(s')\bigr] \\
  q_\pi(s, a) &= \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s', a')\bigr]
\end{align}
```

**Bellman optimality equations** (define $v_*, q_*$):

```math
\begin{align}
  v_*(s) &= \max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_*(s')\bigr] \\
  q_*(s, a) &= \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma \max_{a'} q_*(s', a')\bigr]
\end{align}
```

### 2.4.3. Exploration vs. Exploitation

Trade-off: explore new actions to find better policies, or exploit known high-reward actions.

- **$\varepsilon$-greedy**: select the greedy action with probability $1 - \varepsilon$, a random action with $\varepsilon$. Simple but effective baseline.
- **UCB**: adds an uncertainty bonus to action-value estimates, naturally decaying exploration over time.
- **Thompson sampling**: samples from a posterior distribution over value functions.

### 2.4.4. Dynamic Programming

Assumes full knowledge of the MDP. Two main algorithms:

**Policy Iteration**:
1. *Policy Evaluation*: iterate $v_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)[r + \gamma v_k(s')]$ until convergence
2. *Policy Improvement*: set $\pi'(s) = \arg\max_a q_\pi(s, a)$

**Value Iteration** (combines both steps):

```math
\begin{align}
  v_{k+1}(s) = \max_{a} \sum_{s'} p(s' \mid s, a)\bigl[r(s, a) + \gamma v_k(s')\bigr]
\end{align}
```

Converges to $v_*$ because the Bellman operator $\mathcal{T}$ is a $\gamma$-contraction.

### 2.4.5. Temporal Difference Learning

TD methods learn from incomplete episodes using bootstrapping.

**TD(0) update** (Algorithm 1):

```math
\begin{align}
  V(s_t) \leftarrow V(s_t) + \alpha \bigl[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\bigr]
\end{align}
```

The bracketed term $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**.

**Comparison with Monte Carlo and DP**:

| Method | Model needed? | Episode needed? | Bootstraps? |
|--------|--------------|-----------------|-------------|
| Dynamic Programming | Yes | No | Yes |
| Monte Carlo | No | Yes (full) | No |
| TD Learning | No | No (incremental) | Yes |

### 2.4.6. SARSA (On-Policy TD Control)

**Algorithm 2**: updates $Q(s, a)$ using the next action $a'$ sampled from the current policy $\pi$:

```math
\begin{align}
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\bigl[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\bigr]
\end{align}
```

*On-policy*: evaluation and improvement use the same $\varepsilon$-greedy policy. Safer for real-world interaction because it accounts for exploratory actions.

### 2.4.7. Q-Learning (Off-Policy TD Control)

**Algorithm 3**: updates $Q(s, a)$ using the *greedy* action in $s_{t+1}$, regardless of the behavior policy:

```math
\begin{align}
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\bigl[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\bigr]
\end{align}
```

*Off-policy*: the behavior policy can explore freely while the target policy remains greedy. Converges to $q_*$ under tabular conditions.

**Key difference from SARSA**: SARSA uses $Q(s_{t+1}, a_{t+1})$ (actual next action); Q-learning uses $\max_{a'} Q(s_{t+1}, a')$ (best possible action).

### 2.4.8. Dyna-Q (Integrated Learning and Planning)

**Algorithm 4**: combines real experience with simulated trajectories from a learned model.

```
Repeat:
  1. Take action a in env, observe (s, a, r, s')
  2. Update Q(s, a) with real transition (Q-learning step)
  3. Update model: Model(s, a) ← (r, s')
  4. Repeat n times:
       Sample random (s, a) from past experience
       r, s' ← Model(s, a)
       Update Q(s, a) using simulated (r, s')
```

Improves sample efficiency by reusing model-generated experience for planning.

# 3. Value Function Methods

## 3.1. Deep Q-Network (DQN)

**Input**: raw pixels $s \in \mathbb{R}^{84 \times 84 \times 4}$ (4 stacked grayscale frames)
**Output**: $Q(s, a; \theta) \in \mathbb{R}^{|A|}$ — one Q-value per discrete action

Two key innovations that stabilize training when combining neural networks with Q-learning:

1. **Experience Replay**: store transitions $(s, a, r, s')$ in a replay buffer of size $N$. Sample random minibatches to break temporal correlations and improve data efficiency.

2. **Target Network**: maintain a separate network $Q(s, a; \theta^-)$ with parameters $\theta^-$ that are copied from $\theta$ every $C$ steps. Prevents the moving-target instability problem.

**Loss function** minimized by SGD:

```math
\begin{align}
  L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Bigl[\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\bigr)^2\Bigr]
\end{align}
```

> [!IMPORTANT]
> Without experience replay and a target network, combining off-policy Q-learning with nonlinear function approximators (neural nets) is known to diverge. DQN was the first approach to demonstrate stable learning at scale on the ALE.

**Deadly triad**: function approximation + bootstrapping + off-policy learning can cause instability. DQN mitigates (but does not fully solve) this.

## 3.2. DQN Extensions

| Variant | Key Contribution |
|---------|-----------------|
| Double DQN | Separates action selection and evaluation to reduce overestimation bias |
| Dueling DQN | Decomposes $Q(s,a) = V(s) + A(s,a)$ with a shared feature encoder |
| Prioritized Experience Replay | Samples transitions with higher TD error more frequently |
| Distributional DQN (C51) | Models full return distribution $Z(s,a)$ instead of its expectation |
| Rainbow | Combines all of the above into a single agent |

# 4. Policy Gradient Methods

## 4.1. Policy Gradient Theorem

Parameterize policy as $\pi(a \mid s; \theta)$. Objective: maximize expected return $J(\theta) = \mathbb{E}_\pi[G_0]$.

**Policy gradient theorem**:

```math
\begin{align}
  \nabla_\theta J(\theta) = \mathbb{E}_\pi\bigl[\nabla_\theta \log \pi(a \mid s; \theta) \cdot Q^\pi(s, a)\bigr]
\end{align}
```

## 4.2. REINFORCE (Algorithm 6)

Monte Carlo policy gradient. Uses the full episode return $G_t$ as an unbiased estimate of $Q^\pi(s_t, a_t)$:

```
Initialize θ
Repeat:
  Generate episode (s0,a0,r1,...,sT) following π(·|s;θ)
  For t = 0 to T-1:
    G ← Σ_{k=t+1}^{T} γ^(k-t-1) r_k
    θ ← θ + α γ^t G ∇θ log π(at|st;θ)
```

**Variance reduction with baseline**: subtract $b(s_t)$ (e.g., $v(s_t)$) from $G_t$:

```math
\begin{align}
  \theta \leftarrow \theta + \alpha \gamma^t (G_t - b(s_t)) \nabla_\theta \log \pi(a_t \mid s_t; \theta)
\end{align}
```

The baseline is unbiased because $\mathbb{E}[\nabla_\theta \log \pi \cdot b(s)] = 0$.

## 4.3. Actor-Critic (Algorithm 7)

Combines a **critic** $v(s; w)$ (value function) and an **actor** $\pi(a \mid s; \theta)$ (policy):

```
Initialize θ, w
Repeat:
  s ← current state
  a ← π(a|s;θ)   # actor selects action
  r, s' ← env.step(a)
  δ ← r + γ v(s';w) - v(s;w)   # TD error (critic signal)
  w ← w + α_w δ ∇_w v(s;w)      # update critic
  θ ← θ + α_θ δ ∇_θ log π(a|s;θ)  # update actor
  s ← s'
```

TD error $\delta$ serves as an estimate of the advantage function $A(s, a)$.

## 4.4. Advanced Policy Gradient Methods

| Method | Key Idea | Difference from Vanilla PG |
|--------|----------|---------------------------|
| TRPO | Constrain KL divergence between old and new policy per update step | Monotone improvement guarantee; no step-size tuning |
| PPO | Clip probability ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$ | Simpler than TRPO; similar empirical performance |
| A3C | Asynchronous parallel workers sharing gradients asynchronously | No replay buffer needed; faster wall-clock time |
| DDPG | Deterministic policy gradient for continuous action spaces | Actor outputs $\mu(s)$ directly; off-policy with replay |

# 5. Model-Based RL

Model-based methods learn a transition model $\hat{p}(s' \mid s, a)$ and/or a reward model $\hat{r}(s, a)$, then use it for planning (e.g., via rollouts or dynamic programming).

**Advantages**: dramatically improved sample efficiency by generating synthetic experience.
**Challenges**: compounding model errors in multi-step rollouts (model bias).

**World models** (Ha & Schmidhuber, 2018): learn a compressed latent representation of the world with an RNN, and train a controller purely in "dream" rollouts.

# 6. Exploration vs. Exploitation (Chapter 7)

| Strategy | Mechanism | Use Case |
|----------|-----------|----------|
| $\varepsilon$-greedy | Random action with prob $\varepsilon$ | Simple baseline |
| UCB | Bonus $\propto \sqrt{\ln t / N(a)}$ | Bandits; limited state spaces |
| Thompson sampling | Sample from posterior over $Q$ | Bayesian RL |
| Count-based exploration | Bonus $\propto 1 / \sqrt{N(s)}$ | Tabular or pseudo-count extensions |
| Intrinsic motivation | Curiosity / prediction error as bonus | Sparse-reward environments |
| RND (Random Network Distillation) | Surprise = error of distilling fixed random network | Hard-exploration games (e.g., Montezuma's Revenge) |

# 7. Advanced Mechanisms (Chapters 9–14)

## 7.1. Attention and Memory

- **Attention**: soft weighting over keys enables focusing on relevant parts of input
- **NTM / DNC**: external memory banks with read/write heads extend LSTM capabilities to long-range dependencies

## 7.2. Hierarchical RL

Learns policies at multiple timescales. Options framework: a high-level policy selects *options* (sub-policies with termination conditions). Enables reuse of skills and tackling long-horizon tasks.

## 7.3. Multi-Agent RL

- **Cooperative**: agents share reward; challenges include credit assignment
- **Competitive**: zero-sum games (e.g., self-play in AlphaGo)
- **Mixed**: general-sum games (e.g., social dilemmas)

## 7.4. Meta-Learning (Learning to Learn)

MAML and related approaches optimize an initialization $\theta$ such that a few gradient steps on a new task lead to good performance. Applied to RL for fast adaptation to new environments.

# Experiments and Benchmarks

- **Dataset / Environments**:
  - Arcade Learning Environment (ALE): 57 Atari 2600 games; discrete control with pixel observations
  - OpenAI Gym: unified API for Atari + MuJoCo continuous control tasks
  - MuJoCo: physics-based robotics benchmarks (HalfCheetah, Hopper, Ant, etc.) with continuous action spaces
  - DeepMind Lab: 3D first-person navigation tasks
  - DeepMind Control Suite: standardized continuous control using MuJoCo

- **Key results cited**:
  - DQN surpassed human-level performance on 29 of 49 Atari games (Mnih et al., 2015)
  - AlphaGo defeated world champion Lee Sedol 4–1 (Silver et al., 2016), combining deep RL with MCTS and self-play
  - Rainbow (combining 6 DQN extensions) achieved state-of-the-art on ALE in 2017
  - PPO became OpenAI's default algorithm for continuous control due to stability and simplicity

# Comparison with Similar Methods

| Dimension | Model-Free | Model-Based |
|-----------|-----------|-------------|
| Sample efficiency | Low | High |
| Asymptotic performance | High (given enough data) | Limited by model accuracy |
| Scalability | Good (no model learning) | Harder for high-dim observations |

| Dimension | On-Policy (SARSA, PPO) | Off-Policy (Q-learning, DDPG) |
|-----------|----------------------|-------------------------------|
| Data reuse | Cannot reuse old data | Can reuse replay buffer data |
| Stability | More stable | Potentially unstable (deadly triad) |
| Sample efficiency | Lower | Higher |
