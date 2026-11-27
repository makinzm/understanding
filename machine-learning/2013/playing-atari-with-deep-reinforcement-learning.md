# Meta Information

- URL: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602. DeepMind Technologies.

# Playing Atari with Deep Reinforcement Learning

## Overview

This paper introduces Deep Q-Network (DQN), the first deep learning model to learn control policies directly from raw high-dimensional pixel input using reinforcement learning. The model applies a convolutional neural network (CNN) trained with a Q-learning variant, taking raw Atari 2600 screen images as input and outputting action-value estimates. Trained on seven games without any game-specific modifications to the architecture or hyperparameters, DQN outperforms all prior RL methods on all seven games and surpasses human expert performance on three (Breakout, Enduro, and Pong).

**Applicability:** Any domain where an agent must learn sequential decision-making from high-dimensional visual observations without hand-crafted feature engineering—including robotics, video games, and simulation-based control.

## Background

### Markov Decision Process Formulation

The agent interacts with the Atari 2600 emulator in a sequence of actions, observations, and rewards. At each timestep $t$, the agent selects an action $a_t \in \mathcal{A} = \{1, \ldots, K\}$ from the set of legal game actions. The emulator returns the next screen image $x_{t+1} \in \mathbb{R}^{210 \times 160 \times 3}$ (raw RGB pixels) and a scalar reward $r_t$.

Because a single frame is insufficient to infer the full game state (e.g., ball velocity in Pong), the agent's state is a sequence of the last $k$ preprocessed frames stacked together:

$$s_t = (x_{t-k+1}, \ldots, x_t) \in \mathbb{R}^{84 \times 84 \times k}$$

where each frame is preprocessed by converting to grayscale and downsampling to 84×84 pixels. In experiments, $k = 4$.

### Action-Value Function and Q-Learning

The action-value function $Q^*(s, a)$ gives the expected sum of discounted future rewards from state $s$ taking action $a$ and following optimal policy thereafter:

$$Q^*(s, a) = \max_\pi \mathbb{E}\left[\sum_{t'=t}^{\infty} \gamma^{t'-t} r_{t'} \mid s_t = s, a_t = a, \pi \right]$$

where $\gamma \in [0, 1]$ is the discount factor. The optimal $Q^*$ satisfies the Bellman equation:

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

In tabular Q-learning, this is solved iteratively. DQN approximates $Q^*$ with a neural network $Q(s, a; \theta)$ with parameters $\theta$.

## Deep Reinforcement Learning

### Two Core Innovations

**1. Experience Replay**

The agent stores its transitions $(s_t, a_t, r_t, s_{t+1})$ in a replay memory $\mathcal{D}$ of fixed capacity $N$ (1 million frames). At each training step, a random minibatch of transitions is sampled from $\mathcal{D}$ for the gradient update. This decouples consecutive observations, which would be highly correlated if used directly, and allows each transition to be used in multiple updates.

**2. ε-greedy Exploration**

The agent follows an ε-greedy policy: with probability $\varepsilon$ it selects a random action, otherwise it selects $a = \arg\max_{a} Q(s, a; \theta)$. ε is annealed linearly from 1.0 to 0.1 over the first 1 million frames, then held fixed at 0.1.

### Loss Function

At each training step, the parameters $\theta$ are updated by minimizing:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(y - Q(s, a; \theta)\right)^2\right]$$

where the target $y$ is:

$$y = \begin{cases} r & \text{if episode terminates at } s' \\ r + \gamma \max_{a'} Q(s', a'; \theta) & \text{otherwise} \end{cases}$$

The gradient with respect to $\theta$ is:

$$\nabla_\theta L(\theta) = \mathbb{E}\left[\left(y - Q(s, a; \theta)\right) \nabla_\theta Q(s, a; \theta)\right]$$

Note that the target $y$ uses the same network parameters $\theta$ (a fixed-target variant was introduced in the extended DQN paper).

### Algorithm: Deep Q-Learning with Experience Replay

```
Initialize replay memory D with capacity N
Initialize action-value network Q with random weights θ

For each episode:
  Initialize and preprocess first state: s_1 = {x_1}, φ_1 = φ(s_1)
  For t = 1 to T:
    With probability ε: select random action a_t
    Otherwise: a_t = argmax_a Q(φ(s_t), a; θ)

    Execute a_t in emulator, observe reward r_t and image x_{t+1}
    Set s_{t+1} = (s_t, a_t, x_{t+1}), preprocess φ_{t+1} = φ(s_{t+1})
    Store transition (φ_t, a_t, r_t, φ_{t+1}) in D

    Sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D
    Set y_j = r_j                              if episode ends at φ_{j+1}
             r_j + γ * max_{a'} Q(φ_{j+1}, a'; θ)  otherwise

    Perform gradient descent step on (y_j - Q(φ_j, a_j; θ))²
```

## Network Architecture

| Layer | Type | Config | Output Shape |
|---|---|---|---|
| Input | Preprocessed frames | 84×84×4 stacked grayscale | $84 \times 84 \times 4$ |
| Conv1 | Convolutional | 16 filters, 8×8, stride 4, ReLU | $20 \times 20 \times 16$ |
| Conv2 | Convolutional | 32 filters, 4×4, stride 2, ReLU | $9 \times 9 \times 32$ |
| FC1 | Fully connected | 256 units, ReLU | $256$ |
| Output | Linear | One unit per action $|\mathcal{A}|$ | $|\mathcal{A}|$ |

The network takes the stacked frame $\phi \in \mathbb{R}^{84 \times 84 \times 4}$ as input and outputs a scalar $Q$-value estimate for each of the $|\mathcal{A}|$ legal actions simultaneously. This is more efficient than computing $Q(s,a)$ separately for each action.

## Preprocessing

Raw Atari frames at $210 \times 160$ RGB are preprocessed per frame:
1. Convert RGB to grayscale: $x_t \in \mathbb{R}^{210 \times 160}$
2. Downsample to $84 \times 84$ pixels: $\hat{x}_t \in \mathbb{R}^{84 \times 84}$
3. Stack the last $k=4$ processed frames: $\phi_t = (\hat{x}_{t-3}, \hat{x}_{t-2}, \hat{x}_{t-1}, \hat{x}_t) \in \mathbb{R}^{84 \times 84 \times 4}$

Rewards are clipped: positive rewards → $+1$, negative rewards → $-1$, zero unchanged. This normalizes the loss scale across games with different score magnitudes.

## Comparison with Prior Methods

| Method | Feature Input | Function Approx. | Exploration |
|---|---|---|---|
| Sarsa (λ) | Hand-crafted features | Linear | ε-greedy |
| Contingency | Hand-crafted features | Linear | ε-greedy |
| HyperNEAT | Raw pixels | Neural network | Evolutionary |
| **DQN (this work)** | Raw pixels (84×84×4) | CNN + Q-learning | ε-greedy with replay |

Key differences from prior deep RL attempts (e.g., Neural Fitted Q / NFQ):
- **Scales to large state spaces**: CNNs efficiently extract spatial features from raw pixels.
- **Experience replay** breaks temporal correlations that cause instability in online Q-learning with neural networks.
- **End-to-end training**: No separate feature extractor; the CNN and value head are trained jointly.

Compared to TD-Gammon (which successfully used a neural network with TD learning for backgammon), DQN handles a much more complex visual input space and is applied across diverse tasks with a single architecture.

# Experiments

- **Dataset / Environment**: Atari 2600 game emulator (ALE - Arcade Learning Environment). Seven games: Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.
- **Hardware**: Not specified in the paper.
- **Optimizer**: RMSProp with minibatch size 32.
- **Training frames**: 10 million frames per game.
- **Replay memory size**: 1 million most recent transitions.
- **ε schedule**: Annealed from 1.0 to 0.1 over 1 million frames, then fixed at 0.1.
- **Discount factor**: $\gamma = 0.99$.
- **Frame skip**: Agent selects a new action every $k=4$ frames (k=3 for Space Invaders); the action is repeated for skipped frames.
- **Results**: DQN outperforms all prior RL methods (Sarsa, Contingency, HyperNEAT) on all 7 games. Exceeds human expert performance on Breakout (168 vs. 31), Enduro (470 vs. 368), and Pong (20 vs. -3). Significantly lags human performance on Q*bert, Seaquest, and Space Invaders.

> [!NOTE]
> The comparisons with human performance are particularly striking for Breakout, where DQN scored 168 against the human expert's 31—the learned strategy of tunneling through the brick wall is qualitatively different from the human approach.

> [!IMPORTANT]
> All seven games were trained with a single fixed architecture and hyperparameter set. The only game-specific adaptation is the output layer size (matching the number of legal actions) and the frame-skip value for Space Invaders (k=3 instead of k=4).

> [!TIP]
> For the extended version of this work with target networks and evaluation across 49 Atari games, see the 2015 Nature paper: Mnih et al., "Human-level control through deep reinforcement learning."
