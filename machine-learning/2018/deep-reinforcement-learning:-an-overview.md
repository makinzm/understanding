# Meta Information

```bibtex
@article{DBLP:journals/corr/abs-1810-06339,
  author       = {Yuxi Li},
  title        = {Deep Reinforcement Learning},
  journal      = {CoRR},
  volume       = {abs/1810.06339},
  year         = {2018},
  url          = {http://arxiv.org/abs/1810.06339},
  eprinttype    = {arXiv},
  eprint       = {1810.06339},
  timestamp    = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1810-06339.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Overview

This markdown describes summary about the paper to remind me of its content.

---

# Terminologies

We have to understand and memorize these terminologies before going into details.

| Term | Overview | Description |
| --- | --- | --- |
| $s_t,S$ | state, state space | A representation of the current situation of the agent in the environment. |
| $a_t,A$ | action, action space | A set of all possible actions the agent can take. |
| $r_t$ | reward | A scalar feedback signal indicating the immediate benefit of the agent's action. |
| $p(s_{t+1}/s_t,a_t)$ | state transition probability | | The probability of transitioning to state $s_{t+1}$ given the current state $s_t$ and action $a_t$. |
| $\gamma$ | discount factor | A factor between 0 and 1 that determines the importance of future rewards. |
| $\pi(a_t/s_t)$ | policy | A mapping from states to a probability distribution over actions. |
| $v_{\pi}(s_t)$ | state-value function | The expected return (cumulative reward) starting from state $s_t$ and following policy $\pi$. |
| $q_{\pi}(s_t,a_t)$ | action-value function | The expected return starting from state $s_t$, taking action $a_t$, and thereafter following policy $\pi$. |
| $v_{*}(s_t)$ | optimal state-value function | The maximum expected return achievable from state $s_t$ over all policies. |
| $q_{*}(s_t,a_t)$ | optimal action-value function | The maximum expected return achievable from state $s_t$ and action $a_t$ over all policies. |

# 1. Introduction

RL (Reinforcement Leanrning) prevails in many fields such as robotics, games, and navigation.

Deep RL is like alchemy without explanation but automatically extract features from raw input.

In this book, we will overview Deep RL from basics to advanced topics.

# 2. Background

ML (Machine Learning) have three paradigms: Supervised Learning, Unsupervised Learning, and Reinforcement Learning.

Algorithms have a dataset, which is divided into training and testing sets, and a loss function, like KL divergence.

## 2.4. Reinforcement Learning

### 2.4.1. Problem Setup

Maximize the cumulative reward $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$.

### 2.4.2. Value Functions -> See Section "Terminologies"

### 2.4.3. Exploration vs. Exploitation
