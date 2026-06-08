# Meta Information

- URL: [OASIS: Open Agent Social Interaction Simulations with One Million Agents](https://arxiv.org/abs/2411.11581)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ziyi Yang, Zaibin Zhang, Zirui Zheng, et al. (2024). OASIS: Open Agent Social Interaction Simulations with One Million Agents. arXiv:2411.11581.

# OASIS: Open Agent Social Interaction Simulations with One Million Agents

OASIS is a scalable social media simulator that supports up to one million LLM-based agents interacting on platforms modeled after X (Twitter) and Reddit. It is designed to replicate and study real-world social phenomena—information spreading, group polarization, and herd effects—at a scale beyond what prior simulators supported.

**Who should use this:** Researchers studying computational social science, misinformation, or LLM agent behavior; engineers building agent-based simulation systems.

**When to use:** When studying emergent social dynamics (e.g., opinion polarization, information cascades) using LLM agents as proxies for human users.

## Architecture Overview

The system is composed of five interacting components:

| Component | Role |
|---|---|
| Environment Server | Maintains platform state (users, posts, comments, relations) in a relational database |
| RecSys | Recommends content to agents using platform-specific ranking |
| Agent Module | Issues 21 interaction types via Chain-of-Thought LLM reasoning |
| Time Engine | Schedules agent activation using hourly activity probability vectors |
| Scalable Inferencer | Distributes concurrent agent inference across multiple GPUs |

## Environment Server

The environment server stores platform state in six relational tables: `users`, `posts`, `comments`, `relations`, `traces`, and `recommendations`. It handles dynamic updates—new posts, follows, likes—as the simulation progresses. All agent interactions are logged in `traces`, enabling post-hoc analysis.

**Input**: Action requests from agents (e.g., create post, follow user).
**Output**: Updated platform state and confirmation responses.

## Recommendation System (RecSys)

The RecSys exposes personalized content feeds to each agent. The two platforms use different ranking formulas.

**X (Twitter)**: Each post $p$ is scored for user $u$ using:

```math
\begin{align}
  \text{Score}(p, u) = R \times F \times S
\end{align}
```

where:
- $R = \ln\!\left(\frac{271.8 - (t_{\text{current}} - t_{\text{created}})}{100}\right)$ penalizes older posts (recency)
- $F = \max(1, \log_{1000}(\text{fan\_count} + 1))$ rewards posts from popular users
- $S = \cos(E_p, E_u)$ is the cosine similarity between TwHIN-BERT embeddings of the post and user profile

TwHIN-BERT was trained on 7 billion+ tweets and captures semantic interest alignment between posts and user profiles. Both in-network (followed users) and out-of-network posts are ranked and surfaced.

**Reddit**: Posts are ranked by the hot score:

```math
\begin{align}
  h = \log_{10}(\max(|u - d|, 1)) + \text{sign}(u - d) \cdot \frac{t - t_0}{45000}
\end{align}
```

where $u$ = upvotes, $d$ = downvotes, $t$ = post timestamp (Unix), and $t_0$ is the Reddit epoch. This formula promotes both engagement (vote margin) and recency.

## Agent Module

Each agent maintains a persistent profile (age, MBTI type, profession, interest categories) and acts by querying an LLM with its current context (profile + feed). Agents can perform 21 distinct actions:

- **Content actions**: sign-up, create post, repost, create comment, like/unlike post, dislike/unlike-dislike post, like/unlike comment
- **Social actions**: follow, unfollow, mute, unmute
- **Discovery actions**: search posts, search users

Chain-of-Thought (CoT) reasoning is applied before each action selection, making agent decisions interpretable and reducing purely reactive behavior.

**Input**: Agent profile $\in \mathbb{R}^{d}$ (embedded profile text) + ranked feed of posts $\{p_1, \dots, p_k\}$.
**Output**: Selected action type + optional content (post text, target user ID, etc.).

**Base model**: Llama3-8b-instruct (also tested: Qwen1.5-7B, Internlm2-20b).

## Time Engine

Real social media platforms have non-uniform activity patterns across hours of the day. OASIS models this with a 24-dimensional hourly activity probability vector per agent:

```math
\begin{align}
  P_{ij} = \frac{f_{ij}}{\max_k f_{kj}}
\end{align}
```

where $f_{ij}$ is the number of real posts from agent $i$ in hour $j$, and the denominator normalizes by the peak hour. At each simulation step (= 3 minutes of simulated time), agents activate with probability $P_{ij}$ for the current simulated hour $j$, preventing unrealistic synchronous activation.

## Scalable Inferencer

To support $10^6$ concurrent agents, OASIS uses:

- **Asynchronous message queues**: Each agent sends action requests independently; the server processes them without global synchronization.
- **UUID-based tracing**: Every request is tagged with a unique ID to route responses to the correct agent.
- **GPU resource manager**: Distributes LLM inference across multiple devices.
- **vLLM**: Efficient batched LLM inference to maximize GPU utilization.

**Scalability benchmarks**:
| Agent Count | GPUs (A100) | Time per 3-min timestep |
|---|---|---|
| 10,000 | 2 | ~0.2 hours |
| 100,000 | 5 | ~3 hours |
| 1,000,000 | 27 | ~18 hours |

## User Generation

To scale from hundreds of real users to millions, OASIS uses a scale-free network model preserving power-law degree distributions (observed in real social networks). The algorithm:

1. Seed the network with the 196 real core users (each with 1000+ followers).
2. For each new synthetic user:
   - Sample profile attributes from real distributions: age, MBTI type (16 types), profession (13 categories), interests (9 categories).
   - Follow core users with probability 0.1; follow other ordinary users with a probability proportional to their current follower count (preferential attachment).
3. Repeat until the target population is reached.

The result is a network where ~80% of follow edges connect to core users, consistent with real-world "influencer" dynamics.

## Comparison with Related Work

| System | Max Agents | Platform | Dynamic Interactions | Temporal Modeling |
|---|---|---|---|---|
| SocialSim | ~1,000 | Twitter-like | Limited | No |
| AgentSims | ~100 | Custom | Yes | No |
| **OASIS** | **1,000,000** | X + Reddit | **Yes (21 types)** | **Yes (hourly)** |

Key distinctions from prior work:
- **Scale**: Prior agent-based social simulations rarely exceeded thousands of agents. OASIS reaches one million.
- **Platform fidelity**: OASIS implements real recommendation algorithms (TwHIN-BERT, Reddit hot score) rather than simplified proxies.
- **Temporal realism**: The time engine uses real user activity distributions, not uniform or random scheduling.

# Experiments

- **Dataset (Information Propagation)**: Twitter15 and Twitter16 datasets (Liu et al., 2015; Ma et al., 2016); 198 propagation instances spanning 9 categories (true news, false news, etc.); 100–700 users per instance.
- **Dataset (Group Polarization)**: 196 real users from Twitter propagation data; scaled up to 1 million via synthetic user generation.
- **Dataset (Herd Effect)**: 116,932 real Reddit comments across 7 topics; 21,919 counterfactual content posts; 3,600–10,000 generated Reddit users.
- **Hardware**: Up to 27 × NVIDIA A100 GPUs.
- **LLM**: Llama3-8b-instruct (primary); ablations with Qwen1.5-7B and Internlm2-20b.
- **Results**:
  - Information propagation replicated with ~30% normalized RMSE on scale/breadth metrics; simulated cascades are shallower than real ones.
  - LLM agents exhibit stronger group polarization than humans, especially with uncensored models using intensified language.
  - Agents show stronger herd behavior than humans (particularly for negatively-rated content); larger populations (10K+) enable self-correction through opinion diversity.
  - Misinformation spreads broader than truthful content in 1M-agent simulations, consistent with real-world observations.

> [!NOTE]
> The paper reports that simulated propagation depth is systematically shallower than observed reality, suggesting that current LLM agents underestimate the depth of retweet chains. This is a known limitation.

> [!TIP]
> The OASIS codebase is open-source and built on the CAMEL agent framework: https://github.com/camel-ai/oasis

> [!IMPORTANT]
> Running 1M-agent simulations at full scale requires 27 A100 GPUs and ~18 hours per 3-minute simulated timestep. Smaller-scale experiments (10K agents) are feasible with 2 A100s and 12-minute wall time per step.
