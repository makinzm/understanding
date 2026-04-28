# Meta Information

- URL: [Learning to Orchestrate Agents in Natural Language with the Conductor](https://arxiv.org/abs/2512.04388)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Nielsen, S., Cetin, E., Schwendeman, P., Sun, Q., Xu, J., & Tang, Y. (2025). Learning to Orchestrate Agents in Natural Language with the Conductor. arXiv:2512.04388.

# Overview

The Conductor is a 7B-parameter language model trained via reinforcement learning (RL) to coordinate multiple heterogeneous LLM agents. Rather than using fixed multi-agent topologies or hand-crafted routing rules, the Conductor learns end-to-end to: (1) decompose tasks into focused subtasks, (2) assign appropriate worker agents, and (3) control which agents can observe previous agents' outputs. This approach achieves state-of-the-art results on reasoning benchmarks by automatically discovering effective communication topologies and prompt engineering strategies.

**Applicability**: Useful when a fixed pool of specialized LLMs is available (e.g., a mix of closed-source and open-source models) and one wants to route subtasks to the most capable agent automatically. Suited for developers and researchers building production multi-agent pipelines where manual topology design is impractical.

# Problem Setting

## Limitations of Prior Multi-Agent Approaches

Existing multi-agent frameworks suffer from one or more of the following:
- **Fixed topologies**: Mixture-of-Agents (MoA) and similar methods use predetermined aggregation patterns that cannot adapt per query.
- **No subtask specialization**: Routing methods (e.g., MASRouter, RouterDC) assign entire tasks to one agent without decomposing them into targeted instructions.
- **Inference-scaling methods**: Repeated sampling from a single large model (e.g., majority voting) is expensive and does not leverage agent diversity.

## Goal

Given a task $x$ and a pool of $K$ worker LLMs $\{w_1, \ldots, w_K\}$, train a Conductor $\pi_\theta$ that outputs a **workflow** specifying:

1. Which workers to call and in what order
2. What natural language subtask each worker receives
3. Which previous workers' outputs each worker can observe

# The Conductor Model

## Workflow Representation

The Conductor outputs a structured workflow as three parallel Python lists of length $N$ (number of steps):

```
model_id   = [w_2, w_1, w_3, ...]   # which worker at each step
subtasks   = ["Solve the integral...", "Verify...", ...]  # NL instruction per step
access_list = [[],  [0], [0, 1], ...]  # indices of previous steps visible to each worker
```

- `model_id[i]` $\in \{1, \ldots, K\}$: worker index at step $i$
- `subtasks[i]`: a natural language prompt to worker $i$ — this is the Conductor's "prompt engineering" output
- `access_list[i]` $\subseteq \{0, \ldots, i-1\}$: indices of earlier steps whose outputs are passed to step $i$

The final answer is extracted from the last step's response, or via a final aggregation prompt if `access_list` of the last step contains all previous steps.

## Input/Output Specification

| Component | Specification |
|-----------|--------------|
| Conductor input | Full task description $x$ (string) |
| Conductor output | $(N, \text{model\_id}, \text{subtasks}, \text{access\_list})$ as a Python code block |
| Worker input per step $i$ | $\text{subtasks}[i]$ + concatenated outputs of steps in $\text{access\_list}[i]$ |
| Worker output | Natural language response (string) |
| Final answer | Extracted from last step's response |

# Training with GRPO

## Reward Design

Training uses **Group Relative Policy Optimization (GRPO)**, a policy-gradient variant that does not require a learned value function. For each problem $x$, the Conductor samples $G$ workflow candidates and computes per-candidate rewards:

```math
\begin{align}
r_i = r_{\text{format}}(w_i) + r_{\text{correctness}}(w_i)
\end{align}
```

where:
- $r_{\text{format}}(w_i) = 0$ if the output is not a valid Python list structure; 1 otherwise
- $r_{\text{correctness}}(w_i) = 1$ if the final answer is correct, 0.5 if a partial match occurs (e.g., code compiles but fails tests), 0 otherwise

The GRPO advantage for candidate $i$ within group of size $G$:

```math
\begin{align}
A_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_G)^2}
\end{align}
```

The policy is updated to increase the likelihood of high-advantage workflows via clipped surrogate loss (standard PPO-style clipping).

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Conductor size | 7B parameters |
| Base model | Qwen2.5-7B-Instruct |
| Training problems | 960 (MATH + MMLU + RLPR + LiveCodeBench V1) |
| GRPO iterations | 200 |
| Batch size | 256 |
| Hardware | 2× H100 GPUs |
| KL regularization | Minimal (near-zero coefficient) |

## Training Algorithm (Pseudocode)

```
Initialize conductor π_θ from pretrained 7B LM
For each iteration t = 1, …, 200:
  Sample batch B of problems from training set
  For each problem x in B:
    Sample G=8 workflow candidates from π_θ(· | x)
    Execute each workflow: call workers, collect responses
    Compute rewards r_1, …, r_G (format + correctness)
    Compute advantages A_1, …, A_G using group normalization
  Compute GRPO policy gradient using all (x, workflow, A) triples
  Update θ via gradient ascent with clipping
```

# Worker Pool

The Conductor coordinates seven heterogeneous agents:

| Agent | Type |
|-------|------|
| GPT-5 | Closed-source |
| Claude Sonnet 4 | Closed-source |
| Gemini 2.5 Pro | Closed-source |
| DeepSeek-R1-Distill-Qwen-32B | Open-source (reasoning) |
| Qwen3-32B | Open-source |
| Gemma3-27B | Open-source |
| Conductor (self) | Recursive role |

The Conductor can assign **itself** as a worker (recursive topology), enabling iterative refinement at test time without requiring a larger base model.

# Extensions

## Adaptive Worker Selection

When the full agent pool is unavailable (e.g., budget-constrained inference with only open-source or only proprietary models), the Conductor can be fine-tuned on randomized subsets of the agent pool. This allows the same trained Conductor to adapt to arbitrary worker combinations at deployment time without retraining from scratch.

## Recursive Topologies (Test-Time Scaling)

By including itself in `model_id`, the Conductor creates multi-round refinement pipelines:

```
Round 1: Conductor assigns tasks to workers, aggregates results
Round 2: Conductor (as worker) receives aggregated context, refines
Round N: Final answer extracted
```

This allows **dynamic compute scaling**: allocate more rounds for harder problems without changing the model.

# Emergent Behaviors

Without explicit supervision, the trained Conductor exhibits:

1. **Verification loops**: After a worker produces an answer, the Conductor assigns a second worker to check it.
2. **Role-based specialization**: Coding tasks are routed to coding-specialized agents; math to reasoning models.
3. **Adaptive depth**: Simple problems average ~2 workflow steps; complex coding problems average ~4 steps.
4. **Prompt engineering**: The Conductor generates focused, task-specific subtask descriptions rather than forwarding the original task verbatim — this accounts for a significant portion of performance gains (ablation: removing subtask generation drops LiveCodeBench from 64.29% to 58.62%).

# Experiments

## Datasets

| Dataset | Domain | Split Used |
|---------|--------|-----------|
| MATH | Mathematical reasoning | In-distribution test |
| MMLU | Multi-task language understanding | In-distribution test |
| RLPR | Real-world reasoning | In-distribution test |
| LiveCodeBench V1 | Code generation (training) | In-distribution test |
| LiveCodeBench (current) | Code generation (newer) | Out-of-distribution test |
| AIME 2025 | Advanced math olympiad | Out-of-distribution test |
| GPQA Diamond | Graduate-level science QA | Out-of-distribution test |
| BigCodeBench | Code generation (diverse) | Out-of-distribution test |

## Key Results

| Benchmark | Conductor | Best Baseline | Improvement |
|-----------|-----------|--------------|------------|
| LiveCodeBench | 83.93% | ~81% (GPT-5 solo) | ~3% |
| GPQA Diamond | 87.5% | ~84% (frontier models) | ~3.5% |
| AIME 2025 | 93.3% | ~90% (frontier models) | ~3.3% |
| In-distribution avg. | 75.65% | 71.59% (GPT-5 as coordinator) | ~4% |

## Cost-Efficiency Comparison

| Method | Cost-Adjusted Score |
|--------|-------------------|
| Conductor | 103.49 |
| Repeated single-model sampling (majority vote) | 42.94–66.34 |
| Fixed MoA topology | Lower than Conductor |

The Conductor averages ~3 workflow steps per query, fewer than baselines requiring 4–5 calls.

## Ablation Studies

| Configuration | LiveCodeBench Score |
|--------------|-------------------|
| Full Conductor | 64.29% (in-distribution) |
| No subtask generation (forward original task) | 58.62% |
| GPT-5 as zero-shot coordinator | 59.95%–71.59% avg. |
| 3B Conductor | Similar agent selection; worse prompt quality |

The 3B vs 7B comparison reveals that scaling primarily improves **prompt engineering quality** rather than agent selection strategy.

# Comparison with Related Methods

| Method | Topology | Subtask Specialization | RL Training | Adaptive |
|--------|----------|----------------------|------------|---------|
| MoA (Mixture-of-Agents) | Fixed aggregation | No | No | No |
| MASRouter | Dynamic routing | No | No | No |
| RouterDC | Dynamic routing | No | No | No |
| Smoothie | Weighted ensemble | No | No | No |
| **Conductor** | Dynamic + recursive | Yes (NL prompts) | Yes (GRPO) | Yes |

> [!NOTE]
> The key distinction from routing methods is that the Conductor both *selects* agents and *engineers the prompt* each agent receives. From the paper: "the Conductor learns not only to design targeted communication topologies for effective agent-to-agent collaboration, but also to prompt engineer focused instructions."

> [!IMPORTANT]
> The Conductor's RL training provides essential advantages over using a frontier model (e.g., GPT-5 or Gemini) as a zero-shot coordinator. Zero-shot coordinators consistently underperform the trained Conductor despite having access to the same worker pool.

> [!TIP]
> The recursive topology (Conductor as its own worker) provides a way to scale test-time compute without increasing the Conductor's parameter count — analogous to chain-of-thought repeated sampling but at the multi-agent workflow level.
