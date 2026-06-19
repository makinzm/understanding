# Meta Information

- URL: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

# Overview

DeepSeek-R1 presents a methodology to train large language models (LLMs) to develop strong mathematical, coding, and logical reasoning capabilities through reinforcement learning (RL). The paper introduces two primary models:

1. **DeepSeek-R1-Zero**: Trained via pure RL on DeepSeek-V3-Base, with no supervised fine-tuning (SFT). Demonstrates that reasoning capabilities emerge from RL alone.
2. **DeepSeek-R1**: A multi-stage pipeline that combines cold-start SFT data with RL training, achieving performance comparable to OpenAI o1-1217 on numerous benchmarks.

**Applicability**: This approach is applicable for organizations building reasoning-capable LLMs using RL, particularly where obtaining large labeled reasoning datasets is infeasible. The distilled models (1.5B–70B parameters) are useful when deploying reasoning models under resource constraints.

# Background: Limitations of Prior Approaches

Before DeepSeek-R1, several RL-based reasoning alternatives were considered but rejected:

| Approach | Reason for Rejection |
|---|---|
| Process Reward Models (PRM) | Hard to define fine-grained step correctness; prone to reward hacking; difficult to retrain |
| Monte Carlo Tree Search (MCTS) | Token generation space is exponentially larger than game search spaces; iterative refinement impractical |
| Neural Reward Models | Risk of reward hacking at scale; requires periodic retraining |

The paper instead adopts a **rule-based reward system** with GRPO (Group Relative Policy Optimization) as the core RL algorithm.

# DeepSeek-R1-Zero: Pure RL without SFT

## Training Setup

- **Base model**: DeepSeek-V3-Base
- **Algorithm**: GRPO (see below)
- **No supervised fine-tuning data** used before RL

## Prompt Template

The template instructs the model to output reasoning inside `<think>` tags and the final answer inside `<answer>` tags:

```
A conversation between User and Assistant. The user asks a question,
and the Assistant solves it. The assistant first thinks about the
reasoning process in the mind then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think>
and <answer> </answer> tags respectively.

User: {question}
Assistant:
```

> [!NOTE]
> The template deliberately avoids content-specific biases so the model's natural reasoning behavior can be observed during training, without steering it toward any particular reasoning pattern.

## Reward Functions

Two types of rule-based rewards are used:

**1. Accuracy Reward**: Verifies correctness of the final answer.
- For math: checks if the boxed answer matches the ground truth.
- For code: executes generated code against test cases.

**2. Format Reward**: Enforces the `<think>...</think>` structure to ensure reasoning is separated from the final answer.

> [!IMPORTANT]
> No neural reward model is used. This avoids reward hacking and removes the need for periodically retraining a reward model.

## GRPO Algorithm

Group Relative Policy Optimization (GRPO) is used instead of PPO to reduce memory and computational cost by eliminating the separate critic/value model.

**Step 1 – Sample group outputs**: For each question $q$, sample $G$ outputs $\{o_1, o_2, \ldots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$.

**Step 2 – Compute rewards**: Obtain reward scores $\{r_1, r_2, \ldots, r_G\}$ for each output.

**Step 3 – Compute advantages** using group-normalized baseline:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

**Step 4 – Policy gradient update** with clipped ratio and KL penalty:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

Where:
- $\pi_\theta$: current policy (the LLM being trained), outputs token sequences
- $\pi_{\theta_{\text{old}}}$: old policy (frozen snapshot used to generate samples), same shape
- $\pi_{\text{ref}}$: reference policy (original SFT model), used to regularize drift
- $\epsilon$: clipping threshold (PPO-style)
- $\beta$: KL penalty coefficient
- $\hat{A}_i \in \mathbb{R}$: advantage of output $o_i$ relative to the group

**Input**: Question $q$ (token sequence of arbitrary length)
**Output**: Reasoning trace + answer (token sequence, typically 1K–32K tokens)

> [!TIP]
> GRPO vs PPO: PPO requires training a value network (critic) of the same size as the policy, which doubles GPU memory usage. GRPO replaces the critic with a group-based reward average, making it far more memory efficient for large models.

## Emergent Behaviors ("Aha Moment")

During training, without explicit instruction, DeepSeek-R1-Zero spontaneously develops:
- **Reflection**: Re-evaluating its own intermediate reasoning steps.
- **Extended chain-of-thought**: Producing longer, more thorough reasoning traces as training progresses.
- **Self-verification**: Checking whether earlier conclusions are consistent before committing to an answer.

> [!NOTE]
> An intermediate checkpoint was observed generating: "Wait, I need to reconsider this step..." — anthropomorphic self-correction behavior that emerged purely from reward signals, without any template instructing this behavior.

## Drawbacks of R1-Zero

- **Language mixing**: Mixes languages within a single response (e.g., Chinese within an English reasoning trace).
- **Readability**: Reasoning traces can be disorganized and hard to follow.

# DeepSeek-R1: Multi-Stage Training Pipeline

To address R1-Zero's limitations, DeepSeek-R1 uses a four-stage training pipeline.

## Stage 1 – Cold-Start SFT

**Data**: Thousands of long chain-of-thought (CoT) examples collected via:
- Few-shot prompting with detailed CoT examples
- Direct prompting asking for reflection/verification
- Human annotation refinement

**Format**: `|<bos>|<reasoning_process>|<eos>|<summary>|<eos>`

The cold-start data establishes a readable, structured reasoning format before RL begins.

**Input**: Raw training questions paired with long CoT reasoning traces
**Output**: A fine-tuned checkpoint with improved reasoning structure and language consistency

## Stage 2 – Reasoning-Oriented RL

Same GRPO setup as DeepSeek-R1-Zero, but with an additional **language consistency reward** to penalize language mixing.

The total reward is:

$$
r_{\text{total}} = r_{\text{accuracy}} + r_{\text{format}} + r_{\text{language consistency}}
$$

**Input**: Math/code/logic problems
**Output**: A checkpoint strong in structured reasoning tasks

## Stage 3 – Rejection Sampling + SFT

A two-dataset SFT checkpoint is generated:

| Type | Source | Size |
|---|---|---|
| Reasoning samples | Rejection sampling from Stage 2 checkpoint | ~600K |
| Non-reasoning samples | DeepSeek-V3 SFT data (writing, QA, etc.) | ~200K |

**Rejection sampling**: For each problem, generate multiple rollouts and keep only those where the answer is verified correct. This improves data quality without additional human labeling.

**Input**: Mixed reasoning + general dataset
**Output**: SFT checkpoint combining reasoning and general language capabilities

## Stage 4 – Secondary RL on All Scenarios

RL is applied again using a mixed reward signal across:
- Math and code (rule-based accuracy rewards)
- General tasks (helpfulness/harmlessness preference rewards from human feedback)

**Input**: Broad distribution of tasks (reasoning + non-reasoning)
**Output**: DeepSeek-R1 final model

## Pipeline Summary

```
DeepSeek-V3-Base
      │
      ▼
[Stage 1] Cold-Start SFT (small CoT dataset)
      │
      ▼
[Stage 2] Reasoning RL (GRPO + language consistency reward)
      │
      ▼
[Stage 3] Rejection Sampling SFT (600K reasoning + 200K general)
      │
      ▼
[Stage 4] General RL (reasoning + helpfulness/harmlessness)
      │
      ▼
DeepSeek-R1
```

# Knowledge Distillation to Smaller Models

**Goal**: Transfer DeepSeek-R1's reasoning ability into smaller models (1.5B–70B parameters).

**Method**: Supervised fine-tuning of open-source base models (Qwen2.5, Llama3) on 800K samples curated from DeepSeek-R1.

**No RL is applied** during distillation — SFT alone is sufficient to transfer reasoning patterns.

**Input**: 800K reasoning traces generated by DeepSeek-R1
**Output**: Fine-tuned small models with significantly improved reasoning

> [!IMPORTANT]
> Distillation from a 671B model to a 14B model outperforms training a 32B model from scratch with RL (QwQ-32B-Preview). This suggests teacher model quality matters more than small-model RL training effort.

# Experiments

## Datasets Used

| Benchmark | Domain | Description |
|---|---|---|
| AIME 2024 | Math | American Invitational Mathematics Examination (competition-level problems) |
| MATH-500 | Math | 500 diverse math problems across difficulty levels |
| GPQA Diamond | Science | Graduate-level expert questions in biology, chemistry, physics |
| MMLU | General Knowledge | 57 academic subjects, multiple-choice |
| LiveCodeBench | Code | Coding problems from Aug 2024 – Jan 2025 |
| Codeforces | Code | 10 Division 2 competitive programming contests |
| SimpleQA | Factual QA | Short factual questions with verifiable answers |
| Chinese Benchmarks | Chinese language | CNMO 2024, CMATH |

**Training data**:
- Cold-start: ~thousands of CoT examples (exact count not disclosed)
- Stage 3 SFT: ~600K reasoning samples + ~200K general samples
- Distillation: ~800K samples from DeepSeek-R1

## Key Results

- **AIME 2024**: DeepSeek-R1 achieves 79.8% pass@1 (majority voting: 71.0% → 79.8%), surpassing OpenAI o1-1217.
- **MATH-500**: 97.3%, matching OpenAI o1-1217.
- **Codeforces**: 2,029 Elo rating, outperforming 96.3% of human competitors.
- **GPQA Diamond**: 71.5%, on par with OpenAI o1-1217.
- **Distilled 7B model**: 55.5% on AIME 2024, exceeding QwQ-32B-Preview (50.0%).
- **Distilled 14B model**: Outperforms QwQ-32B-Preview by a large margin across benchmarks.

## Hardware

Not explicitly disclosed; DeepSeek-V3-Base is a 671B MoE model requiring large-scale GPU clusters.

# Comparison with Similar Approaches

| Aspect | DeepSeek-R1 | OpenAI o1 | QwQ-32B-Preview |
|---|---|---|---|
| Training method | GRPO RL + multi-stage SFT | Not disclosed | RL-based (details not public) |
| Model size | 671B MoE (active: ~37B) | Not disclosed | 32B dense |
| Open-source | Yes (weights + paper) | No | Yes (weights only) |
| Distillation | Yes (1.5B–70B released) | No | No |
| Reward model | Rule-based only | Not disclosed | Not disclosed |

**vs. PPO**: GRPO eliminates the critic model, halving approximate memory requirements while achieving comparable policy learning quality through group-based advantage estimation.

**vs. RLHF with Neural Reward**: Pure rule-based rewards avoid the reward hacking risk inherent in neural reward models, which can be exploited by the policy at scale.

# Limitations

- **Language mixing**: R1-Zero mixes languages; R1 mitigates but does not fully eliminate this.
- **Few-shot prompting degrades performance**: Unlike typical LLMs, adding few-shot examples hurts DeepSeek-R1 because its instruction format conflicts with few-shot structure.
- **Software engineering tasks**: Improvements are limited due to the difficulty of automated evaluation (long build/test cycles).
- **General capabilities gap**: Slightly behind DeepSeek-V3 on function calling, multi-turn conversation, and role-playing tasks.
- **Prompt sensitivity**: Performance depends heavily on prompt format; small changes can cause significant degradation.
