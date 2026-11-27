# Meta Information

- URL: [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhao, S., Xie, Z., Liu, M., Huang, J., Pang, G., Chen, F., & Grover, A. (2026). Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. ICML.

# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

## Overview

This paper proposes **On-Policy Self-Distillation (OPSD)**, a training framework that enables a single large language model (LLM) to act simultaneously as both teacher and student. The teacher sees the ground-truth answer alongside the question, while the student sees only the question. By training on student-generated trajectories with dense token-level supervision from the teacher, OPSD bridges the gap between on-policy training (used in RL methods like GRPO) and knowledge distillation (typically requiring a separate, larger teacher model).

> [!NOTE]
> "A sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self."

**Applicability:** OPSD is targeted at practitioners wanting to improve mathematical reasoning in mid-scale LLMs (1.7B–8B parameters) without access to a separate, more powerful teacher model. It is most effective when the base model is sufficiently capable to rationalize verified solution traces (empirically, ≥4B parameters showed meaningful gains).

---

## Problem Setting

### Inputs and Outputs

| Component | Input | Output |
|---|---|---|
| Student policy $p_s(\cdot \mid x)$ | Problem $x \in \mathcal{X}$ | Reasoning trace $\hat{y}$ |
| Teacher policy $p_t(\cdot \mid x, y^\star)$ | Problem $x$ + ground-truth answer $y^\star$ | Token-level distribution over the vocabulary |
| Training objective | Dataset $\mathcal{S} = \{(x, y^\star)\}$ of problem-solution pairs | Updated model parameters $\theta$ |

Both student and teacher **share identical parameters $\theta$** but differ in conditioning:
- Student: $p_s(\hat{y} \mid x; \theta)$
- Teacher: $p_t(\hat{y} \mid x, y^\star; \theta)$

The teacher's conditioning on $y^\star$ (a verified ground-truth solution) nudges the shared model toward producing richer, more accurate reasoning without needing a separate model.

---

## Method: On-Policy Self-Distillation (OPSD)

### Core Idea

OPSD minimizes the per-token divergence between the teacher and student distributions, but evaluated along **student-generated rollouts** (on-policy). This avoids the distribution mismatch common in off-policy distillation while providing dense supervision unavailable in sparse RL rewards.

### Training Objective

$$
\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{(x, y^\star) \sim \mathcal{S}} \left[ \mathbb{E}_{\hat{y} \sim p_s(\cdot \mid x)} \left[ D(p_t \| p_s)(\hat{y} \mid x) \right] \right]
$$

where $D(p_t \| p_s)$ is a **generalized Jensen-Shannon divergence** with $\beta = 0.5$:

$$
D_{\text{GJS}}(p_t \| p_s) = \beta \cdot D_{\text{KL}}(p_t \| p_m) + (1 - \beta) \cdot D_{\text{KL}}(p_s \| p_m), \quad p_m = \beta p_t + (1 - \beta) p_s
$$

This is computed per-token along the student's sampled trajectory $\hat{y} = (\hat{y}_1, \ldots, \hat{y}_T)$:

$$
D(p_t \| p_s)(\hat{y} \mid x) = \sum_{t=1}^{T} D_{\text{GJS}}\!\left(p_t(\cdot \mid x, y^\star, \hat{y}_{<t}) \| p_s(\cdot \mid x, \hat{y}_{<t})\right)
$$

> [!IMPORTANT]
> The teacher policy is **frozen** during training (fixed to its initial state) to prevent mutual collapse. Only the student parameters are updated via backpropagation through the sampled trajectories.

### Algorithm (Pseudocode)

```
Input: Dataset S = {(x, y*)} of problem-solution pairs
       Shared model θ (initialized from instruct-tuned checkpoint)
       Hyperparameters: β=0.5, lr=1e-5, steps N

For each training step:
  1. Sample batch of (x, y*) from S
  2. Sample student trajectory: ŷ ~ p_s(·|x; θ)   # student sees only question
  3. Compute teacher distribution: p_t(·|x, y*, ŷ_{<t}; θ_frozen)   # teacher sees answer
  4. For each token position t in ŷ:
       a. Compute p_t over full vocabulary (not just sampled token)
       b. Compute p_s over full vocabulary
       c. Compute GJS divergence D_GJS(p_t || p_s) at position t
  5. Loss = mean over t of D_GJS values
  6. Backpropagate through student trajectory only
  7. Update θ via AdamW
```

> [!NOTE]
> Step 4 uses **full-vocabulary logit distillation** (comparing distributions over all tokens, not just the sampled token). Ablations show this outperforms sampled-token distillation by ~2% on AIME25 and HMMT25.

---

## Comparison with Related Methods

| Method | Teacher Source | Supervision Type | On-Policy | Requires Ground Truth |
|---|---|---|---|---|
| SFT (Off-Policy) | External demonstrations | Sequence-level | No | Optional |
| GRPO (RL) | None (reward signal) | Sparse (sequence reward) | Yes | Yes (verifier) |
| On-Policy Distillation | Separate larger model | Dense (token-level) | Yes | No |
| **OPSD (this work)** | **Same model (self)** | **Dense (token-level)** | **Yes** | **Yes** |

**vs. GRPO:** GRPO assigns a scalar reward to the entire sequence, leading to zero-gradient problems when all rollouts receive identical rewards (all correct or all wrong). OPSD provides dense per-token supervision regardless of final answer correctness.

**vs. Off-Policy Distillation:** Training on expert demonstrations creates distribution mismatch—the student encounters states at inference time that the teacher never produced. OPSD trains on the student's own trajectories, matching the inference distribution.

**vs. Standard Knowledge Distillation:** Standard KD requires a separate, typically larger teacher model. OPSD reuses the same weights, making it feasible without additional model infrastructure.

---

## Experiments

- **Datasets (Training):** OpenThoughts mathematical reasoning subset — 30K problem-solution pairs with verified chain-of-thought traces.
- **Datasets (Evaluation):**
  - AIME 2024 — 30 competition-level math problems
  - AIME 2025 — 30 competition-level math problems
  - HMMT 2025 — Harvard-MIT Math Tournament problems
  - AMO-Bench — additional math olympiad benchmark
- **Models:** Qwen3-1.7B, Qwen3-4B, Qwen3-8B (instruct-tuned variants)
- **Hardware:** 8× A100 GPUs with LoRA fine-tuning
- **Optimizer:** AdamW, learning rate 1e-5, cosine decay schedule
- **Token Budget:** 2K tokens per rollout (vs. GRPO's 16K), 1 rollout per prompt (vs. GRPO's 8)

### Key Results (Average Accuracy Across Benchmarks)

| Model | Base Instruct | + SFT | + GRPO | + OPSD |
|---|---|---|---|---|
| Qwen3-1.7B | — | — | — | minimal gains |
| Qwen3-4B | 48.3% | — | 49.6% | **50.6%** |
| Qwen3-8B | 50.0% | 50.0% | 51.3% | **52.2%** |

### Token Efficiency

OPSD achieves comparable or superior performance at **4–8× lower token cost** than GRPO:
- GRPO: 16K token budget × 8 rollouts per prompt
- OPSD: 2K token budget × 1 rollout per prompt

### Ablation: Divergence Objective (Qwen3-8B)

| Distillation Strategy | AIME 2025 | HMMT 2025 |
|---|---|---|
| Full-vocabulary logits | **84.1%** | **60.0%** |
| Sampled-token only | 82.1% | 57.3% |

### Ablation: Model Scale

Gains from OPSD increase with model capacity:
- Qwen3-1.7B: minimal improvement (model too small to rationalize solutions)
- Qwen3-4B: moderate improvement
- Qwen3-8B: largest improvement

This supports the hypothesis that **self-rationalization requires a minimum capability threshold**.

---

## Limitations

1. Experiments are limited to ≤8B parameters; behavior at 70B+ scale is untested.
2. OPSD does not explicitly verify whether student trajectories are correct — it relies purely on distribution matching with a teacher that conditions on correct answers.
3. If ground-truth solutions are unavailable or unreliable, teacher conditioning quality degrades.
4. Curriculum learning (gradually increasing problem difficulty) is proposed as future work but not yet implemented.
