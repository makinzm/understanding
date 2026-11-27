# Meta Information

- URL: [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hubotter, J., Lubeck, F., Behric, L., Baumann, A., Bagatella, M., Marta, D., Hakimi, I., Shenfeld, I., Kleine Buening, T., Guestrin, C., & Krause, A. (2026). Reinforcement Learning via Self-Distillation. arXiv:2601.20802.

# Reinforcement Learning via Self-Distillation (SDPO)

## Overview

Self-Distillation Policy Optimization (SDPO) addresses the credit-assignment bottleneck in Reinforcement Learning with Verifiable Rewards (RLVR) for large language models. Standard RLVR pipelines (e.g., GRPO) rely exclusively on scalar outcome rewards (often binary pass/fail). When all rollouts in a batch receive identical rewards, policy gradient advantages collapse to zero, stalling training.

SDPO formalizes **Reinforcement Learning with Rich Feedback (RLRF)**: the environment exposes tokenized feedback $f$ (runtime errors, test output, sample solutions) beyond scalar rewards. The key insight is that a sufficiently capable LLM, conditioned on its own failed attempt and feedback, can generate corrected token distributions retrospectively — a **self-teacher** — whose predictions can be distilled back into the policy without any external supervision.

> [!IMPORTANT]
> SDPO's effectiveness depends on the model's in-context learning capability. It underperforms GRPO on smaller models (e.g., Qwen2.5-1.5B, Qwen3-0.6B) where retrospective correction is weak. For models of 7B+ parameters (Qwen3-8B, Olmo3-7B), improvements are substantial.

## Problem Setting

- **Input**: A question $x \in \mathcal{X}$, a policy LLM $\pi_\theta$, and an environment that produces both scalar reward $r$ and tokenized feedback $f$ per rollout.
- **Output**: An updated policy $\pi_\theta$ that maximizes expected reward with higher sample efficiency than scalar-only methods.
- **Rich feedback types**: compiler errors, failed test case outputs, passing solutions sampled from the same batch.

## Self-Teacher Definition

The **self-teacher** is the current policy conditioned on both the original question $x$ and the feedback $f$:

```math
\begin{align}
  \pi_\theta(\cdot \mid x, f)
\end{align}
```

This is not a separate model; it is the same $\pi_\theta$ with an extended context. The self-teacher predicts, at each token position $t$, what token the corrected response should contain — given that it has seen what went wrong.

## SDPO Loss Function

The SDPO objective minimizes the forward KL divergence from the student (no feedback) to the self-teacher (with feedback), preventing the student from regressing:

```math
\begin{align}
  \mathcal{L}_{\mathrm{SDPO}}(\theta) := \sum_{t} \mathrm{KL}\!\left(\pi_{\theta}(\cdot \mid x, y_{<t}) \;\|\; \mathrm{stopgrad}\!\left(\pi_{\theta}(\cdot \mid x, f, y_{<t})\right)\right)
\end{align}
```

where `stopgrad` blocks gradient flow through the teacher head, ensuring the student is always pulled toward the teacher and not the reverse.

## Gradient Estimator (Proposition 2.1)

The gradient of $\mathcal{L}_{\mathrm{SDPO}}$ takes the form of a policy gradient with per-token advantages:

```math
\begin{align}
  \nabla_{\theta}\mathcal{L}_{\mathrm{SDPO}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\!\left[\sum_{t=1}^{|y|} \sum_{\hat{y}_t \in \mathcal{V}} \nabla_{\theta}\log\pi_{\theta}(\hat{y}_t \mid x, y_{<t}) \cdot \log\frac{\pi_{\theta}(\hat{y}_t \mid x, y_{<t})}{\pi_{\theta}(\hat{y}_t \mid x, f, y_{<t})}\right]
\end{align}
```

The per-token advantage for token $\hat{y}_{i,t}$ in rollout $i$ is:

```math
\begin{align}
  A_{i,t}^{\mathrm{SDPO}}(\hat{y}_{i,t}) = \log\frac{\pi_{\theta}(\hat{y}_{i,t} \mid x, y_{i,<t})}{\pi_{\theta}(\hat{y}_{i,t} \mid x, f_i, y_{i,<t})}
\end{align}
```

This advantage is **nonzero wherever student and self-teacher disagree** — providing dense, token-level credit assignment. In contrast, GRPO advantages are:

```math
\begin{align}
  A_{i,t}^{\mathrm{GRPO}}(\hat{y}_{i,t}) := \mathbf{1}\{y_{i,t} = \hat{y}_{i,t}\}(r_i - \overline{r})
\end{align}
```

GRPO advantages are constant across all tokens within a rollout and collapse to zero when all rollouts share the same reward.

## Algorithm: SDPO

```
Input: policy π_θ; dataset of questions x; rollouts per question G; environment
────────────────────────────────────────────────────────
repeat:
    Sample question x from dataset
    Sample G responses {y_i}_{i=1}^G ~ π_θ(·|x)
    Query environment → obtain rewards r_i, feedback f_i for each y_i

    // Self-teacher forward pass (parallelized with student):
    For each (y_i, f_i): compute log π_θ(y_{i,t} | x, f_i, y_{i,<t})

    // Compute per-token advantages A_{i,t}^SDPO

    // Update θ via gradient descent on L_SDPO(θ)
until converged
```

The only compute overhead vs. GRPO is the self-teacher forward pass, which runs in parallel with the student pass. Memory is managed with **top-K distillation** ($K=100$): the KL is computed over the $K$ most likely vocabulary entries, approximating the tail mass.

## Use Without Rich Feedback (Scalar-Only Environments)

When the environment provides only scalar rewards, SDPO bootstraps rich feedback by treating **successful rollouts from the same batch as sample solutions** for failed rollouts on identical questions. This synthetic feedback consistently outperforms GRPO in scalar-only settings as well.

## Stability Improvements

| Technique | Description |
|-----------|-------------|
| Frozen teacher | Teacher parameters are fixed (EMA or initial snapshot); student updates do not affect teacher during training step |
| Trust-region interpolation | Teacher is a convex combination of current policy and a reference: $\alpha \cdot \pi_{\theta_{\mathrm{ref}}} + (1-\alpha) \cdot \pi_\theta$ |
| Symmetric divergence | Jensen-Shannon divergence instead of forward KL, reducing variance |

## Comparison with Similar Methods

| Method | Feedback Type | Credit Assignment | External Teacher |
|--------|--------------|-------------------|-----------------|
| GRPO | Scalar reward | Sequence-level | No |
| STaR | Binary (pass/fail) | SFT on successes | No |
| RAFT / SFT on teacher | Token-level logits | Token-level | Yes (frozen external) |
| On-policy distillation | Token-level logits | Token-level | Yes (stronger external) |
| **SDPO** | Tokenized env. feedback | Token-level (logit) | No (self) |

> [!NOTE]
> "Each LLM is implicitly a Process Reward Model (PRM) through retrospection." SDPO makes this implicit capability explicit: the self-teacher assigns higher probability to correct tokens conditioned on feedback, providing a process-level signal without training a separate PRM.

# Experiments

## Datasets

- **SciKnowEval L3** — Chemistry, Physics, Biology, Materials Science reasoning subsets. Used for scalar-feedback evaluation.
- **ToolAlpaca** — Tool-API mapping: given an API specification and user request, output the correct tool call. Scalar reward.
- **LiveCodeBench v6 (LCBv6)** — 131 competitive programming problems released Feb–May 2025. Training uses public tests (50% random subset of private); evaluation uses avg@4 rollouts.

## Hardware

Not explicitly specified in the paper.

## Optimizer

Standard AdamW (consistent with GRPO baseline). Improved GRPO variant includes asymmetric clipping, unbiased advantage normalization, and off-policy correction.

## Key Results

**Scalar-only environments (SciKnowEval, ToolAlpaca):**
- On Chemistry with Olmo3-7B, SDPO reaches GRPO's 5-hour accuracy in ~30 minutes — approximately 10x wall-clock speedup.
- SDPO produces 3–7x shorter generations while maintaining higher accuracy, eliminating "superficial reasoning" patterns (circular loops, filler phrases) common in GRPO.

**Rich feedback (LiveCodeBench v6):**
- SDPO: 48.8% vs. GRPO: 41.2% final accuracy (+7.6 pp).
- SDPO reaches GRPO's final accuracy 4x faster (in terms of number of generated tokens).
- Outperforms Claude Sonnet 4 (40.5%) and Claude Opus 4 (39.7%) on LCBv6.

**Dense credit assignment ablation (LCBv6):**
- Logit-level SDPO (K=100): 50.6%
- Token-level SDPO (top-1 argmax): 48.2%
- Sequence-level SDPO: 46.8%
- GRPO: 41.2%

Even sequence-level SDPO (+5.6 pp vs GRPO) demonstrates that rich feedback itself provides significant value beyond dense credit assignment.

**Test-time self-distillation on very hard problems (pass@64 < 0.03):**
- SDPO discovery@2750: 53.2% vs. Best-of-k: 41.5%
- SDPO achieves equivalent discovery with 2.4x fewer generations on hard tasks
- SDPO uniquely solved one question unsolvable by either Best-of-k or multi-turn reprompting within 2750 attempts (found after 321 SDPO attempts, compressing history into weights rather than context)

## Catastrophic Forgetting

SDPO (on-policy) better preserves holdout task performance than off-policy SFT on self-teacher outputs, confirming that on-policy training is critical for avoiding capability regression.
