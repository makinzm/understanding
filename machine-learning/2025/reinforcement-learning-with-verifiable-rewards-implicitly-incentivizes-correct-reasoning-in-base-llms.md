# Meta Information

- URL: [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wen, X., Liu, Z., Zheng, S., Xu, Z., Ye, S., Wu, Z., Liang, X., Wang, Y., Li, J., Miao, Z., Bian, J., & Yang, M. (2025). Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs. arXiv:2506.14245.

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

## Background and Problem Statement

Reinforcement Learning with Verifiable Rewards (RLVR) has become a dominant post-training paradigm for improving LLM reasoning, following methods like DeepSeek-R1, GRPO, and DAPO. However, a controversy arose when Yue et al. (2025) observed that RLVR-tuned models frequently *underperform* base models on the standard Pass@K metric at moderate-to-large K values. This observation led to the hypothesis that RLVR merely re-weights pre-existing correct reasoning paths from the base model rather than generating genuinely new ones.

The key question: **Does RLVR produce new correct reasoning paths, or does it only shift probability mass toward already-existing correct solutions?**

This paper argues that standard Pass@K is an insufficient metric for answering this question because it does not distinguish between correct answers reached through valid reasoning and correct answers reached through flawed reasoning.

## Core Insight: Correct Answers via Incorrect Reasoning

A critical observation motivating this paper is that base LLMs frequently produce correct final answers via *logically flawed chain-of-thought (CoT)*. For example, a model may produce an arithmetic error mid-solution but arrive at the right answer through a subsequent compensating error. Standard Pass@K credits such responses as successes, creating an inflated estimate of the base model's true reasoning quality.

> [!NOTE]
> The authors characterize this failure mode as "correct answer via incorrect reasoning." Pass@K conflates these spurious successes with genuine reasoning, making RLVR look less impactful than it truly is.

## CoT-Pass@K: A New Metric

To properly evaluate RLVR's effect, the authors introduce **CoT-Pass@K**, which requires that at least one of $K$ sampled responses contains both a logically valid CoT *and* a correct final answer.

**Formal Definitions:**

Let $G$ be the total number of sampled responses for a question $q$. Define:

- $C$ = number of responses with correct final answers
- $D$ = number of responses with both correct CoT and correct final answer, where $D \le C$

**Correctness indicator functions:**

```math
\begin{align}
  \mathcal{I}_{\text{Ans}}(a_i) &= \begin{cases} 1 & \text{if } a_i \text{ is correct} \\ 0 & \text{otherwise} \end{cases} \\
  \mathcal{I}_{\text{CoT}}(c_i) &= \begin{cases} 1 & \text{if } c_i \text{ is logically accurate and complete} \\ 0 & \text{otherwise} \end{cases}
\end{align}
```

So $D = \sum_{i=1}^G \mathcal{I}_{\text{CoT}}(c_i) \cdot \mathcal{I}_{\text{Ans}}(a_i)$.

**Per-prompt metrics:**

```math
\begin{align}
  \text{Pass@K}(q) &= 1 - \frac{\binom{G - C}{K}}{\binom{G}{K}} \\
  \text{CoT-Pass@K}(q) &= 1 - \frac{\binom{G - D}{K}}{\binom{G}{K}}
\end{align}
```

The dataset-level metric is the mean over all evaluation questions.

**Supporting metrics for training dynamics analysis:**

```math
\begin{align}
  P(\text{CA}) &= \frac{C}{G} \quad \text{(fraction of responses with correct answers)} \\
  P(\text{CC}|\text{CA}) &= \frac{D}{C} \quad \text{(fraction of correct-answer responses that also have valid CoT)}
\end{align}
```

## Theoretical Analysis: Why RLVR Incentivizes Correct Reasoning

The paper provides a formal theorem establishing that GRPO-style RLVR implicitly incentivizes correct CoT under two mild assumptions.

**Setup:** GRPO samples a group $Y = \{y_1, \dots, y_G\}$ of responses for question $q$ and computes the normalized advantage:

```math
\begin{align}
  \hat{A}(y_i) = \frac{R(y_i) - \mu_Y}{\sigma_Y}
\end{align}
```

where $\mu_Y$ and $\sigma_Y$ are the empirical mean and standard deviation of rewards in the group, and $R(y_i) \in \{0, 1\}$ is the binary verifiable reward (1 if answer correct).

**Assumptions:**

1. **Logical Coherence**: Correct CoTs yield correct answers with higher probability than incorrect CoTs. Formally, $\alpha = P(\text{correct answer} \mid \text{correct CoT}) > \beta = P(\text{correct answer} \mid \text{incorrect CoT})$.
2. **Stable Advantage Estimation**: The group size $G$ is sufficiently large so that $\sigma_Y > 0$ with high probability (i.e., the group is not all-correct or all-incorrect).

**Theorem 1:**

Under these assumptions:

```math
\begin{align}
  \mathbb{E}[\hat{A}(y_i) \mid \text{correct CoT}] &> 0 \\
  \mathbb{E}[\hat{A}(y_i) \mid \text{incorrect CoT}] &< 0
\end{align}
```

This means that in expectation, the GRPO policy gradient update positively reinforces responses with correct CoTs and negatively reinforces those with incorrect CoTs—even though the reward signal only checks the final answer.

**Proof sketch:** Responses with correct CoT have $P(\text{reward}=1) = \alpha$ while those with incorrect CoT have $P(\text{reward}=1) = \beta < \alpha$. The group advantage normalizes by $\sigma_Y$, and since correct-CoT responses earn higher rewards in expectation, they receive positive expected advantage. The full proof follows from computing $\mathbb{E}[R(y_i) - \mu_Y]$ conditional on CoT correctness, yielding positive expectation for correct CoT and negative for incorrect CoT.

> [!IMPORTANT]
> The theorem shows RLVR's incentive for correct reasoning is *implicit*: no explicit CoT reward signal is needed. Correct reasoning is incentivized as a byproduct of optimizing answer correctness, provided the logical coherence assumption holds.

## Comparison with Pass@K Analysis

**Why Pass@K misleads:** When a base model has a small but non-zero probability of producing "correct answer via incorrect reasoning" (CAIR), these spurious successes accumulate at large $K$ and inflate the base model's apparent capability. RLVR selectively reduces CAIR responses, causing the RLVR model to appear worse under standard Pass@K at large $K$, even though it is actually *better* at reasoning.

**CoT-Pass@K gap:** Because $D \le C$ (valid-reasoning successes are a subset of answer-correct successes), we have $\text{CoT-Pass@K} \le \text{Pass@K}$ for any model. The gap between them ($\Delta = \text{Pass@K} - \text{CoT-Pass@K}$) measures the degree of CAIR contamination. RLVR reduces this gap, increasing $P(\text{CC}|\text{CA})$.

| Metric | What it measures | Problem with base models |
|--------|-----------------|--------------------------|
| Pass@K | Any correct answer in K samples | Inflated by CAIR responses |
| CoT-Pass@K | Correct answer *and* valid reasoning | Accurately reflects reasoning ability |
| P(CC\|CA) | Fraction of correct answers with valid CoT | Tracks reasoning quality during training |

## Algorithm: GRPO Policy Update

```
Input: policy π_θ, question q, group size G, reward function R
1. Sample G responses {y_1, ..., y_G} ~ π_θ(·|q)
2. Compute rewards r_i = R(y_i) for each i
3. Compute group statistics: μ_Y = mean(r), σ_Y = std(r)
4. Compute normalized advantages: Â(y_i) = (r_i - μ_Y) / σ_Y
5. Update: ∇_θ J(θ) ≈ (1/G) Σ_i Â(y_i) · ∇_θ log π_θ(y_i | q)
6. Apply policy ratio clipping (PPO-style) to stabilize updates
```

> [!TIP]
> DAPO (Direct Advantage Policy Optimization) is a variant of GRPO that improves training stability by filtering groups where all responses are correct or all are incorrect, which prevents degenerate $\sigma_Y = 0$ cases and aligns with Assumption 2 of Theorem 1.

## CoT Verification Methodology

Verifying CoT correctness is non-trivial. The authors use **DeepSeek-R1-0528-Qwen3-8B as an LLM-as-judge** with structured prompts. To reduce false positives from the verifier, they sample the judge $n=3$ times per response and apply three aggregation strategies:

- **Any-correct**: Verifier returns 1 if at least one judge call says the CoT is correct
- **All-correct**: Verifier returns 1 only if all $n=3$ judge calls agree the CoT is correct
- **Majority-correct**: Verifier returns 1 if at least 2 out of 3 judge calls agree

The authors report results under all three strategies to ensure robustness.

## Experiments

- **Datasets**: AIME 2024 (30 problems), AIME 2025 (30 problems, post-training cutoff to avoid contamination), Math-500 (500 problems), AMC23, Minerva (physics and mathematics)
- **Training data**: DAPO-Math-17k (17,000 questions used to train the RLVR model)
- **Models**: Base model: Qwen2.5-32B; RLVR model: DAPO-Qwen-32B (trained from base); CoT verifier: DeepSeek-R1-0528-Qwen3-8B
- **Hardware**: 32 AMD MI300X GPUs; full training took over 2 weeks
- **Sampling budget**: $G$ up to 1024 responses per question for Pass@K / CoT-Pass@K evaluation
- **Key results**:
  - Standard Pass@K shows the base model (Qwen2.5-32B) catching up to or matching the RLVR model (DAPO-Qwen-32B) at $K \ge 32$–$64$, appearing to contradict RLVR's value
  - CoT-Pass@K reveals a persistent, increasing gap favoring the RLVR model at all $K$ values up to 1024
  - On AIME 2025 (cleanest benchmark), the CoT-Pass@K gap is especially pronounced, confirming that RLVR's advantage is not due to data contamination
  - $P(\text{CC}|\text{CA})$ increases monotonically throughout DAPO training from first steps, confirming the implicit incentive takes effect immediately
  - $P(\text{CA})$ saturates quickly for easy problems but $P(\text{CC}|\text{CA})$ continues growing, showing that answer-level accuracy is an insufficient proxy for training progress

## Differences from Related Work

| Approach | What is optimized | Answer-level reward | CoT-level reward | Key limitation |
|----------|------------------|---------------------|------------------|----------------|
| RLVR / GRPO / DAPO | Policy with verifiable rewards | Yes (binary) | No (implicit only) | Reward does not directly measure reasoning |
| Process Reward Models (PRM) | Per-step reward | Indirect | Yes (explicit) | Requires expensive step-level annotations |
| DPO / Preference optimization | Response ranking | Indirect | No | Depends on quality of preference data |
| **This paper's CoT-Pass@K** | Evaluation metric (not training) | Yes | Yes (required) | Relies on LLM verifier accuracy |

> [!NOTE]
> Unlike PRM-based methods that explicitly train on step-level feedback, RLVR achieves implicit correct-reasoning incentivization without any CoT annotations. The authors argue this is a feature, not a limitation—it means RLVR scales without costly process labels.

## Applicability

- **Who**: ML researchers studying LLM post-training, practitioners selecting between RLVR and SFT/PRM-based approaches
- **When**: Evaluating post-training methods for mathematical reasoning; designing new RLVR training objectives
- **Where**: Tasks with verifiable binary rewards (math, code, formal proofs); benchmarks like AIME, Math-500, AMC

The CoT-Pass@K metric is broadly applicable whenever an LLM judge can assess reasoning validity—not limited to mathematics.

## Future Directions

1. Develop lightweight, reliable CoT verifiers to reduce the cost of large-model-as-judge evaluation
2. Design contamination-free, continuously updated benchmarks
3. Explore RLVR variants that *directly* optimize for CoT correctness rather than relying on the implicit incentive
4. Investigate whether the logical coherence assumption ($\alpha > \beta$) holds across different domains and how its strength affects training outcomes
