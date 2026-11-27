# Meta Information

- URL: [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://arxiv.org/abs/2501.17161)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., & Ma, Y. (2025). SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. arXiv:2501.17161.

# SFT Memorizes, RL Generalizes

## Overview

This paper investigates a fundamental distinction between two foundation model post-training paradigms:

- **Supervised Fine-Tuning (SFT)** tends to *memorize* the training distribution, causing performance degradation on out-of-distribution (OOD) inputs.
- **Reinforcement Learning (RL)** acquires *generalizable knowledge*, enabling consistent improvements on both in-distribution (ID) and OOD evaluations.

The key insight is that SFT maps inputs to outputs by imitating demonstrated behaviors, so the model may learn surface-level patterns without underlying reasoning. RL, by contrast, receives sparse outcome-based rewards that require the model to discover strategies that generalize across task variants.

**Who benefits from this work:**
- Researchers designing post-training pipelines for large language models (LLMs) and vision-language models (VLMs).
- Practitioners choosing between SFT and RL for tasks requiring compositional or rule-based reasoning.

---

## Problem Setting

### Task

Given a base foundation model $\pi_0$, post-training aims to produce a policy $\pi$ that maximizes expected return:

$$\max_{\pi \in \Pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} r_t \right]$$

where:
- $\mathcal{S}$ = state space (text tokens and visual observations)
- $\mathcal{A}$ = action space (output token sequences $\mathcal{V}^n$)
- $r : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ = rule-based reward function

**SFT** approximates this by behavioral cloning from a dataset $\mathcal{D} = \{(x_i, y_i)\}$ of (input, output) pairs, minimizing negative log-likelihood:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i} \log \pi_\theta(y_i \mid x_i)$$

**RL** (implemented with PPO) directly optimizes the return using environment interactions, allowing the model to explore beyond the training distribution.

---

## Sequential Revision Framework (Multi-turn RL)

The paper introduces a **sequential revision** formulation to integrate iterative verification into RL training.

### State Construction

At time step $t \geq 1$, the model input $\mathbf{v}_t^{\text{in}}$ is constructed by appending prior outputs and verifier feedback:

$$\mathbf{v}_t^{\text{in}} = \text{concat}\left(\mathbf{v}_0^{\text{in}},\ [\mathbf{v}_k^{\text{out}},\ \mathbf{v}_k^{\text{ver}}]_{k=0}^{t-1}\right)$$

where:
- $\mathbf{v}_0^{\text{in}} \in \mathcal{V}^m$ = initial text prompt (system prompt + task description)
- $\mathbf{v}_k^{\text{out}} \in \mathcal{V}^n$ = model's output at step $k$
- $\mathbf{v}_k^{\text{ver}} \in \mathcal{V}^p$ = verifier feedback at step $k$

For vision-language tasks, visual observations $o_t \in \mathcal{O}$ are concatenated into the state.

### Inference-Time Scaling

By increasing the number of sequential verification steps $T$, the model can iteratively refine its answer. With 10 verification steps, OOD performance improved by **+5.99%** compared to marginal **+0.48%** improvement with a single step.

---

## Evaluation Environments

### GeneralPoints (GP)

An arithmetic card game where the model must write an equation using all four card values that equals 24 (similar to the "24 Game").

- **GP-L**: Language-only variant (text descriptions of cards)
- **GP-VL**: Vision-language variant (images of cards)
- **In-Distribution (ID)**: Face cards J, Q, K treated as the single value 10
- **Out-of-Distribution (OOD)**: Face cards treated as three distinct values {11, 12, 13}

**Reward function (GP-L):**

| Outcome | Reward |
|---|---|
| Correct equation equal to 24 | $+5$ |
| Legal equation using all cards, but wrong result | $-1$ |
| Numbers not from the card set | $-2$ |
| Illegal equation format | $-3$ |

**Additional reward for GP-VL:**

| Outcome | Reward |
|---|---|
| Failed card recognition | $-1.5$ |

### V-IRL

A real-world visual navigation environment using Google Street View imagery. The model must navigate New York City routes by selecting cardinal directions at each intersection.

- **V-IRL-L**: Language-only (textual landmark descriptions)
- **V-IRL-VL**: Vision-language (actual Street View images)
- **ID**: Absolute orientation (8 cardinal directions: N, NE, E, SE, S, SW, W, NW)
- **OOD**: Relative orientation (4 relative moves: Forward, Back, Left, Right)

**Reward function (V-IRL):**

| Outcome | Reward |
|---|---|
| Correct action at current coordinate | $+1$ |
| Wrong action or exceeded max steps | $-1$ |
| Failed landmark detection (VL only) | $-1.5$ |

---

## Datasets

| Dataset | Training | Evaluation |
|---|---|---|
| GeneralPoints | Card quadruples sampled from a standard 52-card deck; verified by expert solver to have ≥1 solution equaling 24 | Same card distribution, rule variant swapped (OOD) |
| V-IRL | 1,000 unique New York City routes | 18 routes across 9 cities (2 per city), from VLN mini benchmark (Yang et al., 2024) |

---

## Algorithm

### SFT Pipeline

```
Input: Base model π₀, demonstration dataset D = {(xᵢ, yᵢ)}
1. Initialize θ ← π₀
2. For each epoch:
   a. Sample batch (x, y) from D
   b. Compute L_SFT(θ) = -Σ log π_θ(y | x)
   c. Update θ ← θ - η ∇L_SFT(θ)
Output: Fine-tuned model π_θ
```

### RL (PPO) Pipeline with Sequential Revision

```
Input: SFT-initialized model π_SFT, environment E with reward r(·)
1. Initialize θ ← π_SFT
2. For each training episode:
   a. Sample initial state s₀ = v₀^in (text + optional visual obs)
   b. For t = 0, ..., T-1:
      i.  Sample action aₜ = v_t^out ~ π_θ(· | sₜ)
      ii. Get verifier feedback v_t^ver from environment E
      iii. Compute reward rₜ = r(sₜ, aₜ)
      iv. Construct next state:
          s_{t+1} = concat(s₀, [v_k^out, v_k^ver]_{k=0}^t)
   c. Update π_θ using PPO on collected trajectory
Output: RL-trained policy π_θ
```

> [!IMPORTANT]
> SFT initialization is **required** before RL training. Training RL directly on the base model ($\pi_0$) without prior SFT results in complete failure due to the base model's inability to follow instructions.

---

## Compute Budget Comparison

To enable fair comparison between SFT and RL, the authors use FLOPs as a common unit:

$$X_{\text{SFT}} = 6N(D_{\text{init}} + D_{\text{SFT}})$$

$$X_{\text{RL}} = 6N(D_{\text{init}} + D_{\text{RL}}) + 2N D_{\text{buffer}}$$

where:
- $N$ = number of model parameters
- $D_{\text{init}}$ = FLOPs for SFT initialization phase
- $D_{\text{SFT}}, D_{\text{RL}}$ = data tokens processed in respective training
- $D_{\text{buffer}}$ = rollout buffer size ($\lambda \approx 6$ for GP, $\lambda \approx 5.1$ for V-IRL)

---

## Experiments

- **Dataset**: GeneralPoints (synthetic card game) and V-IRL (real-world navigation)
- **Hardware**: 8× NVIDIA H800 GPUs (80 GB each)
- **Backbone Model**: Llama-3.2-Vision-11B (11B parameters)
- **RL Algorithm**: PPO with outcome-based rewards
- **Optimizer**: Adam (standard PPO defaults)

### Key Results

| Task | Method | ID Accuracy | OOD Accuracy | OOD Δ |
|---|---|---|---|---|
| GP-L | SFT | ~88% | ~80% | **-8.1%** |
| GP-L | RL | ~88% | ~91% | **+3.5%** |
| V-IRL-L | SFT | ~39% | ~33% | **-5.6%** |
| V-IRL-L | RL | ~52% | ~61% | **+9.3%** |
| GP-VL | SFT | degrades | degrades | (negative) |
| GP-VL | RL | improves | improves | (positive) |

The multi-turn RL approach on V-IRL improved accuracy from **44.0% → 77.8% (+33.8%)**, surpassing prior methods that required specialized VLM-LLM collaboration.

### Visual Recognition Finding

In GP-VL, RL training improves the model's card recognition accuracy (visual perception sub-task), while SFT training *degrades* visual recognition accuracy as training progresses. This suggests that RL's exploration signal causes the model to develop better perceptual grounding rather than bypassing visual input through memorization.

---

## Comparison with Similar Methods

| Aspect | SFT | RL (PPO) |
|---|---|---|
| Objective | Minimize NLL on demonstrations | Maximize expected return |
| Generalization | Memorizes training distribution | Generalizes to OOD variants |
| Data requirement | Requires labeled (input, output) pairs | Requires reward function / verifier |
| Visual grounding (VLM) | Degrades with more training | Improves with more training |
| Instruction following | Strong (good initialization) | Poor from base model alone |
| Role in pipeline | **Required for warm-start** | **Required for generalization** |

> [!NOTE]
> The paper explicitly argues that SFT and RL are *complementary*, not competing: SFT provides the instruction-following capability that RL needs to bootstrap, while RL provides the generalization that SFT cannot achieve.

> [!TIP]
> Related inference-time scaling work: [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314) and [DeepSeek-R1 (DeepSeekAI, 2025)](https://arxiv.org/abs/2501.12948).

---

## Limitations

1. **SFT on GP-VL**: Despite extensive hyperparameter tuning, SFT failed to achieve even comparable ID performance on the vision-language variant, an unexpected result not fully explained.
2. **RL recovery**: When RL is applied to an overfitted SFT checkpoint (trained too long), RL cannot recover the performance. The SFT initialization checkpoint must be chosen carefully (early stopping).
