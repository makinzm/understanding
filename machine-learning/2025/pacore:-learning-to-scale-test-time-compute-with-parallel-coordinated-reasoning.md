# Meta Information

- URL: [PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning](https://arxiv.org/abs/2601.05593)
- LICENSE: [ Deed - Attribution 4.0 International - Creative Commons ]( https://creativecommons.org/licenses/by/4.0/ )
- Reference: Hu, J., Zhang, Y., Yang, X., et al. (2025). PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning. StepFun, Tsinghua University, Peking University.

# PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning

## Problem and Motivation

Sequential chain-of-thought (CoT) reasoning packs every intermediate state into a single expanding chain, strictly coupling reasoning volume to context window capacity. Once the context window is full, the model cannot reason further. PaCoRe addresses this by replacing sequential depth with parallel breadth: multiple independent reasoning trajectories are launched in coordinated rounds, and each round's outputs are compressed into compact messages that seed the next round.

A key phenomenon the paper identifies is **Reasoning Solipsism**: vanilla language models, when given parallel reasoning outputs from other branches, tend to ignore them and simply proceed with their own reasoning. PaCoRe overcomes this by training explicit synthesis behavior via outcome-based reinforcement learning.

## Architecture

### Inference Pipeline

PaCoRe operates in $R$ rounds. In round $r$, the model generates $K_r$ parallel trajectories:

$$\Omega_r = \{\omega_r^{(1)}, \omega_r^{(2)}, \ldots, \omega_r^{(K_r)}\} \sim \pi_r(x, M_{r-1})$$

where:
- $x$ is the input problem
- $M_{r-1} = \{m_{r-1}^{(1)}, \ldots, m_{r-1}^{(K_{r-1})}\}$ is the compact message set from the previous round
- $K_r$ is the number of parallel trajectories in round $r$ (e.g., $\vec{K} = [32, 4]$ for a two-round high-compute setting)

**Context serialization**: The context is constructed as $P(x, M_{r-1})$, concatenating the problem with the reference messages under the synthesis prompt:

> [!NOTE]
> "Given the original problem and reference responses above, please provide your own comprehensive solution."

**Message compaction function $C(\cdot)$**: After each round, trajectories are compressed into messages. $C(\omega_r^{(i)})$ parses trajectory $\omega_r^{(i)}$, retains only the final conclusion segment as $m_r^{(i)}$, and discards intermediate reasoning steps. This bounds the input length for subsequent rounds.

$$m_r^{(i)} = C(\omega_r^{(i)}), \quad M_r = \{m_r^{(1)}, \ldots, m_r^{(K_r)}\}$$

**Final round**: $K_R = 1$, producing a single output trajectory $\omega_R^{(1)}$ as the answer $y = m_R^{(1)}$.

### Pseudocode

```
Input: problem x, round schedule K⃗ = [K_1, K_2, ..., K_R]
Output: answer y

M_0 = ∅  # empty initial message set
for r = 1 to R:
    Ω_r = []
    for i = 1 to K_r:
        ω = sample from π(x, M_{r-1})  # generate trajectory
        Ω_r.append(ω)
    M_r = {C(ω) for ω in Ω_r}  # compact all trajectories

y = M_R[0]  # single answer from final round
return y
```

## Training Procedure

### Base Model and RL Setup

PaCoRe initializes from an RLVR-8B checkpoint and applies strict on-policy PPO with GAE ($\lambda=1$, $\gamma=1$). The reward function $R(\omega) \in \{0, 1\}$ is a sparse binary signal evaluating correctness of the extracted final answer.

**Hyperparameters:**
- Batch size: 16 instances, 64 responses each
- Maximum sequence length: 131,072 tokens
- Temperature/top-p: 1.0
- Actor learning rate: $2 \times 10^{-6}$
- Critic learning rate: $5 \times 10^{-6}$
- Total training iterations: 700

### Message Cache and Sampling

For each training problem, a **message cache pool** of 24 messages is pre-generated. During training, $|M|$ messages are sampled from this pool with $|M| \sim \mathcal{U}(16, 24)$ for variable-size robustness.

### Two-Stage Curriculum

To prevent the model from ignoring input messages (Reasoning Solipsism) and to force genuine synthesis:

| Stage | Iterations | Filter Criterion | Purpose |
|-------|-----------|-----------------|---------|
| Stage 1 | 250 | $0 < \text{mean}(\text{message\_acc}) < 9/24$ (math), $< 15/24$ (code) | Select instances where messages are often wrong to prevent majority-vote shortcut |
| Stage 2 | 450 | $0 < \text{synthesis\_acc} < 1$ | Retain only instances where synthesis sometimes succeeds and sometimes fails, training genuine recovery |

> [!IMPORTANT]
> Stage 2 filters to cases where the model can already synthesize correctly some of the time. This means the model must learn to produce a correct answer even when all input messages are incorrect—a capability the paper calls "emergent correctness."

## Comparison with Similar Methods

| Method | Scaling Strategy | Context Usage | Synthesis Learning | Key Limitation |
|--------|----------------|--------------|-------------------|---------------|
| Sequential CoT | Depth (longer chains) | O(depth) | None | Context window bound |
| Best-of-N / Self-Consistency | Breadth (majority vote) | O(N) independent | None | Saturates; no cross-branch integration |
| PaCoRe | Breadth + coordination | O(rounds × K_r) bounded | RL-trained synthesis | Requires specialized training |

**vs. Self-Consistency**: Self-Consistency applies majority voting on independent trajectories without any cross-branch communication. It saturates quickly because additional samples provide diminishing returns without integration. PaCoRe explicitly coordinates branches and continues scaling.

**vs. Sequential CoT**: Sequential reasoning is fundamentally bottlenecked by context window size. PaCoRe decouples effective reasoning volume from context limits by compressing intermediate results.

**vs. Best-of-N**: PaCoRe demonstrates emergent correctness—recovering the right answer from a set of all-wrong input messages—which no selection strategy can achieve.

## Experiments

- **Datasets (Math)**: AIME 2025, HMMT 2025, IMO AnswerBench, Apex, Humanity's Last Exam (HLE, text-only)
- **Datasets (Code)**: LiveCodeBench, MultiChallenge
- **Training data (Math)**: Competition archives (AIME, HMMT, IMO, etc.) + open-source datasets, 2.4k problems after multi-stage quality filtering
- **Training data (Code)**: ~29k competitive programming problems + ~5k CodeForces problems with synthetic test case generation
- **Hardware**: Not specified
- **Optimizer**: PPO (on-policy, strict)
- **Baselines**: GPT-5, Qwen3-235B-Thinking, GLM-4.6, DeepSeek-V3.1, Kimi-K2-Thinking, RLVR-8B

**Compute level settings:**
- Low: $\vec{K} = [4]$
- Medium: $\vec{K} = [16]$
- High: $\vec{K} = [32, 4]$

**Key results (PaCoRe-8B at high-compute):**
- HMMT 2025: 94.5% (surpassing GPT-5 at 93.2%)
- AIME 2025: 93.7%
- LiveCodeBench: 78.2%
- Effective test-time token usage: ~2 million tokens

**Ablation findings:**
- Removing message passing degrades performance as compute scales (information bottleneck)
- Variable message set size training ($|M| \sim \mathcal{U}(8,16)$) improves robustness
- Cross-checking linguistic markers (e.g., "However, in message 3...") emerge organically during training, indicating learned synthesis rather than rote behavior

## Applicability

PaCoRe is applicable when:
- **Who**: Practitioners deploying reasoning-heavy LLMs on competition math, competitive coding, or other tasks with verifiable binary rewards
- **When**: Inference-time compute budget is large but context window limits sequential scaling; training data has verifiable correctness signals
- **Where**: Models that can serve as both synthesizer and parallel explorer (single model for all roles); compute-constrained settings benefit from the efficiency of parallel breadth over sequential depth

> [!TIP]
> Model checkpoints, training data, and inference code are available via the StepFun GitHub repository and HuggingFace.
