# Meta Information

- URL: [Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yifan Yu, Yuqing Jian, Junxiong Wang, Zhongzhu Zhou, Donglin Zhuang, Xinyu Fang, Sri Yanamandra, Xiaoxia Wu, Qingyang Wu, Shuaiwen Leon Song, Tri Dao, Ben Athiwaratkun, James Zou, Fan Lai, Chenfeng Xu (2026). Introspective Diffusion Language Models. arXiv:2604.11035.

# Introspective Diffusion Language Models (I-DLM)

## Background: Diffusion Language Models vs. Autoregressive Models

Autoregressive (AR) language models generate tokens sequentially, one at a time, with each step conditioned on all previous tokens. This sequential nature creates a throughput bottleneck at inference time. Diffusion language models (DLMs) instead mask out and re-predict tokens in parallel, offering the potential for higher throughput — but have historically suffered from a significant quality gap compared to AR models.

This paper identifies the root cause of that quality gap and introduces **I-DLM**, a training and decoding method that closes the gap while preserving the parallel generation advantage.

## Core Concept: Introspective Consistency

The key insight is that prior DLMs lack **introspective consistency** — the property that a model agrees with its own generated tokens when re-examining them. A model is introspectively consistent if, after generating token $x_k$, inspecting the same context again yields the same probability distribution.

The paper formalizes this as the **introspective acceptance rate**:

```math
\begin{align}
  \alpha = \frac{1}{L} \sum_{k=1}^{L} \min\!\left(1,\; \frac{p_k(x_k)}{q_k(x_k)}\right)
\end{align}
```

where $p_k$ is the **causal anchor distribution** (the AR-style distribution at position $k$, conditioned strictly on prior tokens) and $q_k$ is the model's **generation distribution** (the distribution used when producing $x_k$ in parallel). When $\alpha = 1$, the model fully endorses its own outputs. Prior DLMs score far below this:

| Model | Introspective Acceptance Rate $\alpha$ |
|---|---|
| AR model (any) | 1.000 (by construction) |
| LLaDA 2.0-flash | 0.568 |
| SDAR | 0.699 |
| I-DLM-8B | ≈ 1.000 |

AR models achieve $\alpha = 1$ structurally: causal masking ensures each position only sees previous tokens, and logit shifting (position $i$ predicts token $i+1$) means the generation distribution and causal anchor distribution are identical.

## Training: Introspective-Consistency Training

I-DLM converts a pretrained AR model (specifically Qwen3) into a DLM via continued training with three design choices that preserve the introspective consistency inherited from AR training:

### 1. Strict Causal Masking

Unlike block-diffusion approaches (e.g., SDAR) that use a mixture of causal and bidirectional attention within blocks, I-DLM applies **strict causal masking uniformly** across all positions. Every position $i$ attends only to positions $j \leq i$.

> [!IMPORTANT]
> Block-causal masking in SDAR allows positions within a block to attend to each other bidirectionally. This breaks the one-to-one correspondence between the generation distribution and the causal anchor distribution, reducing $\alpha$.

### 2. Logit Shifting

The model at position $i$ predicts token $x_{i+1}$, not $x_i$. This one-position shift aligns the DLM's generation distribution with the AR-style causal anchor:

- Input: token sequence $x_0, x_1, \ldots, x_{L-1}$ with some tokens replaced by `[MASK]`
- Hidden state at position $i$: $h_i \in \mathbb{R}^d$
- Prediction target: $x_{i+1}$ (logit shift by one position)

This means that even when tokens are masked during diffusion-style generation, position $i$'s prediction corresponds exactly to the causal AR distribution for $x_{i+1}$.

### 3. All-Masked Objective with Auto-Balanced Loss

All input tokens are replaced with `[MASK]` during training (100% masking rate), eliminating the need to tune a masking schedule. The loss combines cross-entropy over both masked and clean positions:

```math
\begin{align}
  \mathcal{L} = \mathcal{L}_{\text{mask}} + \hat{s} \cdot \mathcal{L}_{\text{clean}}
\end{align}
```

where $\hat{s} = \mathcal{L}_{\text{mask}} / \mathcal{L}_{\text{clean}}$ dynamically reweights the two terms to equalize gradient magnitudes. This prevents either loss term from dominating during early or late training.

**Training efficiency**: I-DLM requires only **4.5 billion tokens** on 8 H100 GPUs, compared to SDAR's 54 billion tokens — a 12× reduction — while achieving dramatically better quality (AIME-24: 69.6 vs. 10.0).

## Decoding: Introspective Strided Decoding (ISD)

ISD is the key inference algorithm. It generates multiple tokens per forward pass while guaranteeing that accepted tokens are statistically equivalent to those from the causal anchor (AR) distribution — a lossless guarantee analogous to speculative decoding.

### Tokens Per Forward (TPF)

The efficiency of any parallel decoding method is measured by the expected number of accepted tokens per forward pass. For ISD with stride $N$:

```math
\begin{align}
  \text{TPF}_N = \frac{2 + P_1 + P_2 + \cdots + P_{N-2}}{2 - P_{N-1}}
\end{align}
```

where $P_k = p^k$ is the cumulative acceptance probability after $k$ positions at per-token acceptance rate $p$. At $p = 1$ (perfect consistency), $\text{TPF}_N = N$. At $p = 0$, $\text{TPF}_N = 1$ (degrades to AR). For a practical $p = 0.90$ and $N = 8$, ISD achieves $\text{TPF} \approx 4.01$.

### ISD Algorithm

**Input**: Prompt tokens $c = (c_1, \ldots, c_m)$, stride $N$, acceptance rate threshold $p$

```
Initialize: output sequence y = []

Bootstrap step:
  1. Append N [MASK] tokens to c
  2. Run one forward pass with logit shift
  3. Accept position m+1 unconditionally (quality-guaranteed via logit shift)
  4. Sample positions m+2 through m+N as speculative proposals q

Stride + Introspection loop (repeat until EOS):
  1. Append N-1 [MASK] tokens to current sequence
  2. Single forward pass:
     a. VERIFY: for each prior speculative token x_k, compute p_k(x_k)/q_k(x_k)
        - Accept if uniform sample u < min(1, p_k/q_k)
        - On rejection: resample from max(0, p - q), discard subsequent tokens
     b. GENERATE: sample N-1 new speculative proposals from unmasked positions
  3. Append accepted tokens + one guaranteed token to y

Adaptive stride: adjust N dynamically based on recent acceptance rates
```

The verification (step 2a) and generation (step 2b) happen in a **single forward pass**, unlike speculative decoding which requires a separate verifier model.

### Comparison with Alternative DLM Decoding Methods

| Method | TPF limit | Efficiency at $p=0.9$ | Key limitation |
|---|---|---|---|
| ISD (ours) | $N$ (linear in stride) | 1.08–2.29× | None |
| SDAR (block diffusion) | $N/2$ (capped) | 0.64–0.72× | Mandatory KV-commit pass doubles overhead |
| TiDAR (branched) | $N/(N+1) < 1$ | < 1× | $O(N^2)$ branch queries; only one selected |
| EAGLE-3 (speculative) | depends on draft | baseline | Separate draft model required |

> [!NOTE]
> I-DLM with ISD is the only DLM decoding method achieving positive compute efficiency (TPF/overhead > 1) at practical acceptance rates.

## Lossless Mode: Relaxed ISD (R-ISD)

For exact equivalence to the base AR model's output distribution, R-ISD adds **gated LoRA adapters** (rank 128) applied only to `[MASK]` positions. These adapters modify only the diffusion-specific forward passes (where `[MASK]` tokens exist), leaving the causal anchor distributions unchanged. This ensures the acceptance criterion $\min(1, p_k/q_k)$ uses exactly the base model's $p_k$, achieving bit-for-bit equivalence with AR sampling.

## Serving System Innovations

I-DLM inherits the causal attention structure of AR models, enabling integration with existing AR serving stacks (SGLang) with three optimizations:

1. **CUDA Graph Capture**: Because ISD's forward-pass structure is fixed per stride, CUDA graphs can be captured and replayed without recompilation. Contributes **42–76%** of total speedup gain.

2. **Stationary-Batch Scheduling**: ISD reuses the same batch object across all stride steps, caching metadata and performing a single batched KV scatter. Reduces CPU scheduling overhead that plagues existing DLM servers.

3. **Kernel Fusion**: Verification (acceptance sampling) is fused into a single Triton kernel with online softmax. In approximately **78%** of positions, the Gumbel-max correction for argmax proposals is bypassed, saving computation.

## Experiments

- **Datasets (Benchmarks)**:
  - Knowledge/Reasoning: ARC-C, MMLU, MMLU-Pro, GPQA, GPQA-Diamond
  - Mathematics: GSM8K, MATH-500, MathBench, AIME-24, AIME-25
  - Code generation: HumanEval, MBPP, LiveCodeBench-v6
  - Instruction following: IFEval
- **Hardware**: 8× NVIDIA H100 GPUs (training); H100 cluster (inference benchmarks)
- **Optimizer**: Not specified explicitly; standard AdamW assumed from context
- **Base models**: Qwen3-8B and Qwen3-32B (converted via continued training)
- **Baselines**: LLaDA-2.1-mini (16B), LLaDA-2.1-flash (100B), SDAR-8B, Qwen3-8B/32B (AR)

**Key Results**:
- I-DLM-8B matches Qwen3-8B (same-scale AR) across all 15 benchmarks
- AIME-24: I-DLM-8B scores 69.6 vs. SDAR-8B's 10.0 and LLaDA-2.1-mini (16B)'s 43.3
- LiveCodeBench-v6: 45.7 vs. LLaDA-2.1-mini's 30.4
- Throughput at concurrency C=16–32: 2.2–3.8× higher than LLaDA-2.1-mini
- I-DLM-32B surpasses LLaDA-2.1-flash (100B, ~3× larger) on most benchmarks

**Ablation (effect of removing introspective training)**:
Removing causal masking and logit shift (reverting to block-diffusion style) degrades HumanEval from 92.7 → 60.3 and MBPP from 92.8 → 67.4, confirming these components are critical.

**Stride size ablation**:
| Stride $N$ | TPF | Quality (HumanEval) |
|---|---|---|
| 2 | 1.80 | stable |
| 4 | 2.96 | stable |
| 8 | 4.01 | stable |

Quality remains stable across stride sizes, demonstrating that larger strides provide throughput gains without accuracy trade-offs.

## Relationship to Related Work

| Method | Masking type | Attention | Logit shift | $\alpha$ | TPF potential |
|---|---|---|---|---|---|
| LLaDA / MDLM | Variable rate | Bidirectional | No | 0.57–0.70 | Limited |
| SDAR | Block-level | Block-causal | No | 0.699 | ≤ N/2 |
| TiDAR | Strided | Causal | Partial | — | < 1 |
| **I-DLM** | All-masked | Strict causal | Yes | ≈ 1.0 | N |

> [!TIP]
> The speculative decoding analogy is useful: ISD is to I-DLM as EAGLE/Medusa is to AR models, except I-DLM uses the same model for both draft and verify (no separate draft model needed), and the acceptance guarantee follows from introspective consistency.

## Applicability

I-DLM is applicable to practitioners who:
- Want the throughput benefits of parallel token generation (DLMs) without sacrificing the quality of AR models
- Are deploying reasoning-heavy tasks (math olympiads, competitive coding) at scale, where both quality and throughput matter
- Have access to a pretrained AR model (e.g., Qwen3) and want to convert it with minimal compute (4.5B tokens suffices)
- Are running high-concurrency inference where batch parallelism amplifies DLM efficiency gains

The method is **not** beneficial for single-request, low-latency settings where AR models already perform well, as the efficiency advantage of ISD grows primarily with batch concurrency.
