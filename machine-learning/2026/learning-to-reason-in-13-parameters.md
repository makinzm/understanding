# Meta Information

- URL: [Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118)
- LICENSE: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Morris, J. X., Mireshghallah, N., Ibrahim, M., & Mahloujifar, S. (2026). Learning to Reason in 13 Parameters. arXiv:2602.04118.

# Learning to Reason in 13 Parameters

## Overview

TinyLoRA is a parameter-efficient fine-tuning method that reduces the number of trainable parameters to extreme degrees (as few as 13 parameters, or 26 bytes) while preserving most of the reasoning gains from reinforcement learning (RL). The key finding is that RL-based fine-tuning requires far fewer parameter updates than supervised fine-tuning (SFT) to achieve comparable performance on math reasoning benchmarks.

- **Who**: Practitioners and researchers who want to fine-tune large language models for reasoning tasks under strict computational or storage constraints.
- **When**: When the goal is to adapt a pretrained LLM using RL (e.g., GRPO) rather than SFT.
- **Where**: Primarily demonstrated on mathematical reasoning benchmarks (GSM8K, MATH500, AIME 24, AMC 23, Minerva Math, OlympiadBench).

> [!NOTE]
> "Models trained with RL achieve 91% accuracy at 13 parameters, whereas models trained using SFT require 100-1000x larger updates to reach the same performance." (Morris et al., 2026)

## Background: LoRA and Its Variants

Standard LoRA represents a weight update as a low-rank factorization:

$$\Delta W = U \cdot V^\top, \quad U \in \mathbb{R}^{d \times r}, V \in \mathbb{R}^{d \times r}$$

where $r \ll d$ is the rank. The total trainable parameters per layer is $O(2dr)$.

**LoRA-XS** further reduces this by introducing a small $r \times r$ matrix $R$ and freezing $U, V$:

$$\Delta W = U R V^\top, \quad R \in \mathbb{R}^{r \times r}$$

Trainable parameters per layer: $r^2$ instead of $2dr$.

> [!TIP]
> See [VeRA (Kopiczko et al., 2024)](https://arxiv.org/abs/2310.11454) for another approach that shares $U, V$ across layers via random fixed matrices to drastically cut the parameter count.

## TinyLoRA Architecture

TinyLoRA extends LoRA-XS by replacing the $r \times r$ matrix $R$ with a low-dimensional vector $\mathbf{v} \in \mathbb{R}^u$ projected through a fixed random tensor $P \in \mathbb{R}^{u \times r \times r}$:

$$R = \sum_{i=1}^{u} v_i P_i, \quad P_i \in \mathbb{R}^{r \times r}$$

The full weight update rule becomes:

$$W' = W + U \left(\sum_{i=1}^{u} v_i P_i\right) V^\top$$

where $U \in \mathbb{R}^{d \times r}$ and $V \in \mathbb{R}^{d \times r}$ are frozen (random or SVD-initialized), $P \in \mathbb{R}^{u \times r \times r}$ is frozen random, and only $\mathbf{v} \in \mathbb{R}^u$ is trained.

**Input**: Pretrained weight matrix $W \in \mathbb{R}^{d \times d}$, scalar vector $\mathbf{v} \in \mathbb{R}^u$.
**Output**: Adapted weight matrix $W' \in \mathbb{R}^{d \times d}$.

### Weight Sharing

To further reduce parameters, TinyLoRA shares $\mathbf{v}$ across multiple transformer modules. If $n$ modules share one $\mathbf{v}$ of size $u$, the total trainable parameters across those modules is $u$ instead of $nu$.

With $n_\text{tie}$ modules sharing each parameter vector and total modules $n_m$, the total parameter count is:

$$\text{Total params} = \frac{n_m \cdot u}{n_\text{tie}}$$

Setting $u = 1$ and $n_\text{tie} = n_m$ gives the extreme case of **13 parameters** for a model with 13 weight matrices.

### Algorithm: TinyLoRA Forward Pass

```
Given frozen W, U, V, P and trainable v:
1. Compute R = sum_i (v[i] * P[i])        # R ∈ R^{r x r}
2. Compute delta_W = U @ R @ V^T           # delta_W ∈ R^{d x d}
3. Return W' = W + delta_W
```

## Why RL Needs Fewer Parameters than SFT

The authors hypothesize that RL via Group Relative Policy Optimization (GRPO) provides a cleaner, sparser learning signal than SFT:

- **SFT** trains via next-token prediction on full demonstrations, requiring the model to absorb all token-level noise and irrelevant stylistic variation.
- **RL (GRPO)** provides binary/scalar rewards based on correctness, selectively reinforcing outputs that solve the task and ignoring those that don't. This avoids overfitting to irrelevant surface patterns.

As a result, RL can steer the model's behavior with very small weight updates, while SFT needs much larger updates to extract task-relevant signal from noisy supervision.

> [!IMPORTANT]
> The paper does not fully explain the gap mechanistically, but raises information-theoretic questions: RL adaptation may be adjusting *output generation style* (e.g., formatting chain-of-thought) rather than injecting fundamentally new knowledge.

## Comparison with Similar Methods

| Method | Trainable Params (per layer) | Fixed Matrices | Sharing Across Layers |
|---|---|---|---|
| LoRA | $O(2dr)$ | None | No |
| LoRA-XS | $r^2$ | $U, V$ (SVD init) | No |
| VeRA | $O(d)$ | $A, B$ (random) | Yes ($A, B$ shared) |
| TinyLoRA | $u$ (as low as 1) | $U, V, P$ (random) | Yes ($\mathbf{v}$ shared) |

- TinyLoRA is more extreme than VeRA: VeRA has per-layer scaling vectors of size $d$, while TinyLoRA has shared scalar vectors of size $u \ll d$.
- Unlike LoRA-XS which keeps separate $r \times r$ matrices per layer, TinyLoRA further compresses $R$ to a $u$-dimensional vector.

## Experiments

- **Datasets**: GSM8K (7,500 math word problems), MATH training set, MATH500, Minerva Math, OlympiadBench, AIME 24, AMC 23
- **Models**: Qwen2.5-3B, Qwen2.5-7B, LLaMA-3, LLaMA-3.2, Qwen2.5-Math
- **Optimizer**: GRPO (for RL runs); standard SFT with next-token prediction
- **Hyperparameters**: Learning rates swept over $\{10^{-7}, \ldots, 2 \times 10^{-4}\}$, multiple random seeds
- **Hardware**: Not specified in detail

### Key Results

- Qwen2.5-7B base: 88.2% on GSM8K → **91.8% with 13 parameters** (RL) vs. requires $\sim 10^4$ parameters with SFT to reach equivalent accuracy.
- At 120 parameters, RL reaches ~95% while SFT reaches only ~84% on GSM8K.
- Across six math benchmarks (MATH500, Minerva Math, OlympiadBench, AIME 24, AMC 23): 196 parameters retain **87% of absolute improvement** vs. full fine-tuning.
- Qwen models are ~10x more parameter-efficient than LLaMA models at equivalent update sizes (reason unknown).

### Bit-Constrained Analysis

When total byte budget is fixed (sub-kilobyte range):
- **Tiled sharing** (repeating a small vector across blocks) outperforms **structured sharing** (grouping by module type) for math reasoning.
- **fp32** precision outperforms bf16 and fp16 even after accounting for storage overhead (4 bytes vs. 2 bytes per parameter), suggesting that numerical precision matters at extremely small parameter counts.

## Ablation Studies

**Frozen rank $r$**: Increasing $r$ from 1 to 2 helps. Larger $r$ values degrade performance, possibly because more degrees of freedom create optimization difficulties when only $\mathbf{v}$ is trained.

**Dimension trade-offs**: Per-module expressivity ($u$, the size of $\mathbf{v}$) should be maximized before increasing the sharing factor $n_\text{tie}$. In other words, give each shared group sufficient expressivity before expanding sharing.

## Limitations

- Results are limited to mathematical reasoning; generalization to scientific writing, creative tasks, or other domains is not validated.
- The mechanistic explanation for why RL requires fewer parameters than SFT remains incomplete.
- Experiments focus on a narrow set of models (Qwen, LLaMA families).
