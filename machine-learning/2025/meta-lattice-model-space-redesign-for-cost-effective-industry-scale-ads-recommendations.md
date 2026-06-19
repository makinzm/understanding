# Meta Information

- URL: [Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations](https://arxiv.org/abs/2512.09200)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Luo, L., Chen, Y., Zhang, Z., Hang, M., Gu, A., Zhang, B., Naumov, M., Yao, Y., et al. (2025). Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations. arXiv:2512.09200.

# Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations

## Overview

Meta Lattice is a unified recommendation framework for industry-scale ads systems that simultaneously addresses three structural challenges: **economic scalability** (too many separate models per domain-objective pair), **data fragmentation** (isolated training datasets per model), and **deployment constraints** (large memory and compute footprints). The framework redesigns the entire model development pipeline—from dataset construction to serving—as an integrated system rather than treating each component independently.

The framework is deployed across critical ads model types serving global Meta users and achieves 10% improvement in revenue-driving top-line metrics, 11.5% uplift in user satisfaction, 6% boost in conversion rate, and 20% capacity savings over the prior system.

> [!NOTE]
> "Model space" in this context refers to the entire space of models trained for different domain-objective combinations (e.g., click-through rate on news feed, conversion rate on marketplace). The paper argues that optimal design decisions at the system level cannot be made by optimizing individual models in isolation.

## Problem Setting

An ads recommendation system must serve predictions across multiple **domains** (platforms, surfaces) and **objectives** (clicks, purchases, engagement). Naïvely, each domain-objective pair receives a dedicated model, yielding $K$ models trained on $K$ isolated datasets. This creates three compounding problems:

1. **Fragmented ID spaces**: Similar users and items appear in multiple models, preventing knowledge sharing.
2. **Attribution window diversity**: Different objectives observe feedback on different time scales (click: seconds; purchase: days), making unified dataset construction non-trivial.
3. **Quadratic resource growth**: Adding a new domain or objective multiplies infrastructure requirements.

## Framework Components

Meta Lattice consists of five tightly integrated components:

| Component | Problem Solved | Mechanism |
|---|---|---|
| Lattice Partitioner | Model proliferation | Domain-objective grouping |
| Lattice Zipper | Delayed feedback fragmentation | Unified multi-window dataset |
| Lattice Filter | Feature redundancy across portfolios | Pareto-optimal selection |
| Lattice Networks | Unified architecture for diverse inputs | Shared backbone + domain heads |
| Lattice KTAP | Knowledge transfer efficiency | Asynchronous teacher embeddings |

### Lattice Partitioner

The Partitioner consolidates $K$ domain-objective models into fewer **portfolio groups** by solving a coverage-efficiency trade-off. The grouping strategy evaluates three criteria:

1. **ID space overlap**: Domains sharing substantial user/item populations are merged to maximize embedding reuse.
2. **Feedback density similarity**: Tasks with similar feedback delay characteristics (fresh/dense vs. delayed/sparse) are grouped together to avoid conflicting gradient signals.
3. **Revenue-weighted compute budget**: Each portfolio's model capacity is allocated proportionally to its revenue contribution.

The Partitioner accounts for 36% of the production gain in ablation analysis, making it the single highest-impact component.

### Lattice Zipper

The Zipper solves the **delayed feedback problem** by constructing a single dataset that jointly covers multiple attribution windows $\{W_1, W_2, \ldots, W_M\}$ where $W_1 < W_2 < \cdots < W_M$.

**Training procedure:**
1. Each impression $x_i$ is deterministically assigned to one attribution window $W_j$ using hashing of the impression signature and a tunable probability distribution (typically uniform).
2. The model is trained with $M$ prediction heads, one per window; impression $x_i$ only receives gradient from head $j$ during training.
3. At inference, only the longest-window head $W_M$ (the "oracle head") is active, since it captures the most complete conversion signal.

This avoids training $M$ separate models while preserving the complementary information each window provides. Fresh short-window data contributes timely signal; long-window data captures delayed conversions.

### Lattice Filter

The Filter selects a feature subset of budget $T$ from a large feature pool $\mathcal{F}$ (up to $|\mathcal{F}| = 12{,}000$ in production) in a way that does not sacrifice quality on any individual task within the portfolio.

**Feature importance vector:** For feature $i$ evaluated across $N$ tasks:

```math
\begin{align}
  \mathbf{F}_i = (f_{i,1}, \, f_{i,2}, \, \ldots, \, f_{i,N})
\end{align}
```

where $f_{i,j}$ is the permutation importance of feature $i$ for task $j$.

**Pareto domination:** Feature $i$ dominates feature $k$ if:

```math
\begin{align}
  \mathbf{F}_i \preceq \mathbf{F}_k \iff f_{i,j} \leq f_{k,j} \quad \forall j \in \{1,\ldots,N\}
\end{align}
```

**Iterative selection algorithm:**
```
Features_selected = {}
remaining_budget = T
While remaining_budget > 0:
    frontier = Pareto_frontier(F \ Features_selected)
    If |frontier| <= remaining_budget:
        Features_selected += frontier
        remaining_budget -= |frontier|
    Else:
        # Randomly sample from frontier to fill quota
        Features_selected += random_sample(frontier, remaining_budget)
        remaining_budget = 0
Return Features_selected
```

This guarantees that no task is "unfairly penalized" by weighted aggregation (which can suppress features critical to low-weight objectives).

### Lattice Networks

The unified neural architecture processes heterogeneous feature modalities—categorical, dense, and sequential—into a shared backbone representation.

**Input specification:**
- Categorical features $\mathbf{F}_c$: Each feature embedded → $\mathbf{O}_c \in \mathbb{R}^{B \times |\mathbf{F}_c| \times d}$
- Dense features $\mathbf{F}_d$: Each feature processed by a small MLP → $\mathbf{O}_d \in \mathbb{R}^{B \times |\mathbf{F}_d| \times d}$
- Sequence features $\mathbf{F}_s$: Attention-based event models produce $\mathbf{O}_s \in \mathbb{R}^{B \times (|\mathbf{F}_s| \cdot K) \times d}$ where $K$ is the sequence length and $B$ is the batch size

**Backbone computation order:**

1. **Transformer Blocks (TB) on sequences with RoPE encoding:**

```math
\begin{align}
  \mathbf{O}'_s = \text{TB}(\text{RoPE}(\mathbf{O}_s))
\end{align}
```

2. **DHEN/Wukong Fusion Block (DWFB) for cross-modal interaction:**

```math
\begin{align}
  \mathbf{O}'_{cd} = \text{DWFB}([\mathbf{O}'_s;\, \mathbf{O}_{cd}])
\end{align}
```

where $\mathbf{O}_{cd} = \text{concat}(\mathbf{O}_c, \mathbf{O}_d)$ is the non-sequence embedding pool.

3. **Extended Context Storage (ECS):** A global key-value store that provides DenseNet-style residual connections, giving all layers direct access to intermediate activations. This enables high-bandwidth information flow without stacking skip connections manually.

4. **Domain heads:** Task-specific MLP heads applied on top of the backbone produce per-objective logits.

**Training innovations to handle domain conflicts:**

- **Parameter untying:** Conflicting domains receive dedicated weight copies in backbone layers to avoid gradient interference from different data distributions.
- **QK-norm:** Layer normalization applied to query and key matrices in attention, preventing attention score collapse on mixed-modality inputs.
- **SwishRMSNorm (SwishRN):** Replaces LayerNorm with a bias-free activation:

```math
\begin{align}
  \mathbf{X}_\text{out} = \text{RMSNorm}(\mathbf{X}_\text{in}) \odot \sigma(\text{RMSNorm}(\mathbf{X}_\text{in}))
\end{align}
```

where $\sigma$ is the sigmoid function. This is numerically stable in FP8 precision.

- **Cross-domain correlation loss:** An auxiliary loss discouraging prediction collapse across domains:

```math
\begin{align}
  \mathcal{L}_\text{corr}(X, Y) = 1 - \frac{\text{Cov}(X, Y)}{\sigma_X \, \sigma_Y}
\end{align}
```

where $X$ is the label distribution and $Y$ is the model prediction distribution, computed over a batch.

**Comparison with prior architectures:**

| Feature | DLRM | DCNv2 | Wukong | Lattice Networks |
|---|---|---|---|---|
| Sequence modeling | No | No | No | Yes (TB + RoPE) |
| Cross-modal interaction | No | Cross-nets | Bit-wise | DWFB |
| Multi-task support | No | No | No | Native (domain heads) |
| Low-precision friendly | Limited | Limited | Limited | Yes (bias-less, SwishRN) |

### Lattice KTAP (Knowledge Transfer via Asynchronous Precomputation)

KTAP transfers knowledge from large teacher models to smaller student models without synchronous inference overhead.

**Mechanism:**
1. A background teacher model continuously computes backbone embeddings $\mathbf{e}_t \in \mathbb{R}^d$ for impressions and stores them in a distributed key-value cache with a time-to-live (TTL) of several hours.
2. The student model retrieves $\mathbf{e}_t$ during training and uses it in two ways:
   - **Feature-level transfer:** $\mathbf{e}_t$ is appended as an additional input feature to the student backbone.
   - **Label-level transfer:** The teacher logit $\hat{y}_t$ is used as a soft label target in a distillation loss alongside the ground truth.
3. If no valid cached embedding exists (cache miss or TTL expired), the student falls back to standard self-training.

**Stability improvements over naive distillation:**
- Feature clipping on teacher embeddings prevents large activations from dominating the student.
- Label smoothing on teacher logits prevents overconfident soft targets from harming calibration.

KTAP achieves 1.3× boost in knowledge transfer efficiency versus traditional soft-label distillation, and the asynchronous design avoids the 2× inference cost of running teacher and student in lockstep.

## Efficiency: Lattice Sketch

Lattice Sketch automates hyperparameter and parallelization strategy search, replacing manual expert tuning.

**Two-phase alternating optimization:**
1. **Hyperparameter search:** Uses beam search guided by a scaling law objective:

```math
\begin{align}
  \max \, \text{OV}_M = \max \frac{\text{AUC}}{f^{0.003}}
\end{align}
```

where $f$ is the FLOP count. This embeds the empirical expectation that AUC scales as $\log(\text{FLOPs})$, penalizing over-parameterized configurations.

2. **FSDP sharding strategy:** Dynamic programming over layer blocks $b \in [1, L]$ and memory budgets $r \in [0, R]$:

```math
\begin{align}
  \text{OPT}_b[l, r] &= \min_{s \in S} \left( \text{OPT}_b[l, r - R_b(s)] + T_b(s) + C(s) \right) \\
  s^*_{l+1,r} &= \arg\min_{s \in S} \left( \text{OPT}_b[l, r - R_b(s)] + T_b(s) + C(s) \right)
\end{align}
```

where $R_b(s)$ is the memory cost of strategy $s$ on block $b$ and $T_b(s)$ is the runtime cost. The algorithm complexity is $O(|K| \times |S| \times LR)$.

**Low-precision training:** Mixed FP8/BF16/FP32 with tensor-wise or row-wise scaling. Bias-less layers and SwishRMSNorm are required for FP8 stability (biases are sensitive to quantization error). Delivers up to 1.6× throughput improvement over BF16 baselines.

# Experiments

- **Dataset (public):** KuaiVideo — 13M entries, 8 features, three-task evaluation (click, follow, like); Kuaishou competition benchmark
- **Dataset (internal, non-sequence):** ~1,000 features, industry-scale samples
- **Dataset (internal, with sequences):** ~2,000 features across 9 event sources, 50B–100B samples
- **Dataset (feature selection):** ~2,000 features selected from a 12,000-feature pool
- **Hardware:** 128 and 1,024 NVIDIA A100 GPUs; FSDP + TorchRec hybrid parallelism
- **Optimizer:** Not specified; BCE loss for main objectives, correlation loss as auxiliary
- **Results:**
  - KuaiVideo: Lattice Networks outperforms or matches 10 baselines (AFN+, AutoInt+, DLRM, DCNv2, FinalMLP, MaskNet, xDeepFM, BST, APG, Wukong) on 7 of 8 metrics
  - Internal non-sequence: up to 1.14% relative loss improvement over baseline
  - Internal mixed-sequence: up to 0.68% relative improvement across multiple objectives
  - Portfolio consolidation: 1.5× FLOPs savings with 1.04× parameter reduction
  - Lattice Sketch: up to 1.6× better training throughput vs. expert-tuned baseline
  - KTAP: 1.3× improvement in knowledge transfer efficiency vs. soft-label distillation
  - Production (A/B): 10% revenue gain, 11.5% user satisfaction uplift, 6% conversion rate boost, 20% capacity saving

## Ablation Attribution (Production)

| Component | Share of Production Gain |
|---|---|
| Lattice Partitioner | 36% |
| Lattice Networks | 23% |
| Lattice KTAP | 17% |
| Lattice Filter | 13% |
| Lattice Zipper | 11% |

## Applicability

Meta Lattice is designed for **large-scale recommender system engineers and researchers** who manage multi-domain, multi-objective ads ranking models. The framework is most applicable when:

- The system trains more than ~10 separate domain-objective models, creating infrastructure overhead.
- Different objectives have mismatched feedback delays (e.g., click within seconds vs. purchase within days).
- Feature selection must be jointly maintained across tasks without penalizing minority objectives.
- Serving latency constraints limit teacher model size, making knowledge distillation attractive.

> [!IMPORTANT]
> The authors explicitly note that some components (especially Lattice Partitioner's grouping decisions) depend on Meta-specific business requirements and partner relationships. Direct replication requires contextual adaptation to the target organization's portfolio structure.
