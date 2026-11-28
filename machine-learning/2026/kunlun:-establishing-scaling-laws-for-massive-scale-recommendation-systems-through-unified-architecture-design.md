# Meta Information

- URL: [Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design](https://arxiv.org/abs/2602.10016)
- LICENSE: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Hou, B., Liu, X., Liu, X., Xu, J., Badr, Y., Hang, M., Chanpuriya, S., et al. (2026). Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design. arXiv:2602.10016.

# Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems

## Overview

Kunlun is a unified recommender system architecture developed at Meta Ads that establishes **predictable power-law scaling** for click-through rate (CTR) prediction at production scale. The paper addresses why large recommendation models—unlike LLMs—fail to exhibit smooth scaling curves, and proposes a series of architectural innovations that together double scaling efficiency over the prior state of the art (InterFormer).

The target audience is ML engineers and researchers working on **industrial-scale recommender systems** where both model quality (Normalized Entropy, NE) and serving efficiency (QPS, MFU) must be co-optimized. The system is deployed in major Meta Ads models handling 70B+ training samples.

## 1. Introduction

Scaling laws (power laws) predict model performance as a function of compute:

$$\text{NE}(C) = \text{NE}_0 - \eta \cdot \log\!\left(\frac{C}{C_0}\right)$$

where $C$ is total FLOPs, $\text{NE}_0$ is baseline Normalized Entropy, and $\eta$ is the **scaling coefficient**. A larger $\eta$ indicates faster quality improvement per unit of additional compute.

Scaling efficiency is defined as:

$$\text{Scaling Efficiency} = \frac{\eta}{\eta_{\text{baseline}}} = \frac{\Delta \text{NE} / \log(C/C_0)}{\eta_{\text{baseline}}}$$

> [!NOTE]
> "Poor scaling efficiency as the main barrier to predictable power-law scaling, stemming from inefficient modules with low Model FLOPs Utilization (MFU)."

Recommendation systems achieve only **3–15% MFU**, far below the 40–60% typical of LLMs, which breaks the compute-quality relationship and prevents predictable scaling.

## 2. Related Work

| Method | Focus | Limitation |
|---|---|---|
| **Wukong** | Non-sequential feature interaction via stacked Wukong layers (Kronecker-structured FFN + feature crossing) | No sequence modeling; limited scaling at massive scales |
| **InterFormer** | Joint sequence + non-sequence modeling with bidirectional information flow | Computational bottlenecks limit scaling efficiency |
| **DIN / DIEN** | Target-aware attention / GRU for sequential user behavior | Sequence-only; ignores rich non-sequential context |
| **HSTU** | Sequence-centric Transformer with request-level optimization | Different deployment scenario; not directly comparable |
| **FlashAttention** | Kernel fusion for standard multi-head attention | Not directly applicable to recommender-specific shapes |

Kunlun's contribution is **unifying** sequence and non-sequence modeling while addressing the MFU gap between LLMs and recommenders.

## 3. Preliminaries

**Task**: CTR prediction for online advertising.

**Input**:
- Non-sequential context features: $\mathbf{x}^{(ns)} \in \mathbb{R}^{B \times F_{ns} \times d}$ where $B$ is batch size, $F_{ns}$ is the number of non-sequential feature fields, $d$ is embedding dimension.
- Sequential behavior features: $\mathbf{x}^{(s)} \in \mathbb{R}^{B \times T \times d}$ where $T$ is sequence length (100–1000+).

**Output**: Click probability $\hat{y} \in [0, 1]$ per sample.

**Primary metric**: Normalized Entropy (NE), measuring calibration quality. Lower NE = better prediction.

**Efficiency metrics**:
- **MFU** (Model FLOPs Utilization): fraction of theoretical peak FLOP throughput utilized.
- **QPS** (Queries Per Second): inference throughput.

## 4. Method

### 4.1 Architecture Overview

Kunlun stacks multiple **Kunlun layers**, each containing:
1. **Feature Preprocessing** (ROTE embeddings)
2. **Low-level module optimizations** (GDPA, HSP, Sliding Window Attention)
3. **High-level computation reallocation** (CompSkip, Event-Level Personalization, Global Interaction Module)

The architecture jointly processes sequence tokens and non-sequence context through a unified transformer-like framework, with information flowing bidirectionally between the two modalities.

### 4.2 Feature Preprocessing: Rotary Temporal Embeddings (ROTE)

Standard positional encodings ignore the actual timestamps of user events. ROTE extends **Rotary Position Embeddings (RoPE)** to incorporate temporal gaps $\Delta t$ between events using **log-scale encoding**:

$$\text{ROTE}(\Delta t) = \text{RoPE}\!\left(\log(1 + \Delta t)\right)$$

This allows the model to distinguish recent interactions (small $\Delta t$) from distant historical patterns (large $\Delta t$), which is critical for event-level recommendation quality.

> [!TIP]
> RoPE encodes positions as complex number rotations, preserving relative position information through dot products. ROTE reuses this mechanism but substitutes absolute position indices with log-scaled temporal gaps.

### 4.3 Low-Level Module Optimizations

These optimizations address the **3–15% MFU** bottleneck by redesigning individual computational kernels.

#### 4.3.1 Generalized Dot-Product Attention (GDPA)

The **Personalized Feed-Forward Network (PFFN)** in prior work treats non-sequence context as a conditioning signal for each sequence token, but its naive implementation cannot be fused with standard attention kernels.

GDPA reformulates PFFN as a **multi-head attention-style operator**:

$$\text{GDPA}_h(Q, K, V) = \text{Activation}_h\!\left(\frac{QK^\top}{\tau}\right) V$$

where:
- $Q = S^{(l)} \in \mathbb{R}^{B \times T \times d_h}$: sequence tokens at layer $l$
- $K, V \in \mathbb{R}^{B \times F_{ns} \times d_h}$: derived from non-sequential context features
- $\tau = \text{maxlen}(\text{seq})$: temperature scaling by maximum sequence length
- $h$: head index in multi-head formulation

This formulation enables **kernel fusion** analogous to FlashAttention, increasing MFU by up to **6×** for the PFFN component alone.

> [!IMPORTANT]
> The key insight is that PFFN's "personalized linear transformation" can be expressed as cross-attention between sequence tokens (queries) and context features (keys/values), making it amenable to the same tiling/fusion tricks used in standard attention.

#### 4.3.2 Hierarchical Seed Pooling (HSP)

Long sequences ($T > 1000$) make quadratic attention computationally prohibitive. HSP compresses long sequences into a fixed number of **seed vectors** through three stages:

1. **Seed Embedding Initialization**: Initialize $S$ seed vectors $\in \mathbb{R}^{B \times S \times d}$ (e.g., $S \ll T$).
2. **Seed-Level Attention**: Cross-attend seed vectors against the full sequence to aggregate information.
3. **SumKronLinear Compression**: Apply parameter-efficient compression via Kronecker decomposition:

$$Y_b = \sum_{i=1}^{k} Z_i^\top X_b W_i$$

where $X_b \in \mathbb{R}^{T \times d}$ is the sequence, $Z_i \in \mathbb{R}^{T \times r}$ and $W_i \in \mathbb{R}^{d \times d'}$ are Kronecker factors.

Parameter reduction: from $O(S \cdot d \cdot T \cdot d)$ to $O(k \cdot (S \cdot T + d^2))$, achieving **14× parameter reduction** while maintaining model expressiveness.

> [!NOTE]
> Unlike separable factorizations (which assume sequence and embedding dimensions are independent), SumKronLinear captures **joint sequence-embedding correlations**, which are critical for recommendation quality.

#### 4.3.3 Sliding Window Attention

Standard self-attention over length-$T$ sequences has $O(T^2)$ complexity. Sliding Window Attention restricts each token to attend only within a local window of size $w$:

$$\text{Attention}(Q_t, K, V) = \text{softmax}\!\left(\frac{Q_t K_{[t-w:t+w]}^\top}{\sqrt{d}}\right) V_{[t-w:t+w]}$$

Complexity: $O(Tw)$ instead of $O(T^2)$.

Effect: **31.1% QPS improvement** while preserving locality-biased temporal dependencies (recent events strongly influence current behavior).

### 4.4 High-Level Computation Reallocation

These techniques redistribute FLOPs across layers and event types based on their contribution to prediction quality.

#### 4.4.1 Computation Skip (CompSkip)

CompSkip implements an **every-other-layer alternation** pattern:
- **Even layers**: skip self-attention; run HSP + PFFN only.
- **Odd layers**: skip HSP + PFFN; run self-attention only.

Effect: ~**43.1% FLOPs reduction** with negligible NE degradation, because the two operations capture complementary information and do not need to be co-located in every layer.

**Algorithm (CompSkip)**:
```
for l in 1..L:
    if l is even:
        h = SelfAttention(h)
    else:
        h = HSP(h) + PFFN(h)
    h = LayerNorm(h + residual)
```

#### 4.4.2 Event-Level Personalization

Not all event types carry equal predictive signal. Clicks are much more informative than impressions. Kunlun allocates compute proportional to event importance:

| Event Type | Model Dimension $d$ | Attention Heads | Tokens | Layers |
|---|---|---|---|---|
| Clicks | Large | More | More | More |
| Conversions | Medium | Medium | Medium | Medium |
| Impressions | Small | Few | Fewer | Fewer |

This asymmetric allocation ensures the most informative events receive the most modeling capacity.

#### 4.4.3 Global Interaction Module

A **Mixture-of-Wukong-Experts** module handles cross-modal feature interactions between sequence and non-sequence representations:
- **Horizontal scaling**: expert parallelism (more experts = more capacity without per-sample cost increase).
- **Vertical scaling**: stacking Wukong layers increases depth.

Each Wukong expert applies Kronecker-structured feature crossing (inherited from prior Wukong architecture) to fuse the unified representation.

### 4.5 Multi-Layer Architecture

The full Kunlun model stacks $L$ Kunlun layers. Residual connections and layer normalization are applied after each sub-module. The final representation is passed to a sigmoid output head for CTR prediction.

## 5. Experiments

**Dataset**: Internal Meta Ads dataset
- Training samples: 70 billion+
- Non-sequential features: hundreds to thousands of feature fields
- Sequential features: 10+ types, lengths from hundreds to 1000+
- Event types: clicks, conversions, impressions
- Training regime: single epoch

**Hardware**: NVIDIA B200 GPUs

**Primary metric**: NE (Normalized Entropy, lower = better)

**Efficiency metrics**: MFU, QPS

### 5.1 Main Results

- **MFU**: Improved from **17% to 37%** on B200 GPUs (baseline → Kunlun)
- **Scaling efficiency**: **2× improvement** over InterFormer (state-of-the-art prior work)
- **NE at 180 GFLOPs**: Kunlun achieves **+0.79% NE gain** vs. InterFormer's +0.50%
- **Production**: **+1.2% improvement** in topline metrics at Meta Ads after deployment

### 5.2 Ablation Study

| Component | MFU Δ | QPS Δ | NE Δ |
|---|---|---|---|
| GDPA (vs. naive PFFN) | +3% | −8% | +0.04% |
| HSP (vs. PMA baseline) | −6.79% | +8.8% | +0.08% |
| Sliding Window Attention | −4.1% | +31.1% | ~0% |
| CompSkip | −2.5% | +35% | ~0% |

> [!CAUTION]
> The ablation table shows trade-offs: Sliding Window and CompSkip sacrifice some MFU to gain large QPS improvements. GDPA improves both NE and MFU at a small QPS cost. Practitioners must balance these trade-offs based on serving latency SLAs.

### 5.3 Scaling Analysis

Kunlun exhibits a smooth power-law NE curve across compute budgets from 10 to 200+ GFLOPs, confirming that fixing the MFU bottleneck restores predictable scaling. InterFormer's scaling curve flattens at large compute, whereas Kunlun's continues to improve.

## 6. Differences from Similar Algorithms

| Aspect | Wukong | InterFormer | HSTU | **Kunlun** |
|---|---|---|---|---|
| Sequence modeling | No | Yes (bidirectional) | Yes (sequence-centric) | Yes (with HSP + GDPA) |
| Non-sequence features | Yes (Kronecker FFN) | Yes | Limited | Yes (GDPA cross-attention) |
| MFU optimization | No | Partial | Yes | Full (kernel fusion) |
| Scaling efficiency | Not reported | Baseline (1×) | Not comparable | 2× over InterFormer |
| Temporal embeddings | No | No | No | Yes (ROTE) |
| Computation skip | No | No | No | Yes (CompSkip) |
| Event-level allocation | No | No | No | Yes |
| Deployment | Yes | Yes | Yes | Yes (Meta Ads) |

## Applicability Conditions

**Who**: ML engineers building large-scale CTR or ranking systems at companies with 10B+ training samples and multi-GPU serving infrastructure.

**When**: When scaling compute beyond a few GFLOPs per inference fails to yield proportional quality gains (i.e., when the scaling curve has flattened).

**Where**: Online advertising, content feed ranking, and other scenarios requiring joint modeling of both long user behavior sequences and rich non-sequential context features.

Kunlun's techniques (ROTE, GDPA, HSP, CompSkip, Event-Level Personalization) can be adopted individually or together. CompSkip and Sliding Window Attention are particularly easy to adopt as drop-in replacements for standard self-attention + PFFN stacks.
