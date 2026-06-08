# Meta Information

- URL: [Lite Transformer with Long-Short Range Attention](https://arxiv.org/abs/2004.11886)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wu, Z., Liu, Z., Lin, J., Lin, Y., & Han, S. (2020). Lite Transformer with Long-Short Range Attention. ICLR 2020.

# Lite Transformer with Long-Short Range Attention

## Overview

Lite Transformer is a mobile-optimized NLP architecture designed for inference on edge devices (phones, embedded systems) under strict computation and memory budgets. The key observation is that standard Transformer self-attention attends both locally and globally but does so inefficiently: some attention heads focus on adjacent tokens (local context) while others attend to distant tokens (global context), leading to redundant computation. The paper proposes **Long-Short Range Attention (LSRA)**, which explicitly assigns two parallel specialized branches — one for local context via convolution and one for long-range dependencies via attention — to replace the single monolithic multi-head self-attention.

**Who uses this:** NLP practitioners deploying models on mobile devices or resource-constrained environments (smartphone assistants, IoT sensors). **When:** When latency, energy consumption, or memory footprint are primary constraints while accuracy loss must be minimized. **Where:** Translation, summarization, language modeling, and other sequence-to-sequence tasks on edge hardware.

## Problem Setting

- **Input:** Token sequence $X = (x_1, x_2, \ldots, x_n)$ where $x_i \in \mathbb{R}^d$; sequence length $n$, model dimension $d$.
- **Output:** Contextual representations $Z = (z_1, z_2, \ldots, z_n)$ where $z_i \in \mathbb{R}^d$, replacing the output of standard multi-head self-attention.
- **Constraint:** Total computation under 500M Mult-Adds (or 1G FLOPs) and model size under 10M parameters, evaluated on sequences of 30 tokens.

## Bottleneck Analysis: Why Standard Transformer Design Is Suboptimal for Mobile

Standard Transformers use bottleneck projections in both attention ($d \to d/h$ per head) and the Feed-Forward Network ($d \to 4d \to d$). In 2D/3D tasks (images, video), the bottleneck makes sense because spatial height/width dimensions are large. For 1D NLP sequences, the authors show this is suboptimal via FLOPs analysis:

For a standard multi-head attention with model dim $d$ and $h$ heads on sequence length $n$:

$$\text{FLOPs}_{\text{attn}} = 4nd^2 + 2n^2d$$

The FFN layer with a $4\times$ hidden expansion contributes:

$$\text{FLOPs}_{\text{FFN}} = 8nd^2$$

So FFN uses $2\times$ more compute than attention for typical short sequences (when $n \ll d$). The paper proposes removing the bottleneck from FFN ("flattening") to give each feature dimension more compute budget.

> [!NOTE]
> The bottleneck analysis reveals that in mobile settings (small $n$, small $d$), the FFN dominates computation. Reducing the FFN hidden factor from $4\times$ to $1\times$ (flattening) and redistributing compute to attention yields better efficiency tradeoffs.

## Long-Short Range Attention (LSRA)

### Architecture

LSRA replaces the standard $h$-head self-attention with two parallel branches that split the $d$-dimensional channel into two halves:

```
Input X ∈ ℝ^(n × d)
         │
    ┌────┴────┐
    │         │
  Left       Right
 Branch     Branch
  (d/2)      (d/2)
  Attn       Conv
    │         │
    └────┬────┘
       Concat → Z ∈ ℝ^(n × d)
```

**Left branch — global (long-range) attention:**
- Uses $h/2$ standard attention heads on the first $d/2$ channels
- Input: $X_L = X W_L^Q \in \mathbb{R}^{n \times d/2}$; similarly for keys and values
- Captures long-range dependencies across the full sequence

**Right branch — local (short-range) convolution:**
- Uses a linear projection followed by depth-wise convolution on the last $d/2$ channels
- Input: $X_R \in \mathbb{R}^{n \times d/2}$ projected to $\mathbb{R}^{n \times d/2}$
- Applies a 1D depth-wise convolution with kernel size $k$ (evaluated: 3, 5, 7, 31×3 for translation tasks)
- Captures local patterns (n-gram features, local syntax)
- Output: $Y_R = \text{DWConv}_k(X_R W_R) \in \mathbb{R}^{n \times d/2}$

**Fusion:**
$$Z = \text{Concat}(Y_L, Y_R) \in \mathbb{R}^{n \times d}$$

### Pseudocode for LSRA Forward Pass

```
LSRA(X: [n, d]):
    # Split channels
    X_L, X_R = X[:, :d/2], X[:, d/2:]   # [n, d/2] each

    # Left branch: multi-head attention (h/2 heads)
    Q = X_L @ W_Q  # [n, d/2]
    K = X_L @ W_K  # [n, d/2]
    V = X_L @ W_V  # [n, d/2]
    Y_L = MultiHeadAttn(Q, K, V, heads=h/2)  # [n, d/2]

    # Right branch: linear + depthwise conv
    H = X_R @ W_R    # [n, d/2], linear projection
    Y_R = DWConv(H, kernel_size=k)  # [n, d/2], local context
    Y_R = LayerNorm(Y_R)

    # Fuse
    return Concat(Y_L, Y_R)  # [n, d]
```

### Motivation: Attention Head Specialization

The authors visualize attention weights in trained standard Transformers and observe that heads divide naturally into two groups:
1. **Local heads**: Attention weight concentrated near the diagonal (adjacent tokens), i.e., $A_{ij} \approx 0$ for $|i-j| > k$.
2. **Global heads**: Diffuse attention patterns across the full sequence.

LSRA makes this specialization explicit: the convolution branch handles all local patterns efficiently with $O(nk \cdot d/2)$ FLOPs (versus $O(n^2 \cdot d/2)$ for an equivalent attention head), freeing the attention branch to focus exclusively on global dependencies.

## Full Lite Transformer Block

The Lite Transformer replaces each standard Transformer encoder/decoder block with:

```
Input X
  │
  ├─ LSRA(X) ──────────── (replaces multi-head self-attention)
  │     │
  │   Residual + LayerNorm
  │
  ├─ FFN(X)  ─────────────  (flattened FFN: d → d → d, no 4× expansion)
  │     │
  │   Residual + LayerNorm
  │
Output Z
```

> [!IMPORTANT]
> The FFN hidden dimension is kept equal to $d$ (not $4d$ as in standard Transformers). This "flat FFN" reduces FFN FLOPs by $4\times$ compared to the standard design, freeing compute for LSRA's dual branches.

## Comparison with Similar Architectures

| Method | Local Context | Global Context | Search Cost | Params |
|--------|-------------|--------------|-------------|--------|
| Standard Transformer | MHA (implicit) | MHA (implicit) | — | high |
| Lightweight Conv (Wu et al., 2019) | DWConv | DWConv | — | medium |
| Dynamic Conv (Wu et al., 2019) | Dynamic DWConv | Dynamic DWConv | — | medium |
| Evolved Transformer (So et al., 2019) | Mixed (NAS) | Mixed (NAS) | 250 GPU-years | medium |
| **Lite Transformer (LSRA)** | **DWConv (explicit)** | **Attn (explicit)** | **None** | **low** |

**vs. Standard Transformer:** Lite Transformer removes the bottleneck from FFN and explicitly separates local/global processing, reducing total FLOPs by ~$2\times$ at similar BLEU.

**vs. Lightweight/Dynamic Convolution:** These methods replace attention entirely with convolution, losing long-range modeling capacity. LSRA retains true attention for global dependencies.

**vs. Evolved Transformer:** Achieves +0.5 BLEU on WMT'14 En-Fr at 100M Mult-Adds while eliminating the 250+ GPU-year architecture search that produced 626,155 lbs of CO₂ emissions.

## Experiments

### Datasets

| Task | Dataset | Language Pair / Domain | Train Size |
|------|---------|----------------------|------------|
| Machine Translation | IWSLT'14 | German→English | 160K sentence pairs |
| Machine Translation | WMT'14 | English→German | 4.5M sentence pairs |
| Machine Translation | WMT'14 | English→French | 36M sentence pairs |
| Abstractive Summarization | CNN/DailyMail | English news | ~280K article-summary pairs |
| Language Modeling | WIKITEXT-103 | English Wikipedia | 103M tokens |

### Hardware

- Training: 16 NVIDIA RTX 2080Ti GPUs (large models)
- Inference evaluation: measured in Mult-Adds (not GPU-specific)

### Optimizer

- Adam optimizer
- Cosine learning rate scheduling
- Dropout: 0.3 (WMT/IWSLT), 0.1 (language modeling)
- Label smoothing: 0.1
- Gradient accumulation over 8 batches

### Results

**Machine Translation — IWSLT'14 De→En:**
- Lite Transformer (2.8M params, 63M Mult-Adds): +3.1 BLEU vs. Transformer baseline of same size

**Machine Translation — WMT'14 En-De:**
- Under 100M Mult-Adds: +1.2 BLEU over Transformer
- Under 500M Mult-Adds: +0.4 BLEU over Transformer

**Machine Translation — WMT'14 En-Fr:**
- Under 100M Mult-Adds: +1.7 BLEU over Transformer
- Under 500M Mult-Adds: +1.2 BLEU over Transformer

**Abstractive Summarization — CNN/DailyMail:**
- $2.5\times$ less computation than standard Transformer at matching R-1/R-2/R-L F1 scores

**Language Modeling — WIKITEXT-103:**
- 1.8 lower perplexity than Transformer at ~500M Mult-Adds
- Throughput: 10,200 vs. 7,600 tokens/second (35% speedup)

**Model Compression:**
- Combined with pruning and quantization: 18.2× model size compression with negligible quality loss, enabling on-device deployment

## Code

Source code is available at [https://github.com/mit-han-lab/lite-transformer](https://github.com/mit-han-lab/lite-transformer).
