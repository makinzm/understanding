# Meta Information

- URL: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint arXiv:2002.05202.

# GLU Variants Improve Transformer

## Overview

This paper proposes replacing the standard ReLU-activated feed-forward sublayer in Transformer models with Gated Linear Unit (GLU) variants using different nonlinear activation functions. The approach is motivated by empirical observation: gating mechanisms that multiply two linear projections (one activated, one linear) tend to yield better training perplexity than standard two-projection FFN layers, without increasing the effective parameter count.

**Applicability:** Practitioners training Transformer-based language models who want a drop-in FFN replacement that empirically improves pre-training perplexity and downstream NLU performance. Particularly relevant for encoder-decoder models (T5-family) and likely for decoder-only models as well.

---

## Background: Standard Transformer FFN

The standard Transformer feed-forward sublayer applies two linear transformations with a ReLU nonlinearity in between:

$$\text{FFN}_{\text{ReLU}}(x, W_1, W_2) = \max(0,\, xW_1)\, W_2$$

where $x \in \mathbb{R}^{n \times d_{\text{model}}}$ is the input sequence, $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, and $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$.

Variants using GELU and Swish activations were also proposed prior to this work:

$$\text{FFN}_{\text{GELU}}(x, W_1, W_2) = \text{GELU}(xW_1)\, W_2$$

$$\text{FFN}_{\text{Swish}}(x, W_1, W_2) = \text{Swish}_1(xW_1)\, W_2$$

---

## Gated Linear Units (GLU) and Variants

### Original GLU

Dauphin et al. (2016) defined GLU as a component-wise product of two linear projections, where one projection is sigmoid-activated:

$$\text{GLU}(x, W, V, b, c) = \sigma(xW + b) \odot (xV + c)$$

where $\odot$ denotes element-wise multiplication, $W, V \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, and $b, c \in \mathbb{R}^{d_{ff}}$.

### New Variants

This paper proposes four new activation functions derived by replacing sigmoid with other nonlinearities:

| Name      | Formula                                              | Activation    |
|-----------|------------------------------------------------------|---------------|
| Bilinear  | $(xW + b) \odot (xV + c)$                           | None (linear) |
| ReGLU     | $\max(0,\, xW + b) \odot (xV + c)$                 | ReLU          |
| GEGLU     | $\text{GELU}(xW + b) \odot (xV + c)$               | GELU          |
| SwiGLU    | $\text{Swish}_\beta(xW + b) \odot (xV + c)$        | Swish         |

### FFN Formulations with GLU Variants

When integrated into Transformer FFN layers, a third projection matrix $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ is added after the gating product (biases omitted for clarity):

$$\text{FFN}_{\text{GLU}}(x, W, V, W_2) = (\sigma(xW) \odot xV)\, W_2$$

$$\text{FFN}_{\text{Bilinear}}(x, W, V, W_2) = (xW \odot xV)\, W_2$$

$$\text{FFN}_{\text{ReGLU}}(x, W, V, W_2) = (\max(0,\, xW) \odot xV)\, W_2$$

$$\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \odot xV)\, W_2$$

$$\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \odot xV)\, W_2$$

> [!IMPORTANT]
> GLU variants use **three** weight matrices ($W$, $V$, $W_2$) rather than two. To maintain parameter parity with the standard two-matrix FFN, the hidden dimension is reduced from $d_{ff} = 3072$ to $d_{ff} = 2048$ (a $\frac{2}{3}$ reduction). This ensures equal total parameters between all compared models.

---

## Algorithm: Computing FFN_SwiGLU

**Input:** $x \in \mathbb{R}^{n \times d_{\text{model}}}$, weights $W, V \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$

**Output:** $y \in \mathbb{R}^{n \times d_{\text{model}}}$

```
1. gate  ← x @ W                          # shape: (n, d_ff)
2. value ← x @ V                          # shape: (n, d_ff)
3. activated_gate ← Swish_1(gate)         # element-wise: gate * sigmoid(gate)
4. hidden ← activated_gate ⊙ value        # element-wise product, shape: (n, d_ff)
5. y ← hidden @ W_2                       # shape: (n, d_model)
6. return y
```

Where $\text{Swish}_1(z) = z \cdot \sigma(z)$ (Swish with $\beta = 1$).

---

## Comparison with Related Methods

| Method                    | Activation Gate    | # Matrices | Gate × Value? | Notes                          |
|---------------------------|--------------------|------------|---------------|--------------------------------|
| FFN_ReLU (standard)       | ReLU               | 2          | No            | Transformer baseline           |
| FFN_GELU                  | GELU               | 2          | No            | Prior variant                  |
| FFN_Swish                 | Swish              | 2          | No            | Prior variant                  |
| FFN_GLU                   | Sigmoid            | 3          | Yes           | Original GLU by Dauphin et al. |
| FFN_Bilinear              | None               | 3          | Yes           | Linear gating                  |
| FFN_ReGLU                 | ReLU               | 3          | Yes           | New variant                    |
| FFN_GEGLU                 | GELU               | 3          | Yes           | New variant (best perplexity)  |
| FFN_SwiGLU                | Swish              | 3          | Yes           | New variant (used in PaLM, LLaMA) |

> [!NOTE]
> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." — Shazeer (2020)

---

# Experiments

## Dataset

- **Pre-training:** C4 (Colossal Clean Crawled Corpus) — a large English web-text corpus used by T5. The specific task is **span-filling** (masked language modeling), with 512 input tokens and 114 output tokens per sequence.
- **Fine-tuning evaluation:**
  - **GLUE** — General Language Understanding Evaluation benchmark (9 tasks: CoLA, SST-2, MRPC, STS-B, QQP, MNLI-m, MNLI-mm, QNLI, RTE)
  - **SuperGLUE** — Harder NLU benchmark (8 tasks: BoolQ, CB, CoPA, MultiRC, ReCoRD, RTE, WiC, WSC)
  - **SQuAD v1.1** — Reading comprehension (Exact Match and F1)

## Model Architecture

- 12 encoder layers, 12 decoder layers
- $d_{\text{model}} = 768$, $h = 12$ attention heads, $d_k = d_v = 64$
- Standard FFN: $d_{ff} = 3072$; GLU variants: $d_{ff} = 2048$

## Training Configuration

- **Optimizer:** Adafactor with inverse-square-root learning rate schedule
- **Pre-training steps:** 524,288 (with linear decay over final 10%)
- **Batch size:** 128 examples per step
- **Hardware:** 32-core TPUv2 cluster (~0.15 seconds/step)
- **Pre-training dropout:** None
- **Fine-tuning steps:** 131,072 at learning rate $10^{-3}$, dropout 0.1

## Key Pre-training Results (Heldout Log-Perplexity on C4)

| Method          | 65,536 Steps     | 524,288 Steps |
|-----------------|------------------|---------------|
| FFN_ReLU        | 1.997 ± 0.005    | 1.677         |
| FFN_GELU        | 1.983 ± 0.005    | 1.679         |
| FFN_Swish       | 1.994 ± 0.003    | 1.683         |
| FFN_GLU         | 1.982 ± 0.006    | 1.663         |
| FFN_Bilinear    | 1.960 ± 0.005    | 1.648         |
| FFN_ReGLU       | 1.953 ± 0.003    | 1.645         |
| **FFN_GEGLU**   | **1.942 ± 0.004**| **1.633**     |
| **FFN_SwiGLU**  | **1.944 ± 0.010**| **1.636**     |

All GLU variants outperform the ReLU baseline at both checkpoints. GEGLU and SwiGLU achieve the best perplexity at the end of pre-training.

> [!IMPORTANT]
> Despite SwiGLU's theoretical similarity to GEGLU (both use smooth activation gates), SwiGLU has become the dominant choice in large-scale models (PaLM, LLaMA, Mistral) possibly due to computational efficiency of the Swish function.

---

## Impact and Adoption

SwiGLU has become the de facto FFN activation for modern large language models:
- **PaLM** (Chowdhery et al., 2022): Uses SwiGLU in all FFN layers
- **LLaMA** (Touvron et al., 2023): Uses SwiGLU with $d_{ff} = \frac{2}{3} \times 4 d_{\text{model}}$
- **Mistral**, **Gemma**, **Qwen**, and many others: All use SwiGLU-based FFN

The key insight is that element-wise gating allows the network to selectively suppress or amplify each hidden dimension based on the input, acting as a learned, input-dependent nonlinearity rather than a fixed one.
