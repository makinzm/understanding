# Meta Information

- URL: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

# LoRA: Low-Rank Adaptation of Large Language Models

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large pre-trained language models. It freezes all pre-trained model weights and injects pairs of trainable rank-decomposition matrices into each Transformer layer. The key insight is that the weight update $\Delta W$ during adaptation has a low *intrinsic rank* — meaning that the semantically meaningful update directions lie in a much smaller subspace than the full weight matrix dimensions suggest.

**Who uses this**: ML practitioners adapting large pre-trained language models (GPT-3, RoBERTa, DeBERTa, etc.) to downstream tasks with limited GPU memory or when serving many task-specific variants simultaneously.

**When**: LoRA is applicable whenever full fine-tuning is computationally infeasible due to model size or when fast task-switching between multiple fine-tuned models is needed.

## Problem Statement

Given a pre-trained autoregressive language model $P_{\Phi}(y \mid x)$ parameterized by $\Phi$, standard full fine-tuning maximizes:

```math
\begin{align}
\max_{\Phi} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_{\Phi}(y_t \mid x, y_{<t})\right)
\end{align}
```

where $\mathcal{Z} = \{(x_i, y_i)\}_{i=1,\ldots,N}$ is the downstream task dataset. For GPT-3 with $|\Phi| \approx 175$ billion parameters, storing a separate fine-tuned copy per task requires roughly 350 GB per checkpoint, making multi-task deployment impractical.

LoRA instead encodes the task-specific increment $\Delta\Phi = \Delta\Phi(\Theta)$ via a much smaller parameter set $\Theta$ with $|\Theta| \ll |\Phi|$, optimizing:

```math
\begin{align}
\max_{\Theta} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{<t})\right)
\end{align}
```

For GPT-3 175B, $|\Theta|$ can be as small as 0.01% of $|\Phi_0|$.

## Motivation: Why Not Adapters or Prefix Tuning?

### Adapter Layers Introduce Inference Latency

Adapter layers insert small bottleneck MLP modules (e.g., $d_{\text{model}} \to r \to d_{\text{model}}$) sequentially into the Transformer. Because they cannot be merged into the frozen weights, they add sequential operations at inference time. On GPT-2 medium with batch size 1 and sequence length 128:

| Method    | Latency (ms)   | Overhead |
|-----------|---------------|----------|
| Fine-Tune / LoRA | 19.8 ± 2.7 | — |
| AdapterL  | 23.9 ± 2.1  | +20.7%   |
| AdapterH  | 25.8 ± 2.2  | +30.3%   |

Under model parallelism, adapter layers require additional synchronous GPU operations (AllReduce, Broadcast), amplifying latency further.

> [!NOTE]
> AdapterH is, defined 5.1 Chapter, adding two adapter layers per Transformer layer, which one is before and one is after the attention/MLP, which H may mean Houlsby, who is this architect of the adapter.
>
> AdapterL is, defined 5.1 Chapter, a single adapter layer after Transformer, which L may mean Lin, who is this architect of the adapter.

### Prefix Tuning Has Non-Monotonic Scaling

Prefix tuning prepends learnable tokens to the input sequence. However, it non-monotonically responds to increasing the number of prefix tokens, and the reserved prefix positions reduce the effective sequence length available for downstream task content.

## Method: Low-Rank-Parametrized Update Matrices

### Core Formula

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constrains the weight update as:

$$W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

### Forward Pass

**Input**: $x \in \mathbb{R}^{k}$ (or a batch $x \in \mathbb{R}^{n \times k}$)

**Output**: $h \in \mathbb{R}^{d}$

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

The frozen weight $W_0$ and the trainable low-rank product $BA$ are summed, so no extra inference latency is introduced once $W = W_0 + BA$ is pre-computed.

### Initialization

- $A$ is initialized with random Gaussian entries: $A \sim \mathcal{N}(0, \sigma^2)$
- $B$ is initialized to zero

This ensures $\Delta W = BA = 0$ at the start of training, so LoRA starts from the exact pre-trained behavior.

### Scaling

The update is scaled by $\frac{\alpha}{r}$ before addition, $x$ is the input to the linear layer, and $h$ is the output:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $\alpha$ is a constant (typically set to the initial rank value). Fixing $\alpha$ instead of tuning it as a hyperparameter is equivalent to adjusting the learning rate, simplifying the sweep.

### Pseudocode

```
# LoRA forward pass (single linear layer)
# W0: frozen pre-trained weight [d x k]
# A:  trainable matrix [r x k], init ~ N(0, σ²)
# B:  trainable matrix [d x r], init = 0
# alpha, r: scalar hyperparameters

def lora_linear(x, W0, A, B, alpha, r):
    base = W0 @ x             # frozen path: [d x k] @ [k] -> [d]
    lora = B @ (A @ x)        # low-rank path: [d x r] @ ([r x k] @ [k]) -> [d]
    return base + (alpha / r) * lora

# At deployment (merge weights, zero latency overhead):
W_merged = W0 + (alpha / r) * (B @ A)   # [d x k]
```

## Applying LoRA to the Transformer

LoRA is applied to the attention weight matrices of each Transformer layer. Given $d_{\text{model}}$, the four attention projection matrices are:

| Matrix | Shape | Applied LoRA? |
|--------|-------|---------------|
| $W_q$  | $d_{\text{model}} \times d_{\text{model}}$ | Yes (primary) |
| $W_k$  | $d_{\text{model}} \times d_{\text{model}}$ | Sometimes |
| $W_v$  | $d_{\text{model}} \times d_{\text{model}}$ | Yes (primary) |
| $W_o$  | $d_{\text{model}} \times d_{\text{model}}$ | Sometimes |
| MLP weights | — | Frozen (not trained) |

The number of trainable parameters is:

$$|\Theta| = 2 \times \hat{L}_{\text{LoRA}} \times d_{\text{model}} \times r$$

where $\hat{L}_{\text{LoRA}}$ is the number of layers with LoRA applied.

> [!IMPORTANT]
> The MLP modules are frozen entirely. Only attention projection matrices receive LoRA adapters. This choice is motivated by empirical evidence (Section 7) that $W_q$ and $W_v$ together provide more benefit than other combinations.

### Memory Benefits for GPT-3 175B

| Metric | Full Fine-tuning | LoRA ($r=4$, $W_q$+$W_v$) |
|--------|-----------------|--------------------------|
| VRAM during training | 1.2 TB | 350 GB (−71%) |
| Checkpoint size | 350 GB | 35 MB (−10,000×) |
| Training throughput | baseline | +25% faster |

## Comparison with Similar Methods

| Method | Inference Latency | # Trainable Params | Task-Switch Cost | Sequence Length | Merges into Weights |
|--------|------------------|-------------------|-----------------|-----------------|---------------------|
| Full Fine-tuning | None | $|\Phi|$ (all) | High (swap full model) | Full | N/A |
| Adapter (AdapterH/L) | +20–30% | Small | Low (swap adapters) | Full | No |
| Prefix Tuning | None | Small | Low | Reduced | No |
| **LoRA** | **None** | **Very small** | **Low (swap A,B)** | **Full** | **Yes** |

> [!NOTE]
> "Unlike adapters, no additional inference latency" is LoRA's key advantage — $W = W_0 + BA$ can be pre-computed and used as a single merged weight matrix.

## Empirical Analysis: Intrinsic Rank

### Which Weight Matrices Should Be Adapted? (Section 7.1)

With a fixed parameter budget of 18M for GPT-3, different allocations of $r$ across matrices were compared. The finding: **adapting $W_q$ and $W_v$ simultaneously at $r=4$ outperforms adapting only $W_q$ at $r=8$**. Spreading parameters broadly across multiple matrices is more effective than concentrating on a single one.

### How Small Can $r$ Be? (Section 7.2)

Experiments on GPT-3 WikiSQL and MultiNLI:

| Rank $r$ | WikiSQL Acc. | MultiNLI Acc. |
|----------|-------------|--------------|
| 1 | 70.4 | 91.0 |
| 2 | 73.4 | 91.2 |
| 4 | 73.0 | 91.3 |
| 8 | 73.8 | 91.6 |
| 64 | 74.0 | 91.6 |

> [!NOTE]
> "ΔW has a very small 'intrinsic rank'." Surprisingly, $r=1$ recovers most of the benefit, and increasing $r$ beyond 4–8 yields diminishing returns.

### Subspace Similarity Analysis (Section 7.3)

To confirm the low-rank hypothesis, the authors compute the normalized subspace similarity between $\Delta W_{r=8}$ and $\Delta W_{r=64}$ using the top-$i$ singular vectors. For $W_q$, the similarity between the top singular directions is 0.67 (top-1) and 0.54 (top-2), while for $W_v$ it is even higher. This shows that **the most important update directions are shared even across very different rank choices**.

> [!NOTE]
> "There is a non-trivial similarity between the two, especially for $\Delta W_q$," suggesting the fine-tuning signal truly lives in a low-dimensional subspace.

## Experiments

### Datasets

| Dataset | Task | Train Size | Dev/Test Size |
|---------|------|-----------|--------------|
| GLUE (MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B) | NLU Classification | Varies per task | Varies |
| WikiSQL | NL-to-SQL generation | 56,355 | 8,421 (test) |
| SAMSum | Dialogue summarization | 14,732 | 819 (test) |
| E2E NLG Challenge | Data-to-text (restaurant domain) | ~42,000 | 4,600 / 4,600 |
| DART | Open-domain data-to-text | ~66,000 | ~8,000 |
| WebNLG | Data-to-text (14 categories) | ~13,000 | ~2,000 |

- **Hardware**: NVIDIA Tesla V100

### Key Results

**RoBERTa / DeBERTa on GLUE** (avg score):

| Model | Method | # Trainable Params | Avg GLUE |
|-------|--------|-------------------|---------|
| RoBERTa base | Full FT | 125M | 86.4 |
| RoBERTa base | LoRA | 0.3M | **87.2** |
| RoBERTa large | Full FT | 355M | 88.9 |
| RoBERTa large | LoRA | 0.8M | **89.0** |
| DeBERTa XXL | Full FT | 1500M | 91.1 |
| DeBERTa XXL | LoRA | 4.7M | **91.3** |

**GPT-2 on E2E NLG (BLEU)**:

| Model | Method | # Trainable Params | BLEU |
|-------|--------|-------------------|------|
| GPT-2 Medium | Full FT | 354.92M | 68.2 |
| GPT-2 Medium | LoRA | 0.35M | **70.4** |
| GPT-2 Large | Full FT | 774.03M | 68.5 |
| GPT-2 Large | LoRA | 0.77M | **70.4** |

**GPT-3 175B**:

| Dataset | Full FT | LoRA (4.7M) | LoRA (37.7M) |
|---------|---------|------------|-------------|
| WikiSQL (Acc.) | 73.8% | 73.4% | **74.0%** |
| MultiNLI (Acc.) | 89.5% | **91.7%** | 91.6% |
| SAMSum (R-1/R-2/R-L) | 52.0/28.0/44.5 | **53.8/29.8/45.9** | 53.4/29.2/45.1 |

> [!IMPORTANT]
> LoRA matches or exceeds full fine-tuning across all evaluated models and tasks, with 100–10,000× fewer trainable parameters.

## Limitations

- **No batched multi-task inference** when $A$ and $B$ are merged into $W$: if $W = W_0 + BA$ is pre-computed, all requests in a batch must use the same task adapter.
- **Rank is a global hyperparameter**: the same rank $r$ is applied to all selected weight matrices, even if different matrices may benefit from different ranks.
- Empirical analysis suggests $W_q$ and $W_v$ are most important, but this may not generalize to all architectures.
