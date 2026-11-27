# Meta Information

- URL: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. arXiv preprint arXiv:2006.03654.

```bibtex
@article{he2020deberta,
  title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
  author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
  journal={arXiv preprint arXiv:2006.03654},
  year={2020},
  url={https://arxiv.org/abs/2006.03654}
}
```

# Introduction

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) is a pretrained transformer language model that improves upon BERT and RoBERTa through two architectural innovations: **disentangled attention** and an **enhanced mask decoder (EMD)**. Developed at Microsoft, DeBERTa achieves state-of-the-art results on NLU benchmarks while requiring only half the pretraining data compared to RoBERTa-large. The 1.5B parameter version of DeBERTa was the first model to surpass human-level performance on the SuperGLUE benchmark.

**Applicable when:**
- You need a strong pretrained encoder for NLU tasks (classification, QA, NER)
- You want better position-aware representations without absolute position encoding in the main transformer body
- You are working on tasks where relative positions between words matter more than absolute positions (e.g., long-document understanding)

> [!NOTE]
> BERT encodes position information by adding absolute position embeddings to token embeddings at the input layer. DeBERTa instead uses *relative* position embeddings throughout the transformer layers and only introduces absolute position information at the final decoding step.

# Disentangled Attention

## Motivation

In standard BERT/RoBERTa, each token is represented by a single embedding vector that merges content (semantic meaning) and absolute position. As a result, the attention weight between two tokens conflates their semantic relationship with their position relationship. DeBERTa separates these two aspects to allow the model to learn content-position interactions more explicitly.

## Representation

Each token $i$ at position $p$ is represented by two vectors:
- $\mathbf{H}_i \in \mathbb{R}^{d}$: the **content embedding** (semantic meaning)
- $\mathbf{P}_{i|j} \in \mathbb{R}^{d}$: the **relative position embedding** from token $i$ to token $j$, encoding $\delta(i, j)$

where the relative distance function clips to a maximum range $k$:

$$\delta(i,j) = \begin{cases} 0 & \text{if } i-j \leq -k \\ 2k-1 & \text{if } i-j \geq k \\ i-j+k & \text{otherwise} \end{cases}$$

This maps integer offsets in $[-k, k]$ to indices in $[0, 2k-1]$, giving a vocabulary of $2k$ distinct relative positions.

## Attention Score Decomposition

The full disentangled attention matrix $\tilde{A}_{i,j}$ between tokens $i$ and $j$ decomposes into four terms:

$$A_{i,j} = \mathbf{H}_i \mathbf{H}_j^T + \mathbf{H}_i \mathbf{P}_{j|i}^T + \mathbf{P}_{i|j} \mathbf{H}_j^T + \mathbf{P}_{i|j} \mathbf{P}_{j|i}^T$$

| Term | Name | Meaning |
|------|------|---------|
| $\mathbf{H}_i \mathbf{H}_j^T$ | Content-to-Content (C2C) | Semantic relationship |
| $\mathbf{H}_i \mathbf{P}_{j|i}^T$ | Content-to-Position (C2P) | How content of $i$ relates to position of $j$ relative to $i$ |
| $\mathbf{P}_{i|j} \mathbf{H}_j^T$ | Position-to-Content (P2C) | How position of $i$ relative to $j$ relates to content of $j$ |
| $\mathbf{P}_{i|j} \mathbf{P}_{j|i}^T$ | Position-to-Position (P2P) | Pure position interaction (dropped) |

The position-to-position term is dropped in practice because it does not depend on content and adds no information beyond a fixed positional prior. The implemented score is:

$$\tilde{A}_{i,j} = \frac{1}{\sqrt{3d}} \left( \mathbf{Q}^c_i (\mathbf{K}^c_j)^T + \mathbf{Q}^c_i (\mathbf{K}^r_{\delta(i,j)})^T + \mathbf{Q}^r_{\delta(j,i)} (\mathbf{K}^c_j)^T \right)$$

where:
- $\mathbf{Q}^c = \mathbf{H} \mathbf{W}_{q,c}$, $\mathbf{K}^c = \mathbf{H} \mathbf{W}_{k,c}$ are content query/key matrices ($\in \mathbb{R}^{n \times d}$)
- $\mathbf{Q}^r = \mathbf{P} \mathbf{W}_{q,r}$, $\mathbf{K}^r = \mathbf{P} \mathbf{W}_{k,r}$ are relative position query/key matrices ($\in \mathbb{R}^{2k \times d}$)
- The scaling factor $\sqrt{3d}$ accounts for three additive terms

## Memory Efficiency

A key efficiency gain: the relative position embedding table $\mathbf{K}^r \in \mathbb{R}^{2k \times d}$ is **shared across all token pairs** with the same relative distance. This reduces the space for position keys from $O(n^2 d)$ (if unique per pair) to $O(kd)$ where $k \ll n$. In practice, $k = 512$ for most experiments.

## Comparison to Prior Relative Position Methods

| Method | Approach |
|--------|----------|
| Shaw et al. (2018) | Add relative position embeddings to key vectors only (C2P term) |
| Dai et al. (Transformer-XL) | Four-component decomposition with sinusoidal relative positions |
| DeBERTa | Disentangled content/position vectors; three terms (C2C, C2P, P2C); learnable relative embeddings |

DeBERTa differs from Transformer-XL by using **learnable** relative position embeddings and by applying the factored representation at every layer independently, rather than using cross-segment recurrence.

# Enhanced Mask Decoder (EMD)

## Problem

When DeBERTa uses purely relative position embeddings throughout all transformer layers, the model cannot distinguish absolute token positions from relative ones alone. For masked language modeling (MLM), this causes ambiguity: given the sentence "[MASK] store sold sold [MASK] products", predicting the first and last masks requires knowing that one is at the beginning of the sentence and one is near the end—information that relative positions alone cannot fully provide.

## Solution

The EMD inserts a **shallow two-layer transformer decoder** after the main transformer encoder. This decoder takes as input the encoder hidden states and incorporates **absolute position embeddings** before the final softmax prediction. Formally:

1. Encoder output: $\mathbf{H}^{enc} \in \mathbb{R}^{n \times d}$ (from $L$ layers of disentangled attention)
2. EMD input: $\mathbf{I} = \mathbf{H}^{enc} + \mathbf{E}^{abs}$ where $\mathbf{E}^{abs} \in \mathbb{R}^{n \times d}$ are standard absolute position embeddings
3. EMD output: fed to the MLM prediction head

> [!IMPORTANT]
> Absolute positions are only added at the EMD stage, not at the encoder input. This means the transformer body learns purely relative positional relationships, while absolute position serves as a supplementary signal only during masked token prediction.

## Effect on Downstream Tasks

Although the EMD is designed for the MLM pretraining objective, it improves downstream fine-tuning as well (ablation shows ~0.2-1.4% improvement across tasks). The absolute position information learned during pretraining through the EMD helps the model implicitly encode positional context even after removing the EMD head at fine-tuning time.

# Scale-Invariant Fine-Tuning (SiFT)

For the 1.5B parameter DeBERTa model, the authors introduce SiFT (Scale-Invariant Fine-Tuning), a virtual adversarial training technique. Standard virtual adversarial training perturbs word embeddings:

$$\tilde{\mathbf{e}}_i = \mathbf{e}_i + \epsilon \cdot \mathbf{r}_i$$

where $\mathbf{r}_i$ is an adversarial perturbation. SiFT first **normalizes** the word embeddings to unit variance before applying perturbations, making the perturbation magnitude scale-invariant across embedding dimensions. This stabilizes training for large models where embedding magnitudes vary significantly across vocabulary items.

# Architecture and Pretraining

## Model Configurations

| Model | Layers | Hidden | Heads | Params | Pretraining Data |
|-------|--------|--------|-------|--------|-----------------|
| DeBERTa-base | 12 | 768 | 12 | ~140M | 78 GB |
| DeBERTa-large | 24 | 1024 | 16 | ~390M | 78 GB |
| DeBERTa-1.5B | 48 | 1536 | 24 | 1.5B | 160 GB |

## Pretraining Data (78 GB corpus for base/large)

| Source | Size |
|--------|------|
| Wikipedia | 12 GB |
| BookCorpus | 6 GB |
| OpenWebText | 38 GB |
| CommonCrawl Stories | 31 GB (filtered) |

The 1.5B model additionally uses a 160 GB corpus (same as RoBERTa's training data scale) and a 128K token vocabulary.

## Training Hyperparameters

| Setting | Base | Large |
|---------|------|-------|
| Batch size | 2,048 | 2,000 |
| Learning rate | 2e-4 | 2e-4 |
| Steps | 1M | 1M |
| Warmup steps | 10,000 | 10,000 |
| Hardware | 64 V100 (4×DGX-2) | 96 V100 (6×DGX-2) |

# Experiments

## Datasets

- **GLUE** (General Language Understanding Evaluation): 9 tasks including CoLA (linguistic acceptability), SST-2 (sentiment), MNLI (NLI, 393K train), QQP (paraphrase), QNLI (QA-NLI), RTE, MRPC, STS-B, WNLI
- **SuperGLUE**: 8 more difficult tasks including BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC
- **SQuAD v1.1**: 87,599 train / 10,570 dev extractive QA examples
- **SQuAD v2.0**: 130,319 total examples (including unanswerable questions)
- **RACE**: 87,866 train / 4,887 dev reading comprehension (multiple choice)
- **CoNLL 2003 NER**: 14,987 train / 3,466 dev / 3,684 test named entity recognition
- **ReCoRD**: 101K train reading comprehension with commonsense reasoning
- **SWAG**: 73,546 train grounded inference (sentence completion)
- **Wikitext-103**: Language modeling benchmark (NLG evaluation)

## Key Results

**DeBERTa-large vs. RoBERTa-large** (trained on half the data):

| Task | RoBERTa-large | DeBERTa-large | Gain |
|------|--------------|--------------|------|
| MNLI-m (Acc) | 90.2 | 91.1 | +0.9 |
| SQuAD v2.0 (F1) | 89.8 | 92.1 | +2.3 |
| RACE (Acc) | 83.2 | 86.8 | +3.6 |
| CoNLL NER (F1) | 92.4 | 93.0 | +0.6 |

**SuperGLUE (DeBERTa-1.5B single model):** 89.9% — first model to surpass human performance (89.8%)

## Ablation Study

Removing each component from DeBERTa-base:

| Ablation | MNLI-m | SQuAD 1.1 F1 | SQuAD 2.0 F1 | RACE |
|----------|--------|--------------|--------------|------|
| Full DeBERTa-base | 86.3 | 92.1 | 82.5 | 71.7 |
| −EMD | 86.1 | 91.8 | 81.3 | 70.3 |
| −C2P | 85.9 | 91.6 | 81.3 | 69.3 |
| −P2C | 86.0 | 91.7 | 80.8 | 69.6 |
| −(EMD+C2P) | 85.8 | 91.5 | 80.3 | 68.1 |
| RoBERTa-ReImp-base | 84.9 | 91.1 | 79.5 | 66.8 |

Both attention components (C2P and P2C) and the EMD contribute independently. Removing C2P has the largest single impact on RACE (+2.4% when present).

# Comparison with Similar Models

| Model | Position Encoding | Absolute Pos | Relative Pos | Key Difference |
|-------|-----------------|--------------|--------------|----------------|
| BERT | Input embedding | ✓ (learned) | ✗ | Baseline: content+position merged |
| RoBERTa | Input embedding | ✓ (learned) | ✗ | Same as BERT, more data/training |
| XLNet | Transformer-XL | ✗ | ✓ (sinusoidal) | Permutation LM; relative pos via fixed formulas |
| Transformer-XL | Recurrence | ✗ | ✓ (sinusoidal) | Fixed sinusoidal relative pos; segment recurrence |
| DeBERTa | Disentangled | Only in EMD | ✓ (learned) | Separate content/pos vectors; learnable relative pos; absolute only at MLM head |

> [!TIP]
> For a visual comparison of how DeBERTa's attention patterns differ from RoBERTa's, the paper includes heatmap visualizations showing that DeBERTa attends more strongly to syntactically and semantically meaningful positions.
