# Meta Information

- URL: [Perch 2.0: The Bittern Lesson for Bioacoustics](https://arxiv.org/abs/2508.04665)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: van Merriënboer, B., Dumoulin, V., Hamer, J., Harrell, L., Burns, A., & Denton, T. (2025). Perch 2.0: The Bittern Lesson for Bioacoustics. arXiv:2508.04665.

# Overview

Perch 2.0 is a pre-trained audio embedding model for bioacoustic species classification, developed at Google DeepMind/Google Research. It extends Perch 1.0 (avian-only) to multi-taxa datasets covering birds, frogs, insects, and marine mammals. The model produces general-purpose 1,536-dimensional embeddings that enable downstream applications—linear probing, nearest-neighbor retrieval, few-shot classification, and clustering—without requiring GPU access at inference time.

**Who should use it:** Ecologists, conservation biologists, and ML researchers needing scalable passive acoustic monitoring (PAM) with limited labeled data per deployment site. Applicable whenever embedding audio once and reusing it across many classification tasks is preferable to per-task fine-tuning.

# Architecture

Perch 2.0 is built on **EfficientNet-B3** (~12 million parameters). The processing pipeline is:

1. **Input**: Raw audio resampled to 32 kHz.
2. **Windowing**: 5-second windows ($160{,}000$ samples each) extracted from recordings.
3. **Frontend**: Log-mel spectrogram ($x \in \mathbb{R}^{T \times F}$, where $T$ is the number of time frames and $F = 128$ mel bins).
4. **Backbone**: EfficientNet-B3 processes the 2-D spectrogram as an image.
5. **Output embedding**: $z \in \mathbb{R}^{1536}$, a global average-pooled feature vector used for all downstream tasks.

> [!NOTE]
> The 1,536-dimensional embedding is intentionally large to retain fine-grained spectro-temporal detail useful for distinguishing the ~15,000 classes encountered during training.

# Training Data

| Source | Recordings | Primary Taxa |
|---|---|---|
| Xeno-Canto | 896,000 | Birds |
| iNaturalist | 571,000 | Multi-taxa (frogs, insects, etc.) |
| Tierstimmenarchiv | 33,000 | Various vertebrates |
| FSD50K | 40,000 | Non-biological sounds (negative class) |

Total: ~1.54 million recordings, ~14,795 distinct classes (14,597 species labels + non-species labels).

Including iNaturalist and Tierstimmenarchiv recordings — absent from Perch 1.0 — provides supervision signal for non-avian taxa, enabling cross-taxa transfer without retraining.

# Novel Training Techniques

## Multi-source Mixup

Standard mixup combines two examples. Perch 2.0 generalizes this to $N \in \{2, \ldots, 5\}$ sources, where $N$ is drawn from a **beta-binomial distribution**. Given $N$ audio segments $\{a_1, \ldots, a_N\}$ and mixing weights $\lambda = (\lambda_1, \ldots, \lambda_N)$ sampled from a symmetric Dirichlet distribution, the mixed waveform is:

```math
\begin{align}
  \tilde{a} = \sum_{i=1}^{N} \lambda_i \cdot a_i, \quad \lambda \sim \text{Dir}(\alpha \mathbf{1}_N)
\end{align}
```

The target label is a **multi-hot vector** reflecting all $N$ vocalizing species, enabling the model to handle the realistic scenario of multiple simultaneous callers.

**Input**: $N$ raw waveforms, each $\in \mathbb{R}^{160000}$; mixing coefficients $\lambda \in \Delta^{N-1}$.
**Output**: Mixed waveform $\tilde{a} \in \mathbb{R}^{160000}$; multi-hot label $y \in \{0,1\}^{C}$ where $C$ is the number of classes.

## Self-distillation Pipeline

Training uses a two-phase **self-distillation** approach:

```
Phase 1 — Prototype Classifier:
  For each mini-batch:
    1. Compute embeddings z = Encoder(a)
    2. Classify via prototype-learning head → soft probabilities p_soft
    3. Back-propagate through encoder + prototype head

Phase 2 — Linear Classifier:
  For each mini-batch:
    1. Compute embeddings z = Encoder(a)          [gradients flow]
    2. Forward through prototype head → p_teacher  [stop-gradient]
    3. Train linear head to match p_teacher as soft targets
    4. Back-propagate through encoder + linear head only
```

> [!IMPORTANT]
> The stop-gradient prevents the linear classifier's loss from distorting the prototype-derived representation. This is what makes random window selection viable in Perch 2.0: soft targets from the teacher reduce noise from windows that happen to lack the target vocalization.

## Source Prediction Auxiliary Loss

An additional self-supervised objective trains the model to predict the **source recording identity** of a 5-second window. Because individual recordings are long and contain many similar windows, this forces the network to capture within-recording fine-grained variation.

- Treated as an $M$-class classification problem where $M$ is the number of source recordings in the mini-batch.
- Gradient flows back through the encoder, regularizing the embedding space.

# Window Selection

Perch 1.0 used **energy-peak selection** (choosing windows with highest spectral energy) to increase the probability of including the target vocalization. Perch 2.0 switches to **random window selection** with no performance loss, attributed to the self-distillation teacher providing reliable soft targets even for low-energy windows.

# Why Supervised Learning Outperforms Self-supervised Learning in Bioacoustics

The paper systematically addresses the question of whether self-supervised pre-training (SSL) can match supervised pre-training in this domain. Key findings:

| Factor | Self-supervised (e.g., DINOv2, AVES, Bird-MAE) | Supervised (Perch 2.0) |
|---|---|---|
| Data scale | Requires ~100M examples for competitive SSL | Effective with ~1.5M labeled recordings |
| Label granularity | None (SSL learns generic features) | 15,000+ species provides fine-grained signal |
| Augmentation design | Requires careful domain-specific tuning | Standard mixup generalizes well |
| Downstream accuracy | Lower on bioacoustic benchmarks | State-of-the-art |

The authors hypothesize that the two-order-of-magnitude gap in dataset scale (vs. 142M-image DINOv2 training) makes SSL impractical for bioacoustics today. When fine-grained labels for hundreds of thousands of examples are available, supervised methods are "increasingly difficult to outperform."

> [!NOTE]
> This finding parallels results from the HEAR benchmark (Turian et al., 2022) where supervised audio models dominated self-supervised baselines across diverse sound classification tasks.

# Evaluation Protocol

Model selection used a composite score across **19 constituent datasets** grouped into three task types:

1. **Pretrained classifier performance**: ROC-AUC on fully-annotated soundscape datasets (e.g., BirdSet test sets). Measures how well the embedding + fixed linear classifier identifies species in multi-label scenarios.
2. **One-shot retrieval**: Nearest-neighbor search using cosine similarity. Given one query embedding, retrieve the correct species from a gallery. Measures embedding geometry directly.
3. **Linear transfer (few-shot)**: Train a linear probe with 16 examples per class. Measures low-data-regime adaptability.

Final scores are combined via **geometric mean** across task types to penalize models that excel on one task but fail on others.

# Experiments

- **Dataset**: BirdSet (bird soundscape classification, multi-label), BEANS (cross-taxa benchmark including frogs, whales, insects)
- **Hardware**: Not explicitly stated (Google DeepMind infrastructure)
- **Optimizer**: Not explicitly stated
- **Key Results**:
  - BirdSet — ROC-AUC: 0.908, cmAP: 0.431, Top-1 accuracy: 0.665 (state-of-the-art without fine-tuning)
  - BEANS — Classification accuracy: 0.838 (linear probe), MAP: 0.502 (prototypical probe)
  - Outperforms domain-specialized marine models (SurfPerch, Google Multispecies Whale Model) on underwater audio despite having minimal marine training data

# Comparison with Related Models

| Model | Pre-training supervision | Taxa scope | Architecture | Embedding dim |
|---|---|---|---|---|
| BirdNET (Kahl et al., 2021) | Supervised (birds) | Avian | CNN | — |
| Perch 1.0 (Hamer et al., 2023) | Supervised (birds) | Avian | EfficientNet-B1 | 1,280 |
| AVES (Hagiwara et al., 2022) | Self-supervised | Avian | Transformer | 768 |
| Bird-MAE | Self-supervised (MAE) | Avian | ViT | 768 |
| SurfPerch | Supervised | Marine | EfficientNet | — |
| **Perch 2.0** (this work) | Supervised (multi-taxa) | Multi-taxa | EfficientNet-B3 | **1,536** |

Key differences from Perch 1.0:
- Expanded training data to non-avian taxa (iNaturalist, Tierstimmenarchiv)
- Multi-source mixup replaces standard 2-source mixup
- Self-distillation replaces direct supervised training
- Source prediction auxiliary loss added
- Random window selection (vs. energy-peak selection)
