# Meta Information

- URL: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). Matryoshka Representation Learning. NeurIPS 2022.

# Matryoshka Representation Learning (MRL)

## Overview

Matryoshka Representation Learning (MRL) is a training framework that encodes information at multiple granularities within a single fixed-dimensional embedding vector by simultaneously optimizing multiple nested sub-dimensions. Instead of training separate models for each target embedding size, MRL trains one model whose first $m$ dimensions constitute a valid representation for any $m \in \mathcal{M}$, analogous to Russian Matryoshka dolls nested inside each other.

**Who should use this**: Practitioners deploying large-scale retrieval or classification systems where inference cost or storage must be dynamically adjusted—e.g., web-scale image search, recommendation engines, adaptive edge inference—without retraining separate models per resource tier.

## Problem Formulation

Standard representation learning trains an encoder $F(\cdot; \theta_F): \mathcal{X} \to \mathbb{R}^d$ that maps input $x$ to a single $d$-dimensional embedding used uniformly downstream. This approach is rigid: using the full $d$-dimensional vector is wasteful for easy queries; training a smaller model from scratch for every target dimension is costly.

**MRL's goal**: Learn a single $\theta_F$ such that $z_{1:m} = F(x; \theta_F)_{1:m} \in \mathbb{R}^m$ is a high-quality representation for each $m \in \mathcal{M}$, where $\mathcal{M} \subset [d]$ is a small set of $|\mathcal{M}| \leq \lfloor \log_2(d) \rfloor$ chosen sizes.

## Loss Function

The MRL training objective jointly minimizes losses across all nested sizes:

$$\min_{\left\{\mathbf{W}^{(m)}\right\}_{m \in \mathcal{M}},\, \theta_F} \frac{1}{N} \sum_{i \in [N]} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\!\left(\mathbf{W}^{(m)} \cdot F(x_i;\, \theta_F)_{1:m};\; y_i\right)$$

**Variable definitions**:
- $x_i \in \mathcal{X}$: input sample (image, text, etc.)
- $y_i$: ground-truth label
- $F(x_i; \theta_F) \in \mathbb{R}^d$: full $d$-dimensional embedding
- $F(x_i; \theta_F)_{1:m} \in \mathbb{R}^m$: first $m$ dimensions of the embedding
- $\mathbf{W}^{(m)} \in \mathbb{R}^{L \times m}$: linear classification head for dimension $m$, where $L$ is the number of classes
- $c_m \geq 0$: importance weight per granularity (set to $1$ uniformly in all experiments)
- $\mathcal{L}$: softmax cross-entropy loss
- $\mathcal{M} \subset [d]$: e.g., $\{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ for ResNet50

The only overhead vs. standard training is $|\mathcal{M}|$ extra linear heads; the backbone $F$ is shared across all granularities.

## MRL vs. MRL-E (Efficient Variant)

| Variant | Classifier Weights | Memory | Accuracy |
|---------|-------------------|--------|----------|
| **MRL** | Independent $\mathbf{W}^{(m)}$ per $m$ | Higher (~$2\times$) | Marginally higher |
| **MRL-E** | Weight-tied: $\mathbf{W}^{(m)} = \mathbf{W}_{1:m}$ (first $m$ cols of $\mathbf{W}^{(d)}$) | ~50% less than MRL | Slightly lower |

MRL-E reduces memory overhead by sharing classifier weights across dimensions, enabling deployment on memory-constrained hardware.

## Training Procedure (Pseudocode)

```
Input: Dataset {(x_i, y_i)}, backbone F(·; θ_F), nested sizes M, loss weights {c_m}
Initialize: Linear heads {W^(m)} for each m ∈ M, backbone parameters θ_F

For each batch {(x_i, y_i)}:
  1. Forward pass: z_i = F(x_i; θ_F)  ∈ R^d
  2. For each m ∈ M:
       a. Slice: z_i^(m) = z_i[1:m]   ∈ R^m
       b. Logits: l_i^(m) = W^(m) · z_i^(m)   ∈ R^L
       c. Loss:  ℓ_i^(m) = CrossEntropy(l_i^(m), y_i)
  3. Aggregate: L_total = (1/N) Σ_i Σ_m c_m · ℓ_i^(m)
  4. Backprop through L_total → update θ_F and all {W^(m)}

Output: Trained F(·; θ_F) whose first m dimensions are valid for any m ∈ M
```

> [!NOTE]
> Only $|\mathcal{M}| \leq \lfloor \log_2(d) \rfloor$ dimensions are explicitly optimized (e.g., 9 values for $d=2048$), but the learned representations "diffuse information in an interpolative fashion across all $d$ dimensions," enabling good performance at any intermediate size.

## Supported Modalities and Architectures

| Modality | Architecture | Full Dim $d$ | Nested Sizes $\mathcal{M}$ |
|----------|-------------|-------------|--------------------------|
| Vision | ResNet50 | 2048 | {8, 16, 32, 64, 128, 256, 512, 1024, 2048} |
| Vision | ViT-B/16 (JFT-300M) | 768 | {12, 24, 48, 96, 192, 384, 768} |
| Vision-Language | ALIGN | 768 | {12, 24, 48, 96, 192, 384, 768} |
| Language | BERT-Base | 768 | {12, 24, 48, 96, 192, 384, 768} |

## Downstream Applications

### Adaptive Classification

A cascade mechanism uses learned thresholds on the maximum softmax probability to route examples through progressively larger representations:

```
Input: x, thresholds {τ_m} learned on validation set
1. Compute z = F(x)
2. For m in sorted(M):  # smallest to largest
     logits = W^(m) · z[1:m]
     prob = softmax(logits)
     if max(prob) ≥ τ_m:
         return argmax(prob), m   # confident early exit
3. return argmax(softmax(W^(d) · z)), d   # full representation
```

Easy examples exit early with low-dimensional embeddings; ambiguous examples use the full representation.

### Adaptive Retrieval

Two-stage pipeline that avoids exhaustive search with full-dimensional embeddings:

```
Input: query q, database D of N items, shortlist size K, dimensions m_low < m_high
Stage 1 - Shortlist (cheap):
    Compute q_low = F(q)[1:m_low]
    Retrieve top K items from D by cosine similarity in R^{m_low}
    → Candidate set C ⊆ D, |C| = K  (K << N)

Stage 2 - Rerank (precise):
    Compute q_high = F(q)[1:m_high]
    Re-score C by cosine similarity in R^{m_high}
    → Return top results from C
```

Using $m_{\text{low}}=16$, $m_{\text{high}}=2048$ achieves 128× FLOP reduction and 14× wall-clock speedup on ImageNet-1K retrieval at equivalent mAP@10 to the full-dimensional baseline.

## Comparison with Related Methods

| Method | Training Cost | Embedding Flexibility | Accuracy at Small $m$ |
|--------|--------------|----------------------|----------------------|
| **Independent FFRep** | $|\mathcal{M}|$ models | Fixed (retrain per $m$) | Optimal per size |
| **SVD** | Post-hoc on pretrained | Continuous | Lossy compression |
| **Slimmable Networks** | Specialized training | Discrete sub-nets | Competitive |
| **MRL** (ours) | 1 model + small heads | Continuous nesting | Matches FFRep |
| **MRL-E** (ours) | Same as MRL | Continuous nesting | Slightly below FFRep |

> [!IMPORTANT]
> MRL matches the accuracy of independently trained Fixed Feature (FF) representations at each nested size $m$, without requiring separate model training per size. This is the key advantage over retraining baselines.

# Experiments

- **Datasets**:
  - ImageNet-1K: 1.28M training images, 50K validation, 1K classes — primary vision benchmark
  - ImageNet-4K: 4.2M training images, 200K query images, 4K classes — large-scale retrieval
  - JFT-300M: 300M noisy-labeled images — large-scale ViT pretraining
  - ALIGN: 1.8B image-text pairs — vision-language contrastive pretraining
  - Wikipedia + BooksCorpus: standard BERT pretraining corpus — language modeling
  - ImageNetV2, ImageNetR, ImageNetA, ImageNet-Sketch: robustness evaluation benchmarks
  - FLUID: long-tail sequential learning benchmark for few-shot and novel-class evaluation

- **Hardware**: A100 GPUs (vision experiments), TPU pods (JFT-300M scale)

- **Optimizer**: Standard SGD with cosine LR schedule for ResNet50; AdamW for ViT/BERT

- **Key Results**:
  - ResNet50 on ImageNet-1K: 76.3% top-1 accuracy at ~37 dims vs. 512-dim baseline — 14× size reduction
  - Adaptive Retrieval (ImageNet-1K): mAP@10 parity at 128× fewer FLOPs, 14× wall-clock speedup (16-dim shortlist → 2048-dim rerank)
  - Adaptive Retrieval (ImageNet-4K): 32× theoretical, 6× practical speedup at parity mAP@10
  - Long-tail FLUID: +2% novel class accuracy versus fixed feature baselines, no degradation on seen classes
  - Robustness: MRL representations at least as robust as full-dimensional; +0.6% on ImageNet-A
  - ALIGN (vision-language): MRL representations at 96-dim match full 768-dim zero-shot retrieval accuracy on MS-COCO and Flickr30K

> [!TIP]
> Code and pretrained models are open-sourced at the paper's GitHub repository. Training uses the FFCV data pipeline for high-throughput ImageNet training.
