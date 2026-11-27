# Meta Information

- URL: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

---

# Problem: Scalability of BERT for Semantic Similarity

Standard BERT for semantic textual similarity requires feeding both sentence $A$ and $B$ together into the network using cross-attention. For a corpus of $n = 10{,}000$ sentences, finding the most similar pair requires $\binom{n}{2} \approx 50{,}000{,}000$ inference passes — approximately **65 hours** on a modern GPU. This makes BERT impractical for retrieval and clustering at scale.

Simple alternatives like averaging BERT token embeddings or using the `[CLS]` token representation were shown empirically to produce **worse** sentence embeddings than averaging GloVe vectors.

---

# Sentence-BERT (SBERT)

## Overview

SBERT fine-tunes BERT using **Siamese** and **Triplet** network structures so that the resulting fixed-size sentence embeddings can be compared efficiently via cosine similarity. Once encoded, the 10,000-sentence problem reduces to $n$ forward passes and pairwise cosine similarity lookups — approximately **5 seconds**.

**Who uses this:** NLP engineers and researchers who need scalable sentence embeddings for semantic search, clustering, paraphrase detection, or any task requiring efficient sentence-level similarity.

## Architecture

Given a sentence $s$, SBERT feeds it through a BERT (or RoBERTa) encoder to produce token embeddings $\mathbf{H} \in \mathbb{R}^{T \times d}$ where $T$ is the number of tokens and $d = 768$ (base) or $d = 1024$ (large). A **pooling layer** then aggregates $\mathbf{H}$ into a fixed-size sentence embedding $\mathbf{u} \in \mathbb{R}^d$.

Three pooling strategies are explored:
- **Mean pooling** (default): $\mathbf{u} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{H}_t$
- **Max pooling**: $\mathbf{u}_i = \max_t \mathbf{H}_{t,i}$
- **CLS token**: $\mathbf{u} = \mathbf{H}_{\text{[CLS]}}$

Ablation studies find mean pooling optimal for most tasks.

---

## Training Objectives

### 1. Classification Objective (for NLI data)

Given two sentence embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$, a classification vector is constructed:

$$\mathbf{c} = [\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|] \in \mathbb{R}^{3d}$$

This is passed through a weight matrix $\mathbf{W}_t \in \mathbb{R}^{3d \times k}$ and a softmax layer:

$$\hat{y} = \text{softmax}(\mathbf{W}_t \mathbf{c})$$

where $k$ is the number of labels (e.g., entailment, neutral, contradiction). Cross-entropy loss is used.

> [!IMPORTANT]
> The element-wise difference $|\mathbf{u} - \mathbf{v}|$ is the most critical component of the concatenation. Ablation shows that omitting it dramatically degrades performance. Removing both $\mathbf{u}$ and $\mathbf{v}$ from the concatenation and keeping only $|\mathbf{u} - \mathbf{v}|$ still achieves reasonable performance.

### 2. Regression Objective

The loss is the mean squared error between the cosine similarity and the gold label $y \in [0, 1]$:

$$\mathcal{L} = \left(\text{cos}(\mathbf{u}, \mathbf{v}) - y\right)^2, \quad \text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This is used when training directly on sentence similarity scores.

### 3. Triplet Objective

Given an anchor $\mathbf{a}$, a positive $\mathbf{p}$, and a negative $\mathbf{n}$, all in $\mathbb{R}^d$, the loss is:

$$\mathcal{L} = \max\!\left(0,\ \|\mathbf{a} - \mathbf{p}\| - \|\mathbf{a} - \mathbf{n}\| + \varepsilon\right)$$

where $\varepsilon = 1$ (margin). This ensures that the anchor is always closer to the positive than to the negative by at least $\varepsilon$.

---

## Training Procedure

**Algorithm (Classification objective, NLI fine-tuning):**

```
Input: Sentence pairs (s1, s2) with labels y from SNLI + MultiNLI
1. For each (s1, s2, y):
   a. u = MeanPool(BERT(s1))   # u ∈ R^d
   b. v = MeanPool(BERT(s2))   # v ∈ R^d
   c. c = [u; v; |u - v|]      # c ∈ R^{3d}
   d. logits = W_t · c          # W_t ∈ R^{3d × k}
   e. loss = CrossEntropy(softmax(logits), y)
2. Optimize with Adam (lr=2e-5, warmup=10% of steps)
3. Batch size: 16, Epochs: 1
```

The siamese structure means **both BERT encoders share the same weights**, so the network has the same parameter count as a single BERT model.

---

# Differences from Related Methods

| Method | Encoding | Similarity | Scalability | Training |
|---|---|---|---|---|
| BERT (cross-encoder) | Both sentences together | Cross-attention | $O(n^2)$ | Pre-trained |
| InferSent | GRU with max pooling | Cosine | $O(n)$ | NLI supervision |
| Universal Sentence Encoder | Transformer / DAN | Cosine | $O(n)$ | Multi-task |
| **SBERT** | BERT with mean pooling | Cosine | $O(n)$ | NLI + Siamese fine-tuning |

> [!NOTE]
> BERT cross-encoders significantly outperform SBERT on tasks like STS Benchmark (85.64 vs. 79.23 Spearman) but are $O(n^2)$ in encoding cost. SBERT closes much of this gap while achieving $O(n)$ scalability.

> [!TIP]
> The sentence-transformers library, released alongside this paper, provides a production-ready implementation: [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)

---

# Experiments

## Datasets

| Dataset | Task | Size | Notes |
|---|---|---|---|
| SNLI | NLI (entailment/neutral/contradiction) | 570,000 pairs | Training |
| Multi-Genre NLI | NLI | 430,000 pairs | Training |
| STS 2012–2016 | Semantic Textual Similarity | ~1,000–3,000 pairs each | Evaluation |
| STS Benchmark | STS | train/dev/test | Supervised fine-tuning |
| SICK-Relatedness | Semantic similarity | ~5,000 pairs | Evaluation |
| Argument Facet Similarity (AFS) | Argumentative similarity | 6,000 pairs | 3 topics |
| Wikipedia sections | Triplet distinction | 1.8M triplets | Training for Wikipedia task |
| SentEval suite | Transfer learning | Various | 7 classification tasks |

## Hardware

Not explicitly stated; benchmarks reported on CPU and GPU (V100 implied).

## Key Quantitative Results

**Semantic Textual Similarity (Spearman correlation, unsupervised):**

| Method | STS avg. (7 tasks) |
|---|---|
| Average GloVe embeddings | 61.32 |
| InferSent-v2 | 65.01 |
| Universal Sentence Encoder | 71.22 |
| SBERT-NLI-base | 74.89 |
| **SBERT-NLI-large** | **76.55** |

**STS Benchmark (Spearman correlation):**

| Method | STSb Score |
|---|---|
| BERT-STSb-large (cross-encoder) | 85.64 |
| SBERT-NLI-STSb-large | 86.10 |
| SBERT-NLI-large | 79.23 |

Two-stage training (NLI → STSb) achieves 86.10, surpassing the BERT cross-encoder trained only on STSb.

**Wikipedia Section Distinction (triplet accuracy):**

| Method | Accuracy |
|---|---|
| Dor et al. (2018) | 74.00% |
| **SBERT-WikiSec-large** | **80.78%** |

## Computational Efficiency

| Method | GPU (sent/sec) | CPU (sent/sec) |
|---|---|---|
| Avg. GloVe | — | 6,469 |
| InferSent-v2 | 1,876 | 137 |
| Universal Sentence Encoder | 1,318 | 67 |
| SBERT-base (smart batching) | 2,042 | 83 |

SBERT with smart batching (grouping sentences of similar length to minimize padding) is **55% faster** than Universal Sentence Encoder on GPU.

---

# Ablation: Pooling and Concatenation

**Pooling strategy impact (NLI classification task, Spearman on STS 2016):**

| Pooling | Score |
|---|---|
| Mean | 80.78 |
| CLS token | 79.80 |
| Max | 79.31 |

**Concatenation ablation (regression on STSb dev):**

| Representation | Score |
|---|---|
| $(u, v, \|u-v\|)$ | **74.13** |
| $(\|u-v\|)$ only | 71.25 |
| $(u, v)$ | 70.97 |
| $(u, v, u \cdot v)$ | 72.27 |
| $(u, v, \|u-v\|, u \cdot v)$ | 73.74 |

The element-wise difference $|\mathbf{u} - \mathbf{v}|$ consistently contributes the most to performance.
