# Meta Information

- URL: [Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking](https://arxiv.org/abs/2601.04720)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A., Liu, D., Zhou, J., & Lin, J. (2026). Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking. arXiv preprint arXiv:2601.04720.

# Qwen3-VL-Embedding and Qwen3-VL-Reranker

## Abstract

This technical report introduces the Qwen3-VL-Embedding and Qwen3-VL-Reranker model series from Tongyi Lab, Alibaba Group. Built on the Qwen3-VL foundation model, these models form an end-to-end pipeline for high-precision multimodal search by mapping text, images, document images, and video into a unified representation space.

Key properties:
- Embedding model uses a multi-stage training paradigm from contrastive pre-training to reranking model distillation
- Supports Matryoshka Representation Learning (MRL) for flexible embedding dimensions
- Handles inputs up to 32,768 tokens
- Supports over 30 languages
- Released in 2B and 8B parameter sizes
- Qwen3-VL-Embedding-8B achieves 77.8 overall score on MMEB-V2, ranking first as of January 2026

## 1 Introduction

The exponential growth of multimodal internet content (images, documents, infographics, screenshots, videos) requires retrieval systems that understand semantic concepts across modalities. Two critical modules in multimodal retrieval architectures are embedding models and reranking models.

Prior work established the importance of contrastive learning on image-text pairs (CLIP, Radford et al. 2021). More recent work trains unified multimodal embeddings using VLMs: E5-V, GME, BGE-VL, and VLM2Vec. VLM-based approaches benefit from built-in cross-modal alignment, fine-grained attention, and multilingual knowledge.

> [!NOTE]
> "Building on these breakthroughs, the multimodal retrieval community has increasingly explored training unified multimodal embedding models based on VLMs."

The Qwen3-VL series extends Qwen3-VL-Instruct with:
1. **Embedding models** producing semantically rich high-dimensional vectors via multi-stage contrastive training and knowledge distillation
2. **Reranking models** using a cross-encoder for precise relevance scoring

## 2 Model Architecture

Both models are built on the Qwen3-VL backbone with causal attention. After training on large-scale multimodal relevance data, they retain the backbone's world knowledge and multimodal perception while gaining relevance estimation capability.

**Model specifications:**

| Model Type | Size | Layers | Seq Length | Embedding Dim | MRL Support | Quantization |
|---|---|---|---|---|---|---|
| Qwen3-VL-Embedding | 2B | 28 | 32K | 2048 | Yes | Yes |
| Qwen3-VL-Embedding | 8B | 36 | 32K | 4096 | Yes | Yes |
| Qwen3-VL-Reranker | 2B | 28 | 32K | - | - | - |
| Qwen3-VL-Reranker | 8B | 36 | 32K | - | - | - |

### Embedding Method

The embedding model follows a bi-encoder architecture. Input: an instruction (system message) + multimodal instance (user message: text, image, video, or any combination). A special PAD token (`<|endoftext|>`) is appended, and the last hidden state of that token serves as the dense vector representation.

**Input template:**
```
<|im_start|>system
{Instruction}
<|im_end|>
<|im_start|>user
{Instance}
<|im_end|>
<|im_start|>assistant
<|endoftext|>
```

- Input: multimodal instance of any combination of text/image/video tokens
- Output: dense vector $\mathbf{e} \in \mathbb{R}^{d}$ where $d \in \{2048, 4096\}$ for 2B and 8B respectively

> [!IMPORTANT]
> The default instruction is "Represent the user's input." Users can customize the instruction for task-specific tuning, making the model instruction-aware.

### Reranking Method

The reranking model adopts a pointwise cross-encoder architecture. Input: a relevance-defining instruction + the query + the document, all passed as user messages. The model predicts "yes" or "no" as the next token, and the relevance score is derived from the logit difference:

```math
\begin{align}
  s = \text{sigmoid}(\text{logit}(\text{yes}) - \text{logit}(\text{no}))
\end{align}
```

**Input template:**
```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct
provided. Note that the answer can only be "yes" or "no".
<|im_end|>
<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}
<|im_end|>
<|im_start|>assistant
```

> [!NOTE]
> Unlike the embedding model which processes query and document independently (bi-encoder), the reranking model performs cross-attention between query and document representations, yielding more fine-grained but computationally heavier relevance estimates.

## 3 Data

### 3.1 Dataset Format

The training data comprises multiple sub-datasets $\mathcal{D} = \{D_i\}_{i=1}^M$. Each sub-dataset $D_i$ is a quadruple $(I_i, Q_i, C_i, R_i)$:

- $I_i$: Instruction — textual description of the relevance criteria
- $Q_i = \{q_j\}_{j=1}^{N_q}$: Queries — text, image, video, or multimodal combinations
- $C_i = \{d_j\}_{j=1}^{N_d}$: Corpus — same modality flexibility as queries
- $R_i = \{(q_j, \{d_{j,k}^+\}_{k=1}^{n^+}, \{d_{j,k}^-\}_{k=1}^{n^-})\}_{j=1}^{N_q}$: Relevance labels with positive and negative documents per query

### 3.2 Data Synthesis

Due to significant imbalance in publicly available and proprietary data, the authors use data synthesis extending the Qwen3 Embedding methodology to multimodal scenarios.

**Seed Pool Construction:**
1. Collect high-quality raw image and video datasets with coarse-grained quality filtering (resolution, aspect ratio)
2. Apply scene cut detection and remove static/corrupted video segments
3. Use Qwen3-VL-32B to generate fine-grained categorical labels
4. Filter samples with low-confidence annotations or poor visual-text correspondence (measured via GME similarity scores)
5. Perform category-wise rebalancing

**Image Task Annotation** (3 paradigms):
1. **Image Classification**: Query is image + classification instruction; document is the class label. Tasks include object recognition, scene parsing, landmark identification, action recognition.
2. **Image Question Answering**: Query is image + question; document is the answer. Covers factoid, visual reasoning, OCR-based extraction, and domain-specific knowledge.
3. **Image Retrieval**: Query is a search text; document is the candidate image. Synthesizes queries across semantic depths from direct descriptions to knowledge-centric localization.

**Video Task Annotation** (4 paradigms):
1. **Video Classification**: Query is video + task; document is the category
2. **Video Question Answering**: Query is video + question; document is the answer
3. **Video Retrieval**: Query is a text description; document is the video
4. **Moment Retrieval**: Query is a text (optionally with a keyframe); document is a specific video segment. The model localizes target actions/objects/characters at a temporal granularity.

> [!NOTE]
> A two-step annotation approach is used: the model first generates a descriptive caption for each image or video, then performs task-specific annotation. This improves quality and consistency.

### 3.3 Positive Refinement and Hard Negative Mining

Hard negatives are critical for contrastive learning. The mining pipeline has two stages:

**Recall:** For each sub-dataset $D_i$, an embedding model retrieves the top-$K$ most similar documents for each query based on cosine similarity scores $S = \{s_{j,k}\}_{k=1}^K$.

**Relevance Filtering:**
- **Positive Refinement**: Retain query $q_j$ only if at least one positive document $d^+$ achieves score $s > t^+$ (threshold). Discard queries without any retrieved positive.
- **Hard Negative Selection**: Select non-positive documents $d$ as hard negatives only if their score satisfies $s < \bar{s}^+ + \delta^-$, where $\bar{s}^+$ is the mean score of refined positives and $\delta^-$ is a safety margin preventing false negative inclusion.

## 4 Training Strategy

A multi-stage training pipeline addresses the imbalance between abundant weakly-supervised data and scarce high-quality samples.

### 4.1 Multi-Stage Training

**Stage 1: Contrastive Pre-training (s0)**
- Trains on 300M large-scale, multimodal, multi-task synthetic data
- Uses InfoNCE loss (Equation 1) with hard negatives mined using an existing open-source model (GME)
- Produces Qwen3-VL-Embedding: s0

**Stage 2: Multi-Task Contrastive Learning and Supervised Fine-Tuning (s1)**
- Uses ~40M combined curated public + proprietary + synthetic data
- s0 is used to re-mine data for higher quality
- Tailored contrastive objectives per task type (retrieval/classification/STS)
- Simultaneously trains Qwen3-VL-Reranker on 4M retrieval-specific subset using the binary yes/no objective
- Produces Qwen3-VL-Embedding: s1

**Stage 3: Distillation and Model Merging (s2 → s3)**
- Curates 4M balanced subset from public and proprietary sources
- Qwen3-VL-Reranker generates fine-grained relevance scores as supervision signal
- Embedding model minimizes cross-entropy distribution matching loss (Equation 3)
- Produces Qwen3-VL-Embedding: s2 (strong on retrieval, weaker on classification/QA)
- **Model Merging**: s2 and s1 are merged via the methodology of Li et al. (2024) to produce the final Qwen3-VL-Embedding: s3 with balanced performance across all tasks

### 4.2 Implementation

- Uses Low-Rank Adaptation (LoRA) with weights initialized from Qwen3-VL-Instruct
- Dynamic resolution and frame rates:
  - Images: preserve aspect ratio, cap at 1,280 tokens ($\approx 1.3 \times 10^6$ pixels)
  - Videos: sample at 1 FPS with max 64 frames; cap total tokens at 4,500 ($\approx 9.2 \times 10^6$ pixels)

## 5 Training Objectives

### 5.1 Loss Functions for the Embedding Model

Four distinct loss functions address different data types:

**Retrieval Data Loss (InfoNCE):**

```math
\begin{align}
  \mathcal{L}_{\text{retrieval}} = -\frac{1}{N} \sum_i^N \log \frac{e^{s(q_i, d_i^+)/\tau}}{Z_i}
\end{align}
```

where $s(\cdot, \cdot)$ is cosine similarity, $\tau$ is the temperature, and $Z_i$ aggregates:

```math
\begin{align}
  Z_i &= e^{s(q_i, d_i^+)/\tau} + \sum_k^K m_{ik} e^{s(q_i, d_{i,k}^-)/\tau} \\
      &+ \sum_{j \neq i} m_{ij} e^{s(q_i, q_j)/\tau} + \sum_{j \neq i} m_{ij} e^{s(d_i^+, d_j)/\tau} + \sum_{j \neq i} m_{ij} e^{s(q_i, d_j)/\tau}
\end{align}
```

The five terms correspond to: (1) the positive document, (2) $K$ explicit hard negatives, (3) other in-batch queries as negatives, (4) other in-batch documents contrasted against $d_i^+$, and (5) other in-batch documents contrasted against $q_i$.

The masking factor $m_{ij}$ mitigates false negatives:

```math
\begin{align}
  m_{ij} = \begin{cases} 0, & \text{if } s_{ij} > s(q_i, d_i^+) + 0.1 \text{ or } d_j = d_i^+ \\ 1, & \text{otherwise} \end{cases}
\end{align}
```

> [!NOTE]
> In Stage 2, the query-query ($\sum s(q_i, q_j)$) and document-document ($\sum s(d_i^+, d_j)$) terms are removed from $Z_i$. This empirically improves performance on high-quality multimodal retrieval data.

**Classification Data Loss:** Same InfoNCE formulation, but negative samples are restricted to explicitly incorrect labels for the same query. Other in-batch labels are ignored to avoid false negatives.

**Semantic Textual Similarity (STS) Loss (CoSent):**

```math
\begin{align}
  \mathcal{L}_{\text{sts}} = \log \left( 1 + \sum_{\hat{s}(q_i, d_j) > \hat{s}(q_m, d_n)} \exp \left( \frac{\cos(q_m, d_n) - \cos(q_i, d_j)}{\tau} \right) \right)
\end{align}
```

where $\hat{s}(q_i, d_j)$ denotes the ground-truth similarity score. This preserves the ordering induced by the human-annotated scores, which are real-valued rather than binary.

**Distillation Loss (Stage 3):** The reranker's relevance logits serve as soft supervision. For each query $q$, offline reranker scores are pre-computed for one positive and $k$ negatives. The embedding model minimizes a cross-entropy distribution-matching objective:

```math
\begin{align}
  \mathcal{L}_{\text{distill}} = - \sum_{i=1}^{k+1} P_{\text{reranker}}(d_i | q) \log P_{\text{embedding}}(d_i | q)
\end{align}
```

where $P(d_i | q)$ is the softmax distribution over $k+1$ candidates.

#### 5.1.1 Additional Techniques for Efficient Inference

**Matryoshka Representation Learning (MRL):** Each loss is computed on both the full-dimensional embedding and truncated lower-dimensional prefixes of the same representation. Training over a dense set of MRL dimensions generalizes to intermediate dimensions not explicitly included during training.

**Quantization-Aware Training (QAT):** During training, the optimization objective is computed using both full-precision embeddings and their quantized (int8 or binary) counterparts. This uses Learned Step Size Quantization (LSQ), which treats the quantization scale as a learnable parameter updated via backpropagation. A Straight-Through Estimator (STE) propagates gradients through the non-differentiable rounding operation.

### 5.2 Loss Function for the Reranking Model

Reranking is framed as binary classification:

```math
\begin{align}
  \mathcal{L}_{\text{reranking}} = -\log p(l \mid I, q, d)
\end{align}
```

where $l \in \{\text{"yes"}, \text{"no"}\}$ is the relevance label and $p(\cdot|*)$ is the VLM probability. Inference score: $s = \text{sigmoid}(\text{logit}(\text{yes}) - \text{logit}(\text{no}))$.

> [!TIP]
> This pointwise reranking approach contrasts with listwise or pairwise reranking. It is computationally heavier than embedding-based retrieval but provides more accurate relevance estimation for a small candidate set (e.g., top-100).

## 6 Evaluation

### 6.1 Multimodal Benchmarks (MMEB-V2)

MMEB-V2 (Meng et al., 2025) covers 78 datasets across three domains—Image (36 datasets), Video (18 datasets), and Visual Document (24 datasets)—with task categories including classification, QA, retrieval, grounding, moment retrieval, ViDoRe, VisRAG, and OOD.

**Key results (Qwen3-VL-Embedding-8B vs. best prior models):**

- Overall: **77.8** vs. 72.9 (RzenEmbed-8B, best prior open-source) — +6.7% improvement
- Image overall: 80.1 vs. 75.9
- Video overall: 67.1 vs. 55.7
- VisDoc overall: 82.4 vs. 81.3

The 8B model surpasses all open-source and closed-source models including IFM-TTE (74.1) and Seed-1.6-embedding-1215 (76.9) on the January 2026 leaderboard.

### 6.2 Visual Document Benchmarks

Evaluated on JinaVDR and Vidore-v3 against ColPali-style models, which use multi-vector late interaction (significantly higher computational cost).

- Qwen3-VL-Embedding-8B achieves avg 75.8, comparable to tomoro-colqwen3-embed-8b (77.7) and colnomic-embed-multimodal-7b (75.5) at similar size
- Qwen3-VL-Reranker-8B achieves avg **80.3**, outperforming all ColPali-style models and jina-reranker-m0 (not directly comparable)

> [!IMPORTANT]
> The embedding model achieves competitive performance with ColPali models using single-vector (dense) representations, avoiding multi-vector indexing costs. The reranker substantially outperforms ColPali models of similar parameter size.

### 6.3 Text Benchmarks (MMTEB Multilingual)

On MMTEB (Enevoldsen et al., 2025), Qwen3-VL-Embedding-8B achieves:
- Mean task score: **67.9** (vs. Qwen3-Embedding-8B: 70.6 — slightly lower but competitive)
- Bitext Mining: 77.5, Classification: 72.0, Clustering: 55.8, Retrieval: 69.4, STS: 75.4

The multimodal embedding model maintains competitive performance on pure text tasks despite being trained primarily for multimodal retrieval.

### 6.4 Reranking Evaluation

Evaluated using Qwen3-VL-Embedding-2B to retrieve top-100 candidates, then applying rerankers:

| Model | Size | MMEB-v2 Avg | MMTEB Retrieval | JinaVDR | ViDoRe v3 |
|---|---|---|---|---|---|
| Qwen3-VL-Embedding-2B (baseline) | 2B | 73.4 | 68.1 | 71.0 | 52.9 |
| Qwen3-VL-Reranker-2B | 2B | 75.2 | 70.0 | 80.9 | 60.8 |
| Qwen3-VL-Reranker-8B | 8B | 79.2 | 74.9 | 83.6 | 66.7 |

The 8B reranker improves 4.1 points over the 2B reranker across tasks.

## 7 Analysis

### 7.1 Efficacy of MRL and Embedding Quantization

Evaluated on MSMARCO Passage Ranking (text retrieval, 10k queries) and VL3-Syn (text-to-image retrieval, 10k captions × 2M image corpus) using MRR@10.

**Key findings:**
- Reducing embedding dimension from 1024 to 512 in text retrieval results in only 1.4% performance decrease while achieving 50% storage reduction and doubling retrieval speed
- int8 quantization preserves retrieval performance with negligible degradation
- Binary quantization significantly impairs retrieval effectiveness, especially at lower dimensions

### 7.2 Impact of Spatial and Temporal Granularity

Performance consistently improves with increased visual resource consumption (more image tokens, more video frames) across all task categories. However, diminishing returns are pronounced at the highest resource levels, with slight performance regression at extreme token counts due to long-context degradation.

### 7.3 Performance Across Training Stages

Stage progression on MMEB-V2 (2B model):

| Stage | Image Overall | Video Overall | VisDoc Overall | All |
|---|---|---|---|---|
| s0 (contrastive pre-training) | 65.8 | 57.5 | 74.8 | 66.6 |
| s1 (multi-task contrastive) | 74.8 | 60.3 | 77.1 | 72.1 |
| s2 (distillation) | 71.3 | 59.5 | 80.9 | 71.5 |
| s3 (model merging) | 75.0 | 61.9 | 79.2 | 73.2 |

> [!NOTE]
> Stage s2 (distillation from reranker) boosts retrieval-centric tasks (VisDoc +3.8) but causes slight regression in classification/QA. Model merging of s1 and s2 recovers this regression while retaining the retrieval gains.

## 8 Conclusion

Qwen3-VL-Embedding and Qwen3-VL-Reranker achieve state-of-the-art performance on multimodal retrieval benchmarks by combining:
1. A multi-stage training pipeline that bootstraps data quality and model performance iteratively
2. High-quality synthesized multimodal data with balanced task coverage
3. Practical deployment features: MRL for flexible embedding size, QAT for storage reduction, LoRA for efficient training

Future directions include extending to additional modalities, more efficient training paradigms, enhanced compositional reasoning, and more comprehensive evaluation protocols.

# Experiments

- **Datasets (Training):**
  - Stage 1: 300M synthesized multimodal multi-task data (image classification, image QA, image retrieval, video classification, video QA, video retrieval, moment retrieval)
  - Stage 2: ~40M combined public, proprietary, and synthetic data
  - Stage 3 (distillation): 4M balanced subset from public and proprietary sources
  - Seed pool: diverse raw image and video datasets filtered for quality, labeled with Qwen3-VL-32B, rebalanced by category

- **Datasets (Evaluation):**
  - MMEB-V2: 78 datasets across image (36), video (18), visual document (24) tasks
  - JinaVDR (Günther et al., 2025): visual document retrieval
  - Vidore-v3: visual document retrieval
  - MMTEB Multilingual (Enevoldsen et al., 2025): text embedding benchmark
  - MSMARCO Passage Ranking (Bajaj et al., 2016): text retrieval (10k query sample, full passage corpus)
  - VL3-Syn (Zhang et al., 2025a): text-to-image retrieval (10k captions × 2M images)

- **Hardware:** Not specified

- **Optimizer:** LoRA (Low-Rank Adaptation); model weights initialized from Qwen3-VL-Instruct

- **Results:**
  - Qwen3-VL-Embedding-8B: 77.8 on MMEB-V2 (rank 1 as of January 2026), 67.9 mean task score on MMTEB Multilingual
  - Qwen3-VL-Embedding-2B: 73.2 on MMEB-V2, 63.9 on MMTEB Multilingual
  - Qwen3-VL-Reranker-8B: 80.3 average on visual document benchmarks; consistently outperforms 2B reranker by ~4 points across tasks
  - int8 quantization preserves performance; binary quantization causes notable degradation
  - Reducing embedding dimension from 1024 to 512: only 1.4% MRR@10 drop in text retrieval with 50% storage saving
