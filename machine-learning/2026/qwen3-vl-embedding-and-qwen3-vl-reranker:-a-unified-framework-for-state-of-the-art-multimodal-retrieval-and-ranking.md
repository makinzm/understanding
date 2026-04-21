# Meta Information

- URL: [Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking](https://arxiv.org/abs/2601.04720)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A., Liu, D., Zhou, J., & Lin, J. (2026). Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking. arXiv:2601.04720.

# Overview

Qwen3-VL-Embedding and Qwen3-VL-Reranker are two complementary model series for high-precision multimodal search. They extend the Qwen3-VL foundation model to map diverse input modalities—text, images, document images, and video—into a unified representation space. Together they form an end-to-end two-stage retrieval pipeline: the embedding model performs efficient initial recall via dense vector similarity, and the reranker refines the shortlist with precise relevance scoring.

Both series are released in 2B and 8B parameter configurations supporting 30+ languages with 32K-token input capacity. Qwen3-VL-Embedding-8B achieves 77.8 overall on MMEB-V2, ranking first among all models at the time of publication.

# Architecture

## Qwen3-VL-Embedding (Dual-Encoder)

The embedding model is a **dual-encoder** (also called bi-encoder): query and document are encoded independently, making it suitable for large-scale retrieval where candidate documents are indexed offline.

**Input**: Any combination of text tokens, image patches, and video frames; up to 32K tokens.

**Output**: A fixed-dimensional semantic vector $\mathbf{e} \in \mathbb{R}^{d}$, where $d = 2048$ for 2B and $d = 4096$ for 8B.

The final representation is extracted from the hidden state of the `[EOS]` token in the last transformer layer:

```math
\begin{align}
\mathbf{e} = \text{LastHiddenState}[\text{EOS}] \in \mathbb{R}^{d}
\end{align}
```

Similarity between query $\mathbf{q}$ and document $\mathbf{d}$ is computed as cosine similarity:

```math
\begin{align}
s(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}
\end{align}
```

**Instruction awareness**: Prepending a task-specific instruction string (e.g., `"Retrieve relevant documents for the query."`) to the query consistently yields 1–5% improvement. The default instruction is `"Represent the user's input."`.

### Matryoshka Representation Learning (MRL)

The embedding model supports **Matryoshka Representation Learning**, which trains the model such that any prefix of dimension $d' \leq d$ of the embedding vector retains strong retrieval quality. This allows practitioners to truncate embeddings to smaller sizes (minimum 64 for 2B, 64 for 8B) without retraining, trading storage and latency for accuracy:

```math
\begin{align}
\mathbf{e}_{d'} = \mathbf{e}[1:d'], \quad d' \in \{64, 128, 256, 512, 1024, 2048, 4096\}
\end{align}
```

| Model | Parameters | Layers | Embedding Dim | MRL |
|---|---|---|---|---|
| Qwen3-VL-Embedding-2B | 2B | 28 | 2048 | ✓ |
| Qwen3-VL-Embedding-8B | 8B | 36 | 4096 | ✓ |

## Qwen3-VL-Reranker (Cross-Encoder)

The reranker is a **cross-encoder**: query and document are concatenated into a single input and processed jointly through all transformer layers, enabling full cross-attention interaction between them. This is more accurate than dual-encoders but computationally expensive, making it suitable for re-scoring a small shortlist (e.g., top-100 candidates).

**Input**: A (query, document) pair; each may contain text, images, screenshots, or video. Input length up to 32K tokens.

**Output**: A scalar relevance score.

The model predicts the generation probability of special tokens `yes` and `no`:

```math
\begin{align}
\text{score} = \log P(\text{"yes"} \mid \text{query}, \text{document})
\end{align}
```

The raw logit score can be mapped to $[0, 1]$ via sigmoid:

```math
\begin{align}
\hat{s} = \sigma(\text{score}) = \frac{1}{1 + e^{-\text{score}}}
\end{align}
```

| Model | Parameters | Layers | Sequence Length |
|---|---|---|---|
| Qwen3-VL-Reranker-2B | 2B | 28 | 32K |
| Qwen3-VL-Reranker-8B | 8B | 36 | 32K |

# Training Methodology

Both models are built on the **Qwen3-VL foundation model** (pre-trained vision-language model with strong multimodal understanding). Training uses a **multi-stage training paradigm** to fully leverage the general multimodal semantic understanding of Qwen3-VL:

1. **Contrastive pre-training**: Large-scale weakly supervised data is used to learn a broad alignment between multimodal inputs.
2. **Supervised fine-tuning**: High-quality labeled retrieval pairs are used for task-specific contrastive learning.
3. **Multi-candidate integration**: Multiple candidate model checkpoints are integrated in the final stage.

For the embedding model, training uses an **InfoNCE / contrastive loss** over (query, positive document, negative documents) triplets:

```math
\begin{align}
\mathcal{L} = -\log \frac{\exp(s(\mathbf{q}, \mathbf{d}^+) / \tau)}{\exp(s(\mathbf{q}, \mathbf{d}^+) / \tau) + \sum_{i} \exp(s(\mathbf{q}, \mathbf{d}_i^-) / \tau)}
\end{align}
```

where $\tau$ is a temperature hyperparameter, $\mathbf{d}^+$ is a positive document, and $\mathbf{d}_i^-$ are negative documents.

Fine-tuning uses **LoRA** (rank 32, alpha 32) targeting `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`, `gate_proj` to preserve the foundation model's general capabilities.

# Two-Stage Retrieval Pipeline

The two models are designed to work together in a standard retrieve-and-rerank pipeline:

**Algorithm: Two-Stage Multimodal Retrieval**
```
Input: query q (text/image/video), corpus D of N documents
Output: top-k ranked documents

Stage 1 – Recall (Embedding):
  1. Encode all documents: e_i = Embed(d_i) for d_i in D  [offline]
  2. Encode query: e_q = Embed(q)
  3. Compute cosine similarities: s_i = cosine(e_q, e_i)
  4. Retrieve top-m candidates C = top_m({d_i}) by s_i  [e.g., m=100]

Stage 2 – Rerank:
  5. For each d_j in C: score_j = Reranker(q, d_j)
  6. Sort C by score_j descending
  7. Return top-k documents
```

This pipeline trades off **efficiency** (Stage 1 is O(N) vector search, feasible for billions of documents) against **accuracy** (Stage 2 applies expensive cross-attention on a small candidate set).

# Supported Input Modalities

| Modality | Embedding Model | Reranker Model |
|---|---|---|
| Text | ✓ | ✓ |
| Image | ✓ | ✓ |
| Screenshot / Document Image | ✓ | ✓ |
| Video | ✓ | ✓ |
| Text + Image (mixed) | ✓ | ✓ |
| Text + Video (mixed) | ✓ | ✓ |

Video is processed by sampling frames at a configurable fps with a maximum frame count. This allows the model to reason over temporal content.

# Comparison with Similar Approaches

| Aspect | Qwen3-VL-Embedding | Qwen3-VL-Reranker | Typical Text-Only Embedding |
|---|---|---|---|
| Input | Multimodal (text/image/video) | Multimodal (text/image/video) | Text only |
| Architecture | Dual-encoder (bi-encoder) | Cross-encoder | Dual-encoder |
| Query-Doc interaction | None (independent encoding) | Full cross-attention | None |
| Retrieval speed | Fast (ANN search) | Slow (pair-by-pair) | Fast (ANN search) |
| Output | Dense vector ∈ ℝ^d | Relevance scalar | Dense vector ∈ ℝ^d |
| Use case | Large-scale recall | Small-set reranking | Text retrieval |
| MRL support | ✓ | ✗ | Varies |

> [!NOTE]
> The dual-encoder design of the embedding model enables offline indexing of document embeddings, making it suitable for retrieval over billions of items. The cross-encoder reranker is applied only to a small shortlist (e.g., top-100), so the computational cost is bounded.

> [!TIP]
> For text-only retrieval baselines, compare against Qwen3-Embedding (text-only counterpart) on MTEB and MMTEB benchmarks.

# Experiments

- **Datasets (Evaluation)**:
  - MMEB-V2 (78 datasets covering image classification, image QA, image retrieval, image grounding, video classification, video retrieval, visual document retrieval)
  - MMTEB (Massive Multimodal Text Embedding Benchmark)
  - JinaVDR (visual document retrieval)
  - ViDoRe v3 (visual document retrieval v3)

- **Hardware**: Not disclosed in available documentation.
- **Optimizer**: Not disclosed in available documentation.

- **Key Results (Qwen3-VL-Embedding-8B)**:
  - MMEB-V2 Overall: 77.9; Image: 80.1; Video: 66.1; VisDoc: 83.3
  - MMTEB Mean (Task): 67.88; Retrieval: 81.08; STS: 75.41; Classification: 71.95

- **Key Results (Qwen3-VL-Reranker-8B)**:
  - MMEB-V2 Overall: 79.2; Image: 80.7; Video: 55.8; VisDoc: 86.3
  - MMTEB: 74.9; JinaVDR: 83.6; ViDoRe v3: 66.7

- **Instruction effect**: Providing a task-specific instruction to the embedding or reranker model yields a 1–5% performance improvement across tasks.

> [!IMPORTANT]
> Qwen3-VL-Reranker-8B (79.2 on MMEB-V2) outperforms Qwen3-VL-Embedding-8B (77.9) due to full cross-attention between query and document, confirming the trade-off between accuracy and retrieval speed inherent to cross-encoder architectures.
