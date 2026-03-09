# Meta Information

- URL: [Diffusion-Pretrained Dense and Contextual Embeddings](https://arxiv.org/abs/2602.11151)
- LICENSE: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- Reference: Eslami, S., Gaiduk, M., Krimmel, M., Milliken, L., Wang, B., & Bykov, D. (2026). Diffusion-Pretrained Dense and Contextual Embeddings. arXiv:2602.11151.

# Diffusion-Pretrained Dense and Contextual Embeddings

## Overview

This paper introduces **pplx-embed**, a family of multilingual text embedding models that applies multi-stage contrastive learning on top of diffusion-pretrained language model backbones. The core motivation is that decoder-only autoregressive (causal) models, commonly used as embedding backbones, process each token with only left-context attention, limiting their ability to form bidirectional, contextually-rich representations. By first pretraining with a masked diffusion objective—which inherently uses bidirectional attention—the backbone captures richer contextual dependencies before any contrastive fine-tuning.

Two model variants are released:

| Model | Description |
|---|---|
| **pplx-embed-v1** (0.6B and 4B) | Standard dense retrieval model |
| **pplx-embed-context-v1** (0.6B and 4B) | Contextual embeddings incorporating global document-level context via late chunking |

**Target users:** Organizations and researchers needing high-performance multilingual dense retrieval over large corpora (billions of documents), particularly where storage efficiency (INT8 / binary quantization) matters.

---

## Architecture and Base Models

- **Base models:** Qwen3-0.6B and Qwen3-4B (decoder-only transformers converted to bidirectional encoders)
- **Embedding dimensions:** $d = 1024$ (0.6B), $d = 2560$ (4B)
- **Pooling:** Mean pooling over all token positions in the final hidden layer
- **Quantization:** Native INT8 training; binary quantization supported post-hoc

> [!NOTE]
> "We leverage bidirectional attention through diffusion-based pretraining to capture comprehensive context within passages."

---

## 1. Continued Diffusion Pretraining (Section 2.1)

### Goal

Convert a causally-masked decoder-only transformer into a bidirectional encoder by continuing pretraining with a masked diffusion (absorbing-state) objective. This enables every token to attend to all other tokens in both directions.

### Forward Process

Given a clean token sequence $x_0 \in \mathcal{V}^L$ (length $L$, vocabulary $\mathcal{V}$), tokens are independently masked to a special `[MASK]` token with probability $t \in [0, 1]$ (continuous time):

$$
q(x_t | x_0) = \prod_{i=1}^{L} \left[ t \cdot \mathbf{1}[\text{mask}] + (1 - t) \cdot \mathbf{1}[x_0^{(i)}] \right]
$$

This is an **absorbing state** diffusion process where the noise level $t$ is uniform-sampled from $[0, 1]$.

### Training Objective

The standard Evidence Lower BOund (ELBO) is optimized:

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_t} \left[ \lambda(t) \cdot \| f_\theta(x_t, t) - x_0 \|^2_{\text{masked}} \right]
$$

where $f_\theta$ is the model predicting original tokens from noisy input, and only masked positions contribute to the loss.

### Pretraining Details

| Parameter | Value |
|---|---|
| Training tokens | ~250 billion |
| Languages | 30 |
| Datasets | FineWeb-Edu, FineWeb2 |
| Training steps | 60,000 |
| Batch size | 1,024 |
| Sequence length | 4,096 tokens |

> [!IMPORTANT]
> The diffusion pretraining does NOT change the model's architecture—it only changes which attention mask is used (full bidirectional vs. causal). After pretraining, the backbone is used as a standard bidirectional encoder for embedding tasks.

---

## 2. Pooling and Quantization (Section 2.2)

### Mean Pooling

For a sequence of $L$ token representations $\mathbf{v}_1, \ldots, \mathbf{v}_L \in \mathbb{R}^d$, the sequence-level embedding is:

$$
\mathbf{e} = \text{MeanPool}(\mathbf{v}_1, \ldots, \mathbf{v}_L) = \frac{1}{L} \sum_{l=1}^{L} \mathbf{v}_l
$$

### Native INT8 Quantization

The float embedding is quantized to signed 8-bit integers in the range $[-127, 127]$ during training:

$$
\mathbf{e}_{\text{INT8}} = \left\lfloor 127 \cdot \tanh\!\left(\frac{1}{L} \sum_{l=1}^{L} \mathbf{v}_l\right) + \frac{1}{2} \right\rfloor
$$

The $\tanh$ nonlinearity ensures the output lies in $(-1, 1)$ before scaling to integer range.

### Binary Quantization

Post-hoc binarization: $\mathbf{e}_{\text{bin}} = \mathbf{1}[\mathbf{e}_{\text{INT8}} \geq 0] \in \{0, 1\}^d$.

- 0.6B: 2–4.4 percentage point nDCG@10 degradation (lower dimensionality $d=1024$ limits resilience)
- 4B: up to 1.6 percentage point degradation (higher dimensionality $d=2560$ provides more redundancy)

---

## 3. Multi-Stage Contrastive Training

The training pipeline has **three sequential stages**, with the contextual variant branching from stage 2.

```
Stage 1: Pair Training (English-only)
    ↓
Stage 2: Pair Training (English + cross-lingual, then full multilingual)
    ↓                                  ↓
Stage 3a: Triplet Training         Stage 3b: Contextual Training
    ↓                                  ↓
pplx-embed-v1                     pplx-embed-context-v1
                                       ↓
                              Model Merging (SLERP)
                                       ↓
                           pplx-embed-v1 (final, merged)
```

### 3.1 Pair Training (Section 2.3)

**Input:** Query $q \in \mathbb{R}^d$, positive document $d^+ \in \mathbb{R}^d$, in-batch negatives $\{d_j^-\}$.

**Loss:** InfoNCE (NT-Xent):

$$
\mathcal{L}_{\text{pair}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\sum_{j} \exp(\text{sim}(q, d_j) / \tau)}
$$

**False negative mitigation:** A masking function filters out in-batch negatives that are semantically close to the positive. Specifically, negative $d_j$ is masked when:

$$
\text{sim}(d^+, d_j) - \text{sim}(q, d_j) < 0.1
$$

**Curriculum:**
1. English-only data
2. English + cross-lingual (query in language A, document in language B)
3. Full multilingual across 60 languages

### 3.2 Contextual Training (Section 2.4)

Targets the **pplx-embed-context-v1** variant, enabling chunk embeddings to incorporate global document context via **late chunking**.

**Late Chunking:** Rather than embedding each chunk independently, first encode the full document, then pool token representations within each chunk's span. This lets chunk embeddings "see" the entire document context.

**Dual Objective:**

$$
\mathcal{L}_{\text{context}} = \alpha \cdot \mathcal{L}_{\text{local}} + \beta \cdot \mathcal{L}_{\text{global}}
$$

| Loss term | Weight | Description |
|---|---|---|
| $\mathcal{L}_{\text{local}}$ | $\alpha = 0.2$ | InfoNCE on chunk-level pairs (in-sequence + in-batch contrasts) |
| $\mathcal{L}_{\text{global}}$ | $\beta \in [0.2, 0.5]$ | InfoNCE on query–document pairs; $\beta$ is cosine-annealed during training |

**Duplicate document masking:** Near-duplicate documents in-batch are identified by MD5 hash of content and excluded from negatives to prevent false negative masking.

### 3.3 Triplet Training (Section 2.5)

**Input:** Query $q$, positive $d^+$, $K$ hard negatives $\{d_k^-\}_{k=1}^K$.

**Loss:** InfoNCE with hard negatives:

$$
\mathcal{L}_{\text{triplet}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{k=1}^{K} \exp(\text{sim}(q, d_k^-) / \tau)}
$$

Hard negatives are mined from higher-quality curated datasets (12 datasets, 92% English, 1% code, 7% multilingual across 15 languages).

### 3.4 Model Merging

Final pplx-embed-v1 is obtained by **Spherical Linear Interpolation (SLERP)** between the contextual model checkpoint (Stage 3b) and the triplet-trained model (Stage 3a):

$$
\text{SLERP}(\theta_{\text{context}}, \theta_{\text{triplet}}, \lambda) = \frac{\sin((1-\lambda)\Omega)}{\sin\Omega} \theta_{\text{context}} + \frac{\sin(\lambda \Omega)}{\sin\Omega} \theta_{\text{triplet}}
$$

where $\Omega$ is the angle between the two parameter vectors in the model weight space.

---

## 4. Training Data (Section 2.6)

### Contrastive Training Data Composition

| Split | Percentage | Description |
|---|---|---|
| English | 65.6% | High-quality English pairs |
| Cross-lingual | 6.7% | Query in one language, document in another |
| Code | 1.0% | Code-related pairs |
| Multilingual | 26.7% | 60 non-English languages |

- Contextual training uses ConTEB training data and synthetic MLDR data
- Synthetic data generated by Qwen3-30B using persona-based prompt approaches

---

## 5. Experiments

### Datasets

| Benchmark | Description | Languages |
|---|---|---|
| MTEB Multilingual v2 | Retrieval across diverse multilingual tasks | Multiple |
| MTEB Code | Code retrieval tasks | Code |
| MIRACL | Multilingual retrieval | 18 languages |
| ConTEB | Contextual retrieval (chunked documents) | English |
| BERGEN | Retrieval-Augmented Generation QA | Multiple |
| ToolRet | Tool/API retrieval | English |
| PPLXQuery2Query (internal) | Query-to-query similarity via URL clustering | Multiple |
| PPLXQuery2Doc (internal) | Query-to-document retrieval over 1B web pages | English + multilingual |

### Hardware

Not explicitly stated in the paper.

### Optimizer

Not explicitly stated in the paper; training uses multi-stage curriculum with progressive data sampling.

### Key Results

**MTEB Multilingual v2 (nDCG@10):**

| Model | Score | Docs/MB |
|---|---|---|
| pplx-embed-4B (INT8) | 69.66% | 390 |
| Qwen3-Embedding-4B | 69.60% | 97 |

pplx-embed achieves comparable accuracy at 4× better storage efficiency due to INT8 quantization.

**ConTEB (nDCG@10, 8 tasks):**

| Model | Score |
|---|---|
| pplx-embed-context-4B | **81.96%** |
| voyage-context-3 | 79.45% |
| Anthropic Contextual | 72.4% |

Sets a new state-of-the-art on ConTEB, with perfect recall on Insurance (100%) and strong performance on Geography (93.04%).

**PPLXQuery2Doc Internal (Recall@1000, 30M corpus):**

| Split | pplx-embed-4B | Qwen3-Embedding-4B |
|---|---|---|
| English | 88.23% | lower |
| Multilingual | 91.66% | substantially lower |

**BERGEN (RAG QA):** pplx-embed-4B achieves the best result on 3 of 5 QA tasks; pplx-embed-0.6B beats the larger Qwen3-Embedding-4B on 3 tasks.

---

## 6. Ablation: Diffusion vs. Autoregressive Backbone (Section 4)

| Backbone | Pooling | English MTEB avg | Notes |
|---|---|---|---|
| Autoregressive | Last token | Baseline | Standard causal LM setup |
| Autoregressive | Mean | Slight improvement | Mean pooling benefits encoder tasks |
| **Diffusion** | **Mean** | **+~1 pp** | Best: bidirectional + mean pooling |

Key finding: The diffusion backbone reduces training loss substantially, and mean pooling is essential for contextual training (last-token pooling cannot represent individual chunks within a sequence).

---

## 7. Comparison with Similar Methods

| Method | Backbone Type | Attention | Contextualization | Quantization |
|---|---|---|---|---|
| pplx-embed (this work) | Diffusion-pretrained decoder | Bidirectional (via diffusion) | Late chunking + dual loss | INT8 native, binary |
| Qwen3-Embedding | Decoder-only (causal) | Unidirectional | None | FP32 / BF16 |
| BGE-M3 | BERT-style encoder | Bidirectional | None | FP32 |
| voyage-context-3 | Unknown | Unknown | Yes (proprietary) | Unknown |
| GTE (GritLM-7B) | Decoder-only | Bidirectional (via instruction tuning) | Limited | FP32 |

> [!IMPORTANT]
> The key differentiator from other encoder-based models (BGE-M3, GTE) is that pplx-embed starts from a decoder and *converts* it through diffusion pretraining, rather than using an encoder from scratch. This allows leveraging the larger pretraining budget of decoder-only LLMs while gaining bidirectional attention.

> [!TIP]
> Late chunking was introduced in the paper "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models" and is the basis for the contextual training approach used here.

---

## 8. Applicability

- **Who:** Teams building large-scale retrieval systems (e.g., search engines, RAG pipelines) that need multilingual support and storage efficiency
- **When:** When the retrieval corpus is web-scale (hundreds of millions to billions of documents) and latency/storage budgets are tight
- **Where:** Production web search, enterprise search, multilingual QA systems, tool/API retrieval systems

The INT8 quantization (4× smaller than FP32) and binary quantization (32× smaller) make this approach especially practical for billion-document indices where embedding storage cost is a bottleneck.
