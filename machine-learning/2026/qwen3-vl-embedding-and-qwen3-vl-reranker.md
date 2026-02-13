# Meta Information

- URL: [Qwen3-VL-Embedding and Qwen3-VL-Reranker](https://arxiv.org/abs/2601.04720)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A., Liu, D., Zhou, J., & Lin, J. (2026). Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking. arXiv:2601.04720.

---

# Qwen3-VL-Embedding and Qwen3-VL-Reranker

## 1. Introduction

Qwen3-VL-Embedding and Qwen3-VL-Reranker form a two-stage multimodal retrieval pipeline that jointly addresses embedding generation and fine-grained relevance scoring over text, images, document images, and video. The system is built on the Qwen3-VL vision-language foundation and supports over 30 languages with context lengths up to 32,768 tokens.

**Who uses this / when / where:**
- Teams building multimodal retrieval-augmented generation (RAG) systems that must retrieve over heterogeneous corpora containing text, images, PDFs, and video frames.
- Applied when a dense embedding index (ANN search) is built first and then optionally re-scored with the cross-encoder reranker for precision-critical scenarios.
- Deployable wherever GPU inference is available; quantized embeddings and flexible Matryoshka dimensions reduce storage and compute footprint for large-scale deployments.

---

## 2. Model Architecture

### 2.1 Embedding Model (Dual-Tower)

**Input:** A single-modal or mixed-modal object — any combination of text strings, image paths/URLs/PIL instances, and video paths/frame sequences.

**Output:** A high-dimensional dense vector extracted from the hidden state of the `[EOS]` token at the last layer of the backbone.

The embedding model follows a dual-tower design so that query and document can be encoded independently, enabling scalable approximate nearest-neighbor retrieval.

**Backbone and dimensions:**

| Model | Layers | Sequence Length | Embedding Dimension |
|---|---|---|---|
| Qwen3-VL-Embedding-2B | 28 | 32,768 | 2,048 |
| Qwen3-VL-Embedding-8B | 36 | 32,768 | 4,096 |

The vision encoder processes images with hierarchical resolution handling; a projection layer maps visual tokens into the language model's embedding space. The final embedding is the normalized hidden state of the `[EOS]` token:

```math
\mathbf{e} = \frac{\mathbf{h}_{\text{EOS}}}{\|\mathbf{h}_{\text{EOS}}\|}
```

where $\mathbf{h}_{\text{EOS}}$ is the last-layer hidden state at the `[EOS]` position.

### 2.2 Reranker Model (Single-Tower / Cross-Encoder)

**Input:** A concatenated pair `(Query, Document)` passed through a single encoder.

**Output:** A scalar relevance score derived from the generation probability of special tokens `yes` and `no`.

The reranker uses cross-attention between query and document tokens for deeper, finer-grained inter-modal interaction. The relevance score is computed as:

```math
s(q, d) = P(\text{"yes"} \mid q, d) = \frac{\exp(\text{logit}_{\text{yes}})}{\exp(\text{logit}_{\text{yes}}) + \exp(\text{logit}_{\text{no}})}
```

**Backbone and dimensions:**

| Model | Layers | Sequence Length |
|---|---|---|
| Qwen3-VL-Reranker-2B | 28 | 32,768 |
| Qwen3-VL-Reranker-8B | 36 | 32,768 |

---

## 3. Training Methodology

### 3.1 Multi-Stage Training for the Embedding Model

Training proceeds in two sequential stages:

**Stage 1 — Large-Scale Contrastive Pre-training:**
The model is trained with an InfoNCE / in-batch contrastive loss over a large collection of multimodal paired data. For a batch of $N$ query-document pairs $\{(q_i, d_i)\}_{i=1}^{N}$, the contrastive loss for query $q_i$ is:

```math
\mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(q_i, d_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q_i, d_j) / \tau)}
```

where $\text{sim}(\cdot, \cdot)$ denotes cosine similarity and $\tau$ is the temperature hyperparameter.

**Stage 2 — Reranking Model Distillation:**
The trained reranker provides soft relevance scores that are used as teacher signals to further refine the embedding model. This cross-model distillation bridges the precision gap between embedding-based retrieval and cross-encoder reranking, enabling the embedding model to encode more nuanced relevance signals without sacrificing the independence of dual-tower inference.

### 3.2 Matryoshka Representation Learning (MRL)

MRL trains the model to produce nested, meaningful representations at multiple vector dimensions simultaneously. For a target embedding dimension $d_m \in \{d_1, d_2, \ldots, d_M\}$ (with $d_1 < d_2 < \cdots < d_M = D$), the total MRL loss is:

```math
\mathcal{L}_{\text{MRL}} = \sum_{m=1}^{M} w_m \cdot \mathcal{L}_{\text{contrast}}^{(d_m)}
```

where $\mathcal{L}_{\text{contrast}}^{(d_m)}$ is the contrastive loss computed using only the first $d_m$ dimensions of the embedding, and $w_m$ are dimension-level weights. At inference time, the user selects any $d_m$; the model degrades gracefully as dimension decreases.

### 3.3 LoRA Fine-Tuning Configuration

Both models support parameter-efficient adaptation via LoRA:

| Hyperparameter | Value |
|---|---|
| Rank | 32 |
| Alpha | 32 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `up_proj`, `down_proj`, `gate_proj` |

---

## 4. Input/Output Specifications

### 4.1 Multimodal Input Object

Each input is a dictionary with the following optional keys:

| Key | Type | Description |
|---|---|---|
| `text` | `str` or `List[str]` | Text content |
| `image` | local path, URL, `PIL.Image`, or list thereof | Still images |
| `video` | file path, URL, frame sequence, or combined list | Video input |

### 4.2 Processing Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `fps` | 1.0 | Frames per second for video sampling |
| `max_frames` | 64 | Maximum number of sampled video frames |
| `max_length` | 8,192 | Token context window for processing |
| `min_pixels` | 4,096 | Minimum image resolution (pixels) |
| `max_pixels` | 1,843,200 | Maximum image resolution (1280 x 1440) |
| `total_pixels` | 7,864,320 | Maximum aggregate pixels across video frames |

### 4.3 Task-Specific Instructions

The embedding model supports instruction-aware encoding. The query is prepended with a natural-language instruction to bias the representation toward a specific retrieval task (e.g., "Retrieve a document that answers this question"). The document side uses either a generic or task-specific instruction. This design follows the GTE/E5-Instruct paradigm.

---

## 5. Experimental Results

### 5.1 MMEB-V2 Benchmark (Embedding Model)

Qwen3-VL-Embedding-8B achieved rank #1 among all models on MMEB-V2 as of January 8, 2025:

| Modality | Datasets | Score |
|---|---|---|
| Image (all tasks) | 36 | 80.1% |
| Video (all tasks) | 18 | 67.1% |
| Visual Documents (VisDoc) | 24 | 82.4% |
| All tasks (overall) | 78 | 77.8% |

### 5.2 MMTEB Benchmark (Embedding Model)

| Metric | Score |
|---|---|
| Mean task score | 67.9% |
| Mean type score | 58.9% |
| Retrieval | 69.4% |
| Reranking | 65.7% |

### 5.3 MMEB-v2 Retrieval with Reranker

Qwen3-VL-Reranker-8B applied on top of retrieval results:

| Retrieval Split | Score |
|---|---|
| Average (all) | 79.2% |
| Image retrieval | 78.2% |
| Video retrieval | 61.0% |
| VisDoc retrieval | 85.8% |

### 5.4 Evaluation Datasets

| Domain | Benchmark Datasets |
|---|---|
| Text | AG News, SQuAD, MS MARCO |
| Image | CIFAR-10, VQAv2, MS COCO |
| Visual Documents | VDRv1, VDRv2, ViDoRe |
| Video | Multiple video QA and retrieval datasets (18 total in MMEB-V2) |

---

## 6. Algorithm Summary

### 6.1 Embedding Inference (Pseudocode)

```
Input:  List of multimodal objects [(text_i, image_i, video_i)], instruction I
Output: Normalized embedding vectors [e_1, ..., e_N]

For each input object o_i:
    1. Prepend instruction I to text_i to form instructed input
    2. Process images through vision encoder + projection layer
    3. Sample video at fps frames/sec, cap at max_frames
    4. Tokenize and concatenate all modalities up to max_length tokens
    5. Run forward pass through Qwen3-VL backbone
    6. Extract hidden state h_EOS at the [EOS] token position (last layer)
    7. Normalize: e_i = h_EOS / ||h_EOS||
    8. (Optional) Truncate to Matryoshka sub-dimension d_m
Return [e_1, ..., e_N]
```

### 6.2 Reranker Inference (Pseudocode)

```
Input:  Query q, list of candidate documents [d_1, ..., d_K]
Output: Ranked list of documents by relevance score

For each candidate d_j:
    1. Concatenate (q, d_j) into a single cross-encoder input
    2. Run forward pass through Qwen3-VL-Reranker backbone
    3. Extract logits for special tokens "yes" and "no"
    4. Compute score: s_j = softmax([logit_yes, logit_no])[0]

Sort documents by s_j descending
Return ranked [d_1, ..., d_K]
```

---

## 7. Key Features and Design Choices

### 7.1 EOS Token as Embedding

Extracting the representation from the `[EOS]` token rather than from a `[CLS]` token or mean pooling is consistent with decoder-only LLM embedding approaches (e.g., LLM2Vec, E5-Mistral). The autoregressive attention ensures the `[EOS]` position attends to all preceding tokens, producing a holistic sequence summary.

### 7.2 Yes/No Probability for Reranking

Using generation probability of `"yes"` vs. `"no"` as the relevance score allows the reranker to leverage the language model's token probability calibration without requiring a separate classification head. This is the same technique used in RankGPT and MonoT5-style listwise rerankers.

### 7.3 Embedding Quantization

Both embedding models support post-hoc quantization of the float32 embedding vectors (e.g., int8, binary). Combined with MRL dimension reduction, this can reduce memory footprint by 32x or more while preserving much of the retrieval quality.

### 7.4 Modality Coverage

| Modality | Supported Formats |
|---|---|
| Text | Any UTF-8 string, multilingual (30+ languages) |
| Image | JPEG, PNG, local path, HTTP URL, PIL.Image |
| Document images | PDF page renders, scanned documents |
| Video | MP4, frame directories, HTTP URL, frame sequences |

---

## 8. Application Use Cases

- **Multimodal RAG:** Index heterogeneous corpora (PDFs, images, video keyframes) into a single vector store; retrieve with Qwen3-VL-Embedding and re-rank with Qwen3-VL-Reranker.
- **Visual document retrieval:** Search over scanned/rendered documents (ViDoRe, VDRv2).
- **Image-text retrieval:** Cross-modal search between natural language queries and image databases.
- **Video-text matching:** Retrieve relevant video clips from text queries using sampled frame embeddings.
- **Multimodal content clustering:** Group semantically similar content across modalities.
- **Multilingual visual search:** Cross-lingual retrieval over image or document collections in 30+ languages.

---

## 9. Comparison with Similar Models

| Aspect | Qwen3-VL-Embedding | CLIP / BLIP-2 | ColPali | E5-Mistral |
|---|---|---|---|---|
| Modalities | Text + Image + Video + Docs | Text + Image | Image + Text | Text only |
| Cross-encoder reranker | Yes (paired model) | No | No | No |
| MRL support | Yes | No | No | Yes |
| Max context | 32,768 tokens | ~77 tokens | ~1,000 tokens | ~32,768 tokens |
| Embedding extraction | EOS token | CLS token | Late interaction | EOS token |
| Multilingual | 30+ languages | Limited | Limited | Yes |

---

## 10. Resources

- Paper: https://arxiv.org/abs/2601.04720
- GitHub: https://github.com/QwenLM/Qwen3-VL-Embedding
- Hugging Face: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B (embedding), https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B (reranker)
- ModelScope: Available under Alibaba ModelScope
