# Meta Information

- URL: [R²AG: Incorporating Retrieval Information into Retrieval Augmented Generation](https://arxiv.org/abs/2406.13249)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ye, F., Li, S., Zhang, Y., & Chen, L. (2024). R²AG: Incorporating Retrieval Information into Retrieval Augmented Generation. arXiv preprint arXiv:2406.13249.

# R²AG: Incorporating Retrieval Information into Retrieval Augmented Generation

## Background and Motivation

Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant documents from an external corpus and appending them to the input prompt. A fundamental limitation of standard RAG is the **semantic gap** between the retriever and the LLM: the retriever optimizes for query-document similarity, while the LLM's pretraining objective (language modeling) operates over different semantic representations. As a result, the LLM cannot distinguish which retrieved documents are truly relevant from which are noise, even when relevance scores exist.

R²AG addresses this by introducing a lightweight, trainable **R²-Former** module that converts retrieval-side signals (relevance scores, document rankings, neighbor relationships) into soft embeddings that are directly injected into the LLM's input space. Both the retriever and the LLM remain **frozen** during training, so the approach is modular and applicable to a wide range of existing RAG systems.

> [!NOTE]
> "A semantic gap exists between LLMs and retrievers due to differences in their training objectives." — Abstract

## Problem Formulation

Given a query $q$ and a corpus $\mathcal{C}$, the retriever returns a ranked list of $k$ documents $\mathcal{D} = \{d_1, d_2, \ldots, d_k\}$ ordered by relevance to $q$. Standard RAG concatenates all documents into the prompt and passes them to the LLM. R²AG additionally computes a **retrieval information embedding** $e_i \in \mathbb{R}^{d_{\text{LLM}}}$ for each document $d_i$ and prepends it to that document's token embeddings in the LLM's input sequence.

- **Input**: query $q$, top-$k$ retrieved documents $\mathcal{D} = \{d_1, \ldots, d_k\}$, document encoder representations $x^q \in \mathbb{R}^{d_{\text{enc}}}$, $x_i^d \in \mathbb{R}^{d_{\text{enc}}}$
- **Output**: generated answer tokens from the LLM, guided by retrieval information embeddings

## Retrieval Feature Extraction

For each document $d_i$ in the ranked list, three scalar features are computed using the encoder representations of the retriever.

### Relevance Score

Measures semantic alignment between the query and a single document:

```math
\begin{align}
  r_i = \text{sim}(x^q, x_i^d)
\end{align}
```

where $\text{sim}(\cdot, \cdot)$ is dot-product or cosine similarity, $x^q \in \mathbb{R}^{d_{\text{enc}}}$ is the query embedding, and $x_i^d \in \mathbb{R}^{d_{\text{enc}}}$ is the document embedding.

### Precedent Similarity

Captures how document $d_i$ relates to higher-ranked documents (those ranked above it). Higher-ranked documents are aggregated with softmax-normalized weights:

```math
\begin{align}
  \gamma_i = \text{sim}\!\left(x_i^d,\ \sum_{j < i} w_j \cdot x_j^d\right), \quad w_j = \frac{\exp(r_j)}{\sum_{j' < i} \exp(r_{j'})}
\end{align}
```

A document with high precedent similarity shares content with already-seen high-ranked documents and may be redundant.

### Neighbor Similarity

Measures local ranking cohesion by averaging similarity with immediately adjacent documents:

```math
\begin{align}
  \zeta_i = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \text{sim}(x_i^d, x_j^d)
\end{align}
```

where $\mathcal{N}_i$ is the set of neighbors (boundary-aware: $\{i-1, i+1\}$ for middle documents, adjusted at ends).

The three features are concatenated into a list-wise feature vector for document $d_i$: $f_i = [r_i, \gamma_i, \zeta_i] \in \mathbb{R}^3$.

## R²-Former Architecture

The R²-Former is a lightweight Transformer encoder that maps the $k$ feature vectors $\{f_1, \ldots, f_k\}$ into retrieval information embeddings compatible with the LLM's token space.

**Processing pipeline** (for a ranked list of $k$ documents):

1. **Linear embedding**: Project each $f_i \in \mathbb{R}^3$ to a higher-dimensional space $h_i^{(0)} \in \mathbb{R}^{h_1}$ via a learned weight matrix $W_{\text{emb}} \in \mathbb{R}^{h_1 \times 3}$.
2. **Positional encoding**: Add trainable position embeddings $p_i \in \mathbb{R}^{h_1}$ to inject rank-order information.
3. **Transformer encoder**: Apply $L$ layers of multi-head self-attention and feed-forward networks to capture inter-document dependencies:

```math
\begin{align}
  H = \text{TransformerEncoder}(h_1^{(0)}, \ldots, h_k^{(0)}) \in \mathbb{R}^{k \times h_1}
\end{align}
```

4. **Projection to LLM space**: Map each hidden state $H_i \in \mathbb{R}^{h_1}$ to the LLM's embedding dimension $d_{\text{LLM}}$ via $W_{\text{proj}} \in \mathbb{R}^{d_{\text{LLM}} \times h_1}$, producing $e_i \in \mathbb{R}^{d_{\text{LLM}}}$.

> [!IMPORTANT]
> The R²-Former is the **only trainable component** during the R²AG training phase. Both the retriever encoder and the LLM backbone are kept frozen.

## Retrieval-Aware Prompting

The retrieval information embedding $e_i$ for document $d_i$ is prepended to that document's token embeddings within the LLM's input sequence. Concretely, if the LLM input normally embeds the concatenation `[query tokens] [doc_1 tokens] ... [doc_k tokens]`, R²AG modifies it to:

```
[query tokens] [e_1] [doc_1 tokens] [e_2] [doc_2 tokens] ... [e_k] [doc_k tokens]
```

Each $e_i$ acts as a **retrieval anchor**: the LLM attends to it when processing $d_i$'s content, using the encoded relevance signal to up-weight or down-weight the document's contribution to the final answer.

> [!NOTE]
> Attention visualization confirms that retrieval information embeddings receive higher attention scores even in deeper LLM layers, validating the anchoring effect.

## Training Objectives

R²AG is trained with **instruction fine-tuning** on a QA training set using two combined losses:

### Query-Document Matching (QDM) Loss

Binary cross-entropy predicting whether each retrieved document is relevant (label $s_i \in \{0, 1\}$):

```math
\begin{align}
  \mathcal{L}_{\text{QDM}} = -\sum_{i=1}^{k} \left[ s_i \log \hat{s}_i + (1 - s_i) \log(1 - \hat{s}_i) \right]
\end{align}
```

where $\hat{s}_i$ is the predicted relevance probability derived from the R²-Former output $H_i$.

### Language Modeling (LM) Loss

Standard autoregressive token prediction loss over the target answer tokens $y = (y_1, \ldots, y_T)$:

```math
\begin{align}
  \mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, q, \mathcal{D}, e_{1:k})
\end{align}
```

### Joint Loss

```math
\begin{align}
  \mathcal{L} = \mathcal{L}_{\text{QDM}} + \mathcal{L}_{\text{LM}}
\end{align}
```

## Comparison with Similar Approaches

| Method | Retriever Frozen | LLM Frozen | Preserves Document Content | Semantic Gap Bridging |
|---|---|---|---|---|
| Standard RAG | ✓ | ✓ | ✓ | ✗ |
| RECOMP | ✓ | ✓ | ✗ (compresses docs) | Partial |
| LongLLMLingua | ✓ | ✓ | ✗ (compresses docs) | Partial |
| RAFT | ✓ | ✗ (fine-tuned) | ✓ | Partial |
| **R²AG** | **✓** | **✓** | **✓** | **✓ (explicit embeddings)** |

Unlike compression-based methods (RECOMP, LongLLMLingua) that condense retrieved documents and risk losing information, R²AG preserves each document in full. Unlike RAFT, it does not require LLM fine-tuning and is thus plug-and-play for any frozen LLM. R²AG can also be combined with RAFT for further gains.

> [!TIP]
> RECOMP and LongLLMLingua reduce token count to fit context windows but may discard evidence. R²AG provides guidance signals without altering document content.

## Applicability

R²AG is best suited for scenarios where:

- **Who**: NLP practitioners deploying RAG systems who want improved answer quality without retraining or replacing the LLM.
- **When**: The retriever returns noisy or mixed-relevance documents (common in open-domain QA, multi-hop QA).
- **Where**: Any encoder-based dense retriever (e.g., DPR, E5) is used; sparse retrievers (BM25) are not directly supported because they lack dense document embeddings.

# Experiments

- **Datasets**:
  - *Natural Questions (NQ)*: Open-domain single-hop QA. Three settings: NQ-10, NQ-20, NQ-30 (top-10, 20, 30 retrieved documents).
  - *HotpotQA*: Multi-hop QA requiring reasoning over multiple documents.
  - *MuSiQue*: Multi-hop QA with distracting documents.
  - *2WikiMultiHopQA*: Multi-hop QA across Wikipedia articles.
  - *DuReader*: Chinese open-domain QA dataset.
- **Retrievers**: E5-large-v2, BM25 (for ablation only), DPR
- **LLMs**: LLaMA2-7B (primary), LLaMA2-13B, Mistral-7B, ChatGPT (closed-source)
- **Baselines**: Standard RAG, RECOMP, LongLLMLingua, RAFT, Selective-Context
- **Hardware**: Not explicitly stated
- **Optimizer**: AdamW (standard for instruction fine-tuning)
- **Results**:
  - R²AG achieves 0.693 Exact Match on NQ-10 vs. 0.386 for standard RAG with LLaMA2-7B (+79.5% relative improvement).
  - R²AG + RAFT achieves the best overall performance across most benchmarks.
  - Inference latency overhead is only **0.8%** compared to standard RAG, due to the lightweight R²-Former.
  - Ablation: removing relevance features degrades EM by 2.45–3.73%; removing neighbor similarity degrades EM by 2.93–9.78%; removing QDM loss degrades EM by 0.27–7.07%.
