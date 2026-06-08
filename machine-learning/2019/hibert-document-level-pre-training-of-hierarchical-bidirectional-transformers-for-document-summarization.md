# Meta Information

- URL: [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/abs/1905.06566)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, X., Wei, F., & Zhou, M. (2019). HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization. ACL 2019.

# HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization

HIBERT is a hierarchically structured pre-trained language model for extractive document summarization. It addresses a fundamental limitation of sentence-level label noise in supervised extractive summarization by using large-scale unlabeled text for pre-training. The model is applicable when: (1) labeled summarization data is limited or noisy, (2) the input consists of multi-sentence documents, and (3) the goal is to select complete sentences as summary content rather than generate new text.

## Problem Setting

Extractive summarization frames the task as sequence labeling: given a document $\mathcal{D} = (S_1, S_2, \ldots, S_n)$ with $n$ sentences, assign a binary label $y_i \in \{0, 1\}$ to each sentence $S_i$ indicating whether it belongs to the summary. The difficulty is that sentence-level labels derived from ROUGE-based matching with reference summaries are noisy and imperfect proxies for actual importance.

> [!NOTE]
> "Training with such inaccurate labels is not ideal. In this paper, we propose to pre-train HIBERT using unlabeled data and fine-tune it on labeled (but noisy) data."

## Model Architecture (Section 3.1)

HIBERT uses two nested Transformer encoders to build document representations hierarchically.

### Sentence Encoder

Each sentence $S_k = (w_1^k, w_2^k, \ldots, w_{|S_k|}^k)$ is processed by a Transformer encoder that outputs contextual word representations. A special EOS token appended to the end of each sentence serves as the sentence-level representation:

$$\mathbf{s}_k = \text{SentenceEncoder}(S_k)[-1] \in \mathbb{R}^{d}$$

where $d$ is the hidden dimension. The encoder uses standard multi-head self-attention with positional embeddings at the word level.

### Document Encoder

The sequence of sentence representations $(\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_n) \in \mathbb{R}^{n \times d}$ is fed into a second Transformer encoder that operates at the sentence level. This produces document-contextualized sentence representations:

$$\mathbf{d}_i = \text{DocumentEncoder}(\mathbf{s}_1, \ldots, \mathbf{s}_n)[i] \in \mathbb{R}^{d}$$

Positional embeddings at the sentence level encode the order of sentences in the document.

| Component | Input | Output | Depth |
|---|---|---|---|
| Sentence Encoder | $(|S_k|) \times d_{\text{vocab}}$ (one-hot) | $d$ (EOS hidden state) | $L_s$ layers |
| Document Encoder | $n \times d$ (sentence vectors) | $n \times d$ (contextual) | $L_d$ layers |

> [!TIP]
> This hierarchical design mirrors the structure of documents themselves—words form sentences, sentences form documents—allowing each encoder to specialize at its own granularity level.

## Pre-training Objective (Section 3.2)

HIBERT is pre-trained using **Masked Document Modeling (MDM)**, a document-level analogue of BERT's Masked Language Modeling.

### Masking Strategy

For each training document, 15% of sentences are randomly selected for masking:
- **80%** of selected sentences are replaced by a special `[MASK]` token at the sentence level
- **10%** are left unchanged (helps the model handle unmasked sentences at test time)
- **10%** are replaced by a random sentence from the corpus (adds noise robustness)

### Sentence Prediction

For each masked sentence $S_k$ in the corrupted document $\tilde{\mathcal{D}}$, the model must reconstruct the original sentence word-by-word. A Transformer decoder with cross-attention over document-level representations is used:

$$p(\mathcal{M} \mid \tilde{\mathcal{D}}) = \prod_{k \in \mathcal{K}} \prod_{j=1}^{|S_k|} p(w_j^k \mid w_{0:j-1}^k, \tilde{\mathcal{D}})$$

where $\mathcal{K}$ is the set of masked sentence indices and $w_0^k$ is a BOS (beginning-of-sentence) token. The decoder attends to the document encoder outputs via cross-attention, conditioning word generation on the full document context.

**Pseudocode for MDM pre-training:**
```
Input: Document D = (S_1, ..., S_n)
1. Select subset K ⊆ {1,...,n}, |K| ≈ 0.15n
2. For k in K:
   - With prob 0.8: replace S_k with [MASK] → S̃_k
   - With prob 0.1: keep S_k unchanged → S̃_k
   - With prob 0.1: sample random sentence → S̃_k
3. Encode D̃ = (S̃_1, ..., S̃_n) with SentenceEncoder and DocumentEncoder → {d_i}
4. For each k in K, decode S_k word-by-word using TransformerDecoder(d_k)
5. Minimize cross-entropy loss over masked sentences
```

> [!IMPORTANT]
> Unlike BERT which masks individual tokens within a single sequence, HIBERT masks entire sentences and requires reconstructing them given document-level context. This forces the model to learn inter-sentence coherence and document structure.

## Fine-tuning for Extractive Summarization (Section 3.3)

After pre-training, the document encoder outputs are used for sentence-level binary classification. A linear projection followed by softmax assigns inclusion probabilities:

$$p(y_i = 1 \mid \mathcal{D}) = \text{softmax}(\mathbf{W}_S \cdot \mathbf{d}_i)$$

where $\mathbf{W}_S \in \mathbb{R}^{2 \times d}$ is a learned weight matrix. At test time, sentences are ranked by $p(y_i = 1)$ and the top-$k$ are selected as the summary (trigram blocking is used to avoid redundancy).

## Two-Stage Pre-training (Section 4)

A key empirical finding is that a two-stage pre-training strategy outperforms single-stage training:

1. **Stage 1 — Open-domain**: Pre-train on GIGA-CM (6.6M documents from Gigaword + CNN/DailyMail) without domain restriction
2. **Stage 2 — In-domain**: Continue pre-training on the target domain corpus (e.g., CNN/DailyMail or NYT)
3. **Fine-tuning**: Supervised training on labeled summarization data

Ablation studies confirm both stages are necessary: skipping Stage 1 loses diverse general knowledge, and skipping Stage 2 loses domain-specific sentence structure patterns.

## Comparison with Similar Methods

| Method | Architecture | Pre-training Level | Pre-training Objective | Hierarchy |
|---|---|---|---|---|
| BERT (Devlin et al., 2019) | Transformer | Word/token | MLM + NSP | None (flat) |
| HIBERT (this paper) | Hierarchical Transformer | Sentence/document | MDM (masked sentence) | Two-level |
| BertSum (Liu, 2019) | BERT + linear | Token | MLM + NSP (from BERT) | None |
| SummaRuNNer | Hierarchical BiLSTM | None | — | Two-level |

> [!NOTE]
> HIBERT uses ~50% fewer parameters than BERT-Base (used in BertSum) while achieving comparable or superior performance, because it allocates model capacity to sentence- and document-level modeling rather than a single flat encoder.

### Difference from BERT for Summarization

BertSum (Liu 2019) applies the pretrained BERT directly to concatenated document tokens, using `[CLS]` token representations for sentence classification. HIBERT differs by:
1. Pre-training explicitly at the document level (not just word level)
2. Using a two-level hierarchy that separates intra-sentence and inter-sentence modeling
3. The pre-training task (predict masked sentences) directly trains the model for sentence-level semantics

## Experiments

- **Datasets**:
  - CNN/DailyMail: 287,226 train / 13,368 validation / 11,490 test documents
  - NYT50: 137,778 train / 17,222 validation / 17,223 test documents
  - GIGA-CM: 6.6M documents for pre-training (Gigaword + CNN/DailyMail combined)
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L (F1)
- **Model sizes**:
  - HibertS: 6-layer Sentence Encoder + 2-layer Document Encoder, $d = 512$
  - HibertM: 6-layer Sentence Encoder + 2-layer Document Encoder, $d = 768$
- **Key results** (CNN/DailyMail):
  - HibertM achieves 42.37 / 19.95 / 38.83 (R1/R2/RL), outperforming ORACLE upper bound baselines for some metrics
  - Pre-training improves over non-pretrained hierarchical transformer by ~1.25 ROUGE points
  - HibertM outperforms BertSum (which uses full BERT-Base) by ~0.5 ROUGE with half the parameters
- **Key results** (NYT50):
  - HibertM achieves 49.47 / 30.11 / 41.63, a ~2.0 ROUGE improvement over the non-pretrained baseline
- **Human evaluation**: On 20 CNN/DailyMail test samples, HibertM was ranked best by annotators in 30% of cases; significantly better than all systems except human references ($p < 0.05$)
