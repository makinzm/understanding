# Meta Information

- URL: [[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, pp. 4171–4186.

# Introduction

BERT (Bidirectional Encoder Representations from Transformers) is a language representation model that pre-trains deep bidirectional representations by jointly conditioning on both left and right context in all layers. Unlike ELMo (which concatenates independently trained left-to-right and right-to-left LSTMs) and GPT (which uses only left-to-right context), BERT uses a masked language model (MLM) objective that enables true bidirectional pre-training.

BERT is applicable to any NLP practitioner who needs a general-purpose language encoder. It suits both sentence-level tasks (e.g., sentiment analysis, natural language inference) and token-level tasks (e.g., named entity recognition, question answering) with minimal architecture modification—only a single output layer is added on top of the pre-trained model.

> [!NOTE]
> "BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures."

---

# Model Architecture

BERT uses the multi-layer bidirectional Transformer encoder from Vaswani et al. (2017). Two model sizes are defined:

| Model | Layers ($L$) | Hidden size ($H$) | Attention heads ($A$) | Parameters |
|-------|-------------|-------------------|----------------------|------------|
| $BERT_{\text{BASE}}$ | 12 | 768 | 12 | 110M |
| $BERT_{\text{LARGE}}$ | 24 | 1024 | 16 | 340M |

$BERT_{\text{BASE}}$ uses the same model size as OpenAI GPT for comparison purposes. The feed-forward intermediate size is $4H$ (i.e., 3072 for BASE, 4096 for LARGE).

> [!IMPORTANT]
> The key architectural difference between BERT and GPT is NOT the Transformer itself, but the attention mask: BERT uses bidirectional (unrestricted) self-attention, while GPT constrains attention to left-context only. Both use the same Transformer building blocks.

---

# Input Representation

BERT's input representation can handle both a single sentence and a pair of sentences (e.g., Question + Passage) in one token sequence.

## Tokenization

BERT uses WordPiece embeddings with a 30,000 token vocabulary.

## Special Tokens

- `[CLS]`: Prepended to every input sequence. The final hidden state of this token ($T_{\text{CLS}} \in \mathbb{R}^{H}$) serves as the aggregate sequence representation for classification tasks.
- `[SEP]`: Inserted between and after sentences to separate them.

## Embedding Summation

For a given token, the input representation is constructed by summing three embeddings:

$$E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

- $E_{\text{token}} \in \mathbb{R}^{H}$: WordPiece token embedding
- $E_{\text{segment}} \in \mathbb{R}^{H}$: Learned embedding indicating sentence A or sentence B
- $E_{\text{position}} \in \mathbb{R}^{H}$: Learned positional embedding (max sequence length = 512)

The full input sequence has shape $\mathbb{R}^{n \times H}$ where $n$ is the sequence length.

---

# Pre-training

BERT is pre-trained on two unsupervised tasks simultaneously.

## Task 1: Masked Language Model (MLM)

Standard language models can only be trained left-to-right or right-to-left, because bidirectional conditioning would allow each word to "see itself" in a multi-layer architecture. The MLM objective solves this by masking tokens and predicting them from context.

### Masking Procedure

For each input sequence, 15% of WordPiece tokens are selected at random. For each selected token position $i$:

1. With probability 0.8: replace token $i$ with `[MASK]`
2. With probability 0.1: replace token $i$ with a random token from the vocabulary
3. With probability 0.1: keep token $i$ unchanged

### Prediction

The final hidden vector for each masked position $T_i \in \mathbb{R}^{H}$ is fed into an output softmax over the vocabulary:

$$P(w \mid T_i) = \text{softmax}(T_i W_e^T + b) \in \mathbb{R}^{|V|}$$

where $W_e \in \mathbb{R}^{|V| \times H}$ is the token embedding matrix (shared with the input embedding) and $b \in \mathbb{R}^{|V|}$ is a bias term.

The loss is the cross-entropy loss computed only over the masked positions.

> [!IMPORTANT]
> The 80/10/10 masking strategy mitigates the mismatch between pre-training (where `[MASK]` tokens appear) and fine-tuning (where they do not). The 10% random replacement forces the model to maintain a distributional representation for every input token, not just the masked ones.

### Comparison with Denoising Auto-Encoders

> [!NOTE]
> "The Masked LM only predicts 15% of tokens in each batch, which suggests that more pre-training steps may be needed for the model to converge. [...] we show that MLM does converge marginally slower than a left-to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost."

## Task 2: Next Sentence Prediction (NSP)

Many downstream tasks (e.g., Question Answering, Natural Language Inference) require understanding the relationship between two sentences. NSP is a binary classification task:

- **Input**: Two sentences A and B
- **Positive (IsNext)**: B is the actual next sentence following A in the corpus (50% of training pairs)
- **Negative (NotNext)**: B is a random sentence from the corpus (50% of training pairs)
- **Output**: Binary prediction from the `[CLS]` token's final hidden state $T_{\text{CLS}} \in \mathbb{R}^{H}$

$$P(\text{IsNext} \mid T_{\text{CLS}}) = \text{softmax}(T_{\text{CLS}} W_{\text{NSP}}) \in \mathbb{R}^{2}$$

where $W_{\text{NSP}} \in \mathbb{R}^{H \times 2}$.

## Pre-training Data

| Corpus | Size |
|--------|------|
| BooksCorpus | 800M words |
| English Wikipedia (text only) | 2,500M words |
| **Total** | **3,300M words** |

> [!IMPORTANT]
> Document-level corpus is used (not sentence-shuffled) to enable extraction of long contiguous sequences, which is critical for learning long-range dependencies.

## Pre-training Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 256 sequences |
| Max sequence length | 512 tokens (but 90% of steps use 128) |
| Tokens per batch | ~128,000 |
| Total steps | 1,000,000 (~40 epochs over 3.3B word corpus) |
| Optimizer | Adam ($\text{lr}=10^{-4}$, $\beta_1=0.9$, $\beta_2=0.999$) |
| L2 weight decay | 0.01 |
| LR warmup | First 10,000 steps, linear decay after |
| Dropout | 0.1 on all layers |
| Activation | GELU |

- Hardware: 4 Cloud TPUs (16 TPU chips) for BERT$_{\text{BASE}}$, 16 Cloud TPUs (64 TPU chips) for BERT$_{\text{LARGE}}$
- Training time: 4 days for BERT$_{\text{BASE}}$, 4 days for BERT$_{\text{LARGE}}$ (on respective hardware)

---

# Fine-tuning

Fine-tuning is straightforward: all parameters of the pre-trained BERT model are fine-tuned end-to-end on labeled data from a downstream task. Only a task-specific output layer is added.

## Fine-tuning for Different Task Types

### 1. Sentence Pair Classification (e.g., MNLI, QQP)

- Input: `[CLS] Sentence A [SEP] Sentence B [SEP]` → $\mathbb{R}^{n \times H}$
- Use $T_{\text{CLS}} \in \mathbb{R}^{H}$ as the aggregate representation
- Classification: $y = \text{softmax}(T_{\text{CLS}} W) \in \mathbb{R}^{K}$ where $W \in \mathbb{R}^{H \times K}$, $K$ = number of classes

### 2. Single Sentence Classification (e.g., SST-2, CoLA)

- Input: `[CLS] Sentence [SEP]` → $\mathbb{R}^{n \times H}$
- Same as above with $K$ classes

### 3. Question Answering — Extractive Span (e.g., SQuAD)

- Input: `[CLS] Question [SEP] Passage [SEP]` → $\mathbb{R}^{n \times H}$
- Introduce start vector $S \in \mathbb{R}^{H}$ and end vector $E \in \mathbb{R}^{H}$ (learned during fine-tuning)
- Probability that token $i$ is the answer start:

$$P_{\text{start}}(i) = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$$

- Probability that token $i$ is the answer end:

$$P_{\text{end}}(i) = \frac{e^{E \cdot T_i}}{\sum_j e^{E \cdot T_j}}$$

- Candidate span score: $S \cdot T_i + E \cdot T_j$ for span $(i, j)$ where $j \geq i$
- For SQuAD 2.0 (unanswerable questions): the null answer span is `[CLS]`, i.e., start=0, end=0. A question is predicted as unanswerable when $s_{\text{null}} > \max_{i,j}(S \cdot T_i + E \cdot T_j) - \tau$, where $\tau$ is a threshold tuned on the dev set.

### 4. Token-level Classification (e.g., NER)

- Input: `[CLS] Sentence [SEP]` → $\mathbb{R}^{n \times H}$
- Each token representation $T_i \in \mathbb{R}^{H}$ is classified independently
- For WordPiece sub-tokens, only the first sub-token's representation is used for prediction

## Fine-tuning Hyperparameters

| Hyperparameter | Range |
|---------------|-------|
| Batch size | 16, 32 |
| Learning rate | 5e-5, 3e-5, 2e-5 |
| Epochs | 2, 3, 4 |
| Dropout | 0.1 (unchanged from pre-training) |

> [!NOTE]
> "All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model."

---

# Experiments

## Datasets

| Task | Dataset | Type | Train Size | Metric |
|------|---------|------|-----------|--------|
| NLI | MNLI | Sentence pair classification | 393K | Accuracy |
| Paraphrase | QQP | Sentence pair classification | 364K | F1 |
| NLI | QNLI | Sentence pair classification | 105K | Accuracy |
| Sentiment | SST-2 | Single sentence classification | 67K | Accuracy |
| Acceptability | CoLA | Single sentence classification | 8.5K | Matthews corr. |
| Similarity | STS-B | Sentence pair regression | 7K | Spearman corr. |
| Paraphrase | MRPC | Sentence pair classification | 3.7K | F1 |
| NLI | RTE | Sentence pair classification | 2.5K | Accuracy |
| QA | SQuAD v1.1 | Extractive span | 100K | EM / F1 |
| QA | SQuAD v2.0 | Extractive span + unanswerable | 130K | EM / F1 |
| Commonsense | SWAG | Multiple choice | 113K | Accuracy |

## GLUE Results

$BERT_{\text{LARGE}}$ achieves an official GLUE score of **80.5** (7.7 points above GPT's 72.8).

---

# Ablation Studies

## Effect of Pre-training Tasks

- Removing NSP hurts performance significantly on QNLI (-3.5), MNLI (-0.5), and SQuAD (-0.6).
- Switching from bidirectional MLM to left-to-right (LTR) degrades all tasks, with SQuAD dropping by 10.7 F1 points.
- Adding a BiLSTM on top of LTR partially recovers SQuAD (+7.1) but still trails the bidirectional model by 3.6 F1.

> [!IMPORTANT]
> ELMo-style concatenation of separate LTR and RTL models is weaker than the deep bidirectional approach because the LTR model cannot condition on right context and vice versa. BERT's MLM allows every layer to jointly attend to both directions.

## Effect of Model Size

Larger models yield consistent improvements even on small tasks (e.g., MRPC with 3,700 training examples), provided the model has been sufficiently pre-trained.

## Masking Strategy

Fine-tuning is robust to masking variations. The feature-based approach (NER) is more sensitive, with the default 80/10/10 strategy performing best.

## Feature-based Approach (CoNLL-2003 NER)

The feature-based approach (concatenating the last 4 layers) is only 0.3 F1 behind fine-tuning on NER, demonstrating BERT produces effective general-purpose features usable without fine-tuning the entire model.

---

# Differences from Related Methods

| Aspect | ELMo | OpenAI GPT | BERT |
|--------|------|-----------|------|
| Architecture | biLSTM (2 layers) | Transformer decoder (12 layers) | Transformer encoder (12/24 layers) |
| Directionality | Concat(LTR, RTL) — shallow | Left-to-right only | Bidirectional (all layers) |
| Pre-training objective | LM (LTR + RTL separately) | LM (LTR) | MLM + NSP |
| Pre-training data | 800M words (BooksCorpus) | 800M words (BooksCorpus) | 3,300M words (BooksCorpus + Wikipedia) |
| Tokens per batch | — | 32K | 128K |
| Special tokens | — | `[CLS]`, `[SEP]`, `[DELIM]` | `[CLS]`, `[SEP]` |
| Sentence separator | — | Delimiter token | `[SEP]` + segment embeddings |
| Fine-tuning | Feature-based (frozen) | All parameters | All parameters |
| Task-specific LR | N/A | Same LR for all | Task-specific LR |

> [!CAUTION]
> $BERT_{\text{BASE}}$ (110M params) and GPT (117M params) are similar in size. The performance gap is primarily attributable to bidirectionality and the larger pre-training corpus, not model capacity.

---

# Code and Models

Pre-trained models and fine-tuning code are publicly available at: https://github.com/google-research/bert
