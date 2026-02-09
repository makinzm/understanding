# Meta Information

- URL: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140), 1-67.

# Introduction

This paper introduces T5 (Text-to-Text Transfer Transformer), a unified framework that converts every NLP task into a text-to-text format where the model receives text input and generates text output. The primary contribution is a comprehensive empirical study systematically comparing transfer learning techniques, model architectures, unsupervised pre-training objectives, datasets, and other factors to determine the state-of-the-art approaches for NLP as of 2019.

## Applicability

This framework applies to:
- **Who**: NLP researchers and practitioners building general-purpose language models
- **When**: Tasks requiring transfer learning from large-scale pre-training, especially when labeled data is limited
- **Where**: Classification, question answering, summarization, translation, and other text generation tasks

> [!NOTE]
> The authors state: "Every task we consider—including translation, question answering, and classification—is cast as feeding our model text as input and training it to generate some target text."

# Text-to-Text Framework

## Unified Format

The T5 framework converts all NLP tasks into a consistent format:

**Input**: Task-specific prefix + input text
**Output**: Target text

### Examples by Task Type

| Task | Input Format | Output Format |
|------|--------------|---------------|
| Translation (English→German) | `translate English to German: That is good.` | `Das ist gut.` |
| Classification (CoLA) | `cola sentence: The course is jumping well.` | `acceptable` or `not acceptable` |
| Summarization | `summarize: <article text>` | `<summary text>` |
| Question Answering (SQuAD) | `question: <question> context: <context>` | `<answer text>` |

This approach allows a single model to handle diverse tasks without architectural modifications.

## Model Architecture

### Baseline: Encoder-Decoder Transformer

T5 uses a standard Transformer encoder-decoder architecture similar to the original Vaswani et al. (2017) model with the following specifications:

**Architecture Details**:
- **Encoder**: 12 layers, each with self-attention and feed-forward network
- **Decoder**: 12 layers, each with self-attention, encoder-decoder attention, and feed-forward network
- **Embedding dimension**: $d_{model} = 768$
- **Feed-forward hidden size**: $d_{ff} = 3072$
- **Attention heads**: 12 heads with dimension $d_{kv} = 64$ per head
- **Total parameters**: ~220 million (baseline)

**Attention Mechanism**:

For self-attention in encoder and decoder, and cross-attention from decoder to encoder:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{kv}}}\right)V$$

where $Q \in \mathbb{R}^{n \times d_{kv}}$, $K \in \mathbb{R}^{m \times d_{kv}}$, $V \in \mathbb{R}^{m \times d_{kv}}$ for sequence lengths $n$ (query) and $m$ (key/value).

**Positional Embeddings**:

T5 uses relative position embeddings rather than absolute positions. For each attention layer, a learned scalar bias is added based on the offset between query position $i$ and key position $j$:

$$b_{i,j} = \text{embedding}(j - i)$$

Position embeddings are shared across all layers but each attention head learns separate embeddings.

### Architecture Variants Compared

The paper compares three architectural approaches:

1. **Encoder-Decoder** (baseline): Full encoder-decoder Transformer
2. **Language Model (decoder-only)**: Causal masking where each token attends only to previous tokens
3. **Prefix LM (decoder-only)**: Full attention over input prefix, causal attention over targets

**Input/Output for each variant**:

- **Encoder-Decoder**:
  - Encoder input: $x \in \mathbb{R}^{n \times d_{model}}$
  - Decoder input: $y_{<t} \in \mathbb{R}^{t \times d_{model}}$
  - Output: $\hat{y}_t \in \mathbb{R}^{|V|}$ (vocabulary distribution)

- **Language Model**:
  - Input: Concatenated $[x; y] \in \mathbb{R}^{(n+m) \times d_{model}}$
  - Output: $\hat{y}_t \in \mathbb{R}^{|V|}$ with causal masking

- **Prefix LM**:
  - Input: Concatenated $[x; y] \in \mathbb{R}^{(n+m) \times d_{model}}$
  - Attention mask: Full attention on $x$, causal on $y$

### Key Finding on Architecture

Encoder-decoder models significantly outperform language model variants:
- **Encoder-decoder with denoising**: 83.28 GLUE average
- **Language model with denoising**: 74.70 GLUE average
- **Prefix LM with denoising**: 82.62 GLUE average

The encoder-decoder architecture benefits from bidirectional attention in the encoder while maintaining autoregressive generation in the decoder.

# Pre-training Objectives

## C4 Dataset

The paper introduces C4 (Colossal Clean Crawled Corpus), a cleaned version of Common Crawl web text:

**Dataset Statistics**:
- **Size**: ~750 GB of text (~156 billion tokens)
- **Source**: April 2019 Common Crawl snapshot
- **Language**: English only

**Cleaning Pipeline** (applied in order):

1. Retain only lines ending with terminal punctuation marks (., !, ?, ")
2. Discard pages with <5 sentences or <3 words per sentence
3. Remove lines with "lorem ipsum" (placeholder text)
4. Filter out pages containing profanity from the "List of Dirty, Naughty, Obscene, and Otherwise Bad Words"
5. Remove JavaScript code by filtering lines with curly braces `{}`
6. Apply deduplication: remove sentences appearing >1 time in any 3-sentence span
7. Use langdetect library to retain only pages with 99%+ probability of English

> [!IMPORTANT]
> The C4 dataset is publicly available and has become a widely-used pre-training corpus for subsequent language models.

## Denoising Objectives Compared

The paper systematically evaluates different corruption strategies for denoising pre-training:

### 1. BERT-style Objective

**Corruption**: 15% of tokens are randomly selected and either:
- Replaced with `[MASK]` token (90% of corrupted tokens)
- Replaced with random token (10% of corrupted tokens)
- Left unchanged (0% - differs from original BERT)

**Target**: Predict only the corrupted tokens, not entire sequence

**Input**: `Thank you [MASK] me to your party [MASK] week`
**Output**: `for inviting last`

### 2. Replace Corrupted Spans

**Corruption**: Randomly sample span lengths from geometric distribution (mean 3), corrupt 15% of tokens
**Replacement**: Each corrupted span replaced by unique sentinel token (`<X>`, `<Y>`, `<Z>`, etc.)

**Input**: `Thank you <X> me to your party <Y> week`
**Output**: `<X> for inviting <Y> last <Z>`

This is the baseline objective used throughout the paper.

### 3. Drop Corrupted Tokens

**Corruption**: Same as replace spans, but corrupted tokens are removed entirely

**Input**: `Thank you me to your party week`
**Output**: `<X> for inviting <Y> last <Z>`

### 4. MASS-style Objective

**Corruption**: Contiguous span of tokens masked (no randomization)
**Target**: Predict only the masked span

Similar to span replacement but with fixed-length spans and different training procedure.

### Objective Comparison Results

All denoising variants perform similarly:

| Objective | GLUE | SuperGLUE | SQuAD | CNN/DM |
|-----------|------|-----------|-------|--------|
| Replace spans (baseline) | 83.28 | 71.36 | 80.88 | 19.24 |
| BERT-style | 83.23 | 71.13 | 80.77 | 19.17 |
| Drop corrupted tokens | 83.64 | 70.90 | 81.08 | 19.50 |
| MASS-style | 82.97 | 71.00 | 80.69 | 19.13 |

**Key finding**: The choice of corruption strategy has minimal impact on downstream performance, but span-based approaches are more computationally efficient.

### Corruption Rate Analysis

The paper tests corruption rates from 10% to 50%:

- **10%**: Slightly lower performance (82.30 GLUE)
- **15%** (BERT's choice): Optimal (83.28 GLUE)
- **25%**: Similar performance (83.22 GLUE)
- **50%**: Significantly worse (82.15 GLUE)

> [!NOTE]
> BERT's choice of 15% corruption rate is near-optimal, but the objective is robust to this hyperparameter.

### Span Length Analysis

For span corruption, the paper compares:

- **i.i.d. corruption** (span length 1): 83.28 GLUE, slower training
- **Average span length 3**: 83.49 GLUE, 1.5× faster training
- **Average span length 10**: 82.97 GLUE, further speedup

**Optimal choice**: Span length 3 provides best speed-performance tradeoff.

## Dataset Variants

The paper compares C4 with several variants:

| Dataset | Description | GLUE | SuperGLUE |
|---------|-------------|------|-----------|
| C4 (baseline) | Cleaned Common Crawl | 83.28 | 71.36 |
| Unfiltered C4 | No cleaning heuristics | 82.51 | 70.39 |
| RealNews-like | News-filtered subset | 82.35 | 69.71 |
| WebText-like | Reddit-filtered (≥3 upvotes) | 82.89 | 70.55 |

**Key finding**: The cleaning heuristics in C4 provide measurable improvements, but even unfiltered web text works reasonably well.

## Pre-training Dataset Size

The paper tests training on smaller fractions of C4:

- **Full C4** (~750 GB): Best performance
- **C4/64** (~11.7 GB): 1.5 point GLUE drop
- **C4/1024** (~732 MB): 3.5 point GLUE drop

This demonstrates that larger pre-training datasets continue to improve performance, even at hundreds of GB scale.

# Fine-tuning Methods

## Multi-task Learning

Instead of pre-training then fine-tuning, the paper explores training on all tasks simultaneously:

**Input mixing strategies**:
1. **Examples-proportional**: Sample tasks proportionally to dataset size
2. **Equal mixing**: Sample each task with equal probability
3. **Temperature-scaled**: $r_m = (K_m)^{1/T}$ where $K_m$ is dataset $m$ size, $T$ is temperature

**Findings**:
- Multi-task learning underperforms pre-train-then-fine-tune by 1-2 points
- Equal mixing works best among mixing strategies
- Adding unsupervised pre-training data to multi-task improves results

## Fine-tuning Procedure

**Standard approach** (used for all experiments):

1. Initialize from pre-trained checkpoint
2. Train on downstream task for $2^{18} = 262,144$ steps
3. Use constant learning rate of 0.001 (no warmup or decay)
4. Batch size: 128 sequences
5. Save checkpoints every 5,000 steps
6. Select best checkpoint based on validation performance
7. Use greedy decoding (no beam search) for generation

**Early stopping**: The paper finds that fine-tuning for fewer steps (e.g., $2^{16}$) hurts performance, while longer training ($2^{19}$) provides minimal gains.

# Scaling Analysis

## Model Sizes

The paper trains models ranging from 60M to 11B parameters:

| Model Size | Layers | $d_{model}$ | $d_{ff}$ | Heads | $d_{kv}$ | Parameters |
|------------|--------|-------------|----------|-------|----------|------------|
| Small | 6 | 512 | 2048 | 8 | 64 | 60M |
| Base | 12 | 768 | 3072 | 12 | 64 | 220M |
| Large | 24 | 1024 | 4096 | 16 | 64 | 770M |
| 3B | 24 | 1024 | 16384 | 32 | 128 | 3B |
| 11B | 24 | 1024 | 65536 | 128 | 128 | 11B |

**Scaling strategy**: Increase model depth and width while keeping parameters balanced between encoder and decoder.

## Computational Budget Analysis

The paper explores different ways to allocate fixed computational budget:

**Trade-offs**:
1. **Larger model + less training**:
   - Train 2× larger model for 0.5× steps
   - Generally underperforms baseline

2. **Ensemble of models**:
   - Train 4 smaller models independently
   - Average predictions at inference
   - Provides 0.5-1 point improvement over single large model

3. **Multi-task training**:
   - Share computation across tasks during pre-training
   - Slight underperformance vs. single-task fine-tuning

**Optimal strategy**: Train single large model for full duration with task-specific fine-tuning.

## Scaling Results

Training with combined best practices (encoder-decoder, span corruption on C4, 15% corruption rate):

| Model | GLUE | SuperGLUE | SQuAD | CNN/DM | WMT En-De | WMT En-Fr | WMT En-Ro |
|-------|------|-----------|-------|--------|-----------|-----------|-----------|
| Base | 83.7 | 73.9 | 83.4 | 19.4 | 26.9 | 38.1 | 26.2 |
| Large | 86.4 | 81.3 | 87.2 | 20.3 | 27.5 | 39.1 | 27.0 |
| 3B | 88.1 | 84.3 | 88.6 | 21.2 | 28.1 | 40.0 | 27.9 |
| 11B | **88.9** | **88.9** | **90.1** | **21.6** | **28.6** | **41.4** | **28.9** |

> [!IMPORTANT]
> The 11B T5 model achieves state-of-the-art results on multiple benchmarks as of 2019, demonstrating that scaling model size with the unified text-to-text framework produces consistent improvements.

# Knowledge Distillation

The paper explores distilling the 11B T5 model into smaller student models:

**Method**:
1. Generate soft labels from 11B teacher on pre-training data
2. Train student model to match teacher's output distribution
3. Use KL divergence loss: $\mathcal{L} = \text{KL}(P_{\text{teacher}} || P_{\text{student}})$

**Results**:
- **Base model distilled**: 87.2 GLUE (vs. 83.7 from scratch, vs. 88.9 teacher)
- **Small model distilled**: 84.8 GLUE (vs. 79.1 from scratch)

Distillation recovers 78% of the teacher's improvement over standard training, providing a practical way to deploy smaller models with near-large-model performance.

# Training Details

## Pre-training Configuration

- **Corpus**: C4 (~750 GB, 156B tokens)
- **Training steps**: $2^{35} \approx 34$ billion tokens (1 epoch over C4)
- **Batch size**: 128 sequences × 512 tokens = 65,536 tokens per batch
- **Sequence length**: 512 tokens (input + target combined)
- **Optimizer**: AdaFactor (Adam variant with less memory)
- **Learning rate schedule**: Inverse square root with 10,000 warmup steps
- **Dropout**: 0.1 on residual connections, layer outputs, attention weights

**Inverse square root schedule**:

$$\text{lr}(t) = \begin{cases}
\frac{t}{10000} \cdot \text{lr}_{\text{base}} & \text{if } t < 10000 \\
\frac{1}{\sqrt{t}} \cdot \text{lr}_{\text{base}} & \text{otherwise}
\end{cases}$$

## Fine-tuning Configuration

- **Training steps**: $2^{18} = 262,144$ steps
- **Batch size**: 128 sequences
- **Learning rate**: 0.001 (constant, no schedule)
- **Checkpoint frequency**: Every 5,000 steps
- **Model selection**: Best validation performance
- **Maximum sequence length**: Task-dependent (e.g., 512 for GLUE, 1024 for summarization)

## Inference

- **Decoding**: Greedy search (argmax at each step)
- **No beam search**: Greedy decoding is competitive and much faster
- **Temperature**: 1.0 (no temperature scaling)

## Vocabulary

- **Size**: 32,000 tokens
- **Method**: SentencePiece with unigram language model
- **Training data**: Mixture of English, German, French, and Romanian C4 for multilingual support
- **Special tokens**: 100 sentinel tokens (`<extra_id_0>` through `<extra_id_99>`) for span corruption

# Benchmark Results

## GLUE (General Language Understanding Evaluation)

The GLUE benchmark consists of 9 sentence classification and sentence-pair tasks:

| Task | Metric | Base | Large | 3B | 11B |
|------|--------|------|-------|-----|-----|
| CoLA | Matthews corr. | 61.4 | 68.8 | 70.5 | 71.2 |
| SST-2 | Accuracy | 94.3 | 96.1 | 96.6 | 97.1 |
| MRPC | F1 | 90.1 | 91.9 | 92.4 | 93.2 |
| STS-B | Pearson corr. | 89.2 | 91.4 | 92.1 | 92.5 |
| QQP | F1 | 88.6 | 90.3 | 90.8 | 91.2 |
| MNLI | Accuracy | 86.2 | 89.7 | 90.6 | 91.1 |
| QNLI | Accuracy | 93.1 | 95.3 | 95.9 | 96.2 |
| RTE | Accuracy | 80.1 | 87.6 | 91.3 | 92.2 |
| WNLI | Accuracy | 89.0 | 94.4 | 95.8 | 96.5 |
| **Average** | - | **83.7** | **86.4** | **88.1** | **88.9** |

## SuperGLUE

A more challenging benchmark with 8 diverse tasks:

| Task | Metric | Base | Large | 3B | 11B |
|------|--------|------|-------|-----|-----|
| BoolQ | Accuracy | 79.7 | 85.2 | 87.1 | 89.1 |
| CB | F1/Accuracy | 90.7 | 96.8 | 97.1 | 98.3 |
| COPA | Accuracy | 71.0 | 83.0 | 90.0 | 94.8 |
| MultiRC | F1a | 70.1 | 78.4 | 82.6 | 86.4 |
| ReCoRD | F1/EM | 86.2 | 90.1 | 92.5 | 94.1 |
| RTE | Accuracy | 80.1 | 87.6 | 91.3 | 92.2 |
| WiC | Accuracy | 70.3 | 75.1 | 77.8 | 79.5 |
| WSC | Accuracy | 83.6 | 89.9 | 93.4 | 95.3 |
| **Average** | - | **73.9** | **81.3** | **84.3** | **88.9** |

> [!NOTE]
> The 11B T5 model achieves 88.9 on SuperGLUE, surpassing human baseline (89.8) on some individual tasks but not the overall average.

## SQuAD (Question Answering)

Stanford Question Answering Dataset with extractive answers:

| Model | Exact Match | F1 Score |
|-------|-------------|----------|
| Base | 83.4 | 90.6 |
| Large | 87.2 | 93.3 |
| 3B | 88.6 | 94.5 |
| 11B | **90.1** | **95.3** |

## CNN/Daily Mail (Summarization)

Abstractive summarization task:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Base | 42.5 | 19.4 | 39.6 |
| Large | 43.2 | 20.3 | 40.1 |
| 3B | 43.8 | 21.2 | 40.7 |
| 11B | **44.2** | **21.6** | **41.2** |

## WMT Translation

Machine translation benchmarks:

| Model | En→De (BLEU) | En→Fr (BLEU) | En→Ro (BLEU) |
|-------|--------------|--------------|--------------|
| Base | 26.9 | 38.1 | 26.2 |
| Large | 27.5 | 39.1 | 27.0 |
| 3B | 28.1 | 40.0 | 27.9 |
| 11B | **28.6** | **41.4** | **28.9** |

# Comparison with Similar Models

## Differences from BERT

| Aspect | BERT | T5 |
|--------|------|-----|
| **Architecture** | Encoder-only | Encoder-decoder |
| **Pre-training objective** | Masked LM + NSP | Span corruption |
| **Task format** | Task-specific heads | Unified text-to-text |
| **Fine-tuning** | Add classification layer | Generate target text |
| **Applications** | Classification, tagging | Generation, classification, QA, translation |

**Key advantage of T5**: The text-to-text framework allows a single model to handle both understanding and generation tasks without architectural modifications.

## Differences from GPT-2

| Aspect | GPT-2 | T5 |
|--------|-------|-----|
| **Architecture** | Decoder-only (language model) | Encoder-decoder |
| **Pre-training** | Causal language modeling | Span corruption denoising |
| **Task format** | Prompt engineering (zero-shot) | Task prefix + fine-tuning |
| **Attention** | Causal (left-to-right only) | Bidirectional encoder + causal decoder |
| **Training data** | WebText (~40 GB) | C4 (~750 GB) |

**Key advantage of T5**: Bidirectional encoder attention improves understanding tasks, while decoder maintains autoregressive generation capability.

## Differences from BART

| Aspect | BART | T5 |
|--------|------|-----|
| **Architecture** | Encoder-decoder | Encoder-decoder (similar) |
| **Pre-training** | Multiple corruption strategies | Span corruption (simpler) |
| **Emphasis** | Denoising autoencoder flexibility | Systematic empirical study |
| **Dataset** | Custom (books, news, etc.) | C4 (public, reproducible) |

**Key difference**: T5 focuses on comprehensive empirical comparison of techniques, while BART emphasizes the denoising autoencoder framework. T5 demonstrates that simpler corruption strategies work as well as BART's more complex approach.

# Limitations and Future Work

## Identified Limitations

1. **Computational cost**: Training 11B models requires substantial resources (weeks on TPU pods)
2. **English-only focus**: While vocabulary supports multiple languages, primary experiments are English-centric
3. **Dataset bias**: C4 inherits biases from web text despite cleaning heuristics
4. **Generation quality**: Greedy decoding is fast but may not produce optimal outputs for generation tasks

## Suggested Directions

1. **Multilingual scaling**: Apply text-to-text framework to diverse languages
2. **Efficient architectures**: Explore sparse attention, mixture-of-experts for reduced computation
3. **Better decoding**: Investigate beam search, sampling strategies for generation quality
4. **Task transfer**: Study zero-shot and few-shot capabilities without fine-tuning
5. **Longer contexts**: Extend to handle documents beyond 512-1024 tokens

# Experiments

- **Dataset**:
  - Pre-training: C4 (Colossal Clean Crawled Corpus, ~750 GB)
  - Fine-tuning: GLUE, SuperGLUE, SQuAD, CNN/Daily Mail, WMT En-De/Fr/Ro
- **Hardware**: TPUv3 pods (not specified in detail)
- **Optimizer**: AdaFactor with inverse square root learning rate schedule
- **Pre-training**: $2^{35}$ ≈ 34 billion tokens, batch size 128 sequences × 512 tokens
- **Fine-tuning**: $2^{18}$ = 262,144 steps, constant learning rate 0.001
- **Model sizes**: 60M to 11B parameters
- **Vocabulary**: 32,000 SentencePiece tokens
- **Results**:
  - GLUE: 88.9 (11B model)
  - SuperGLUE: 88.9 (11B model, approaching human performance)
  - SQuAD: 90.1 EM, 95.3 F1 (11B model)
  - CNN/DM: 21.6 ROUGE-2 (11B model)
  - WMT En-De: 28.6 BLEU (11B model)

> [!TIP]
> Code and pre-trained models are open-sourced at: https://github.com/google-research/text-to-text-transfer-transformer

# Conclusion

T5 demonstrates that a unified text-to-text framework can effectively handle diverse NLP tasks by converting all problems into the same format. The comprehensive empirical study reveals that:

1. **Architecture**: Encoder-decoder models with denoising objectives outperform language model approaches
2. **Pre-training objective**: Various denoising strategies perform similarly; span corruption is efficient
3. **Dataset**: Larger, cleaned web corpora (C4) improve results
4. **Scaling**: Increasing model size from 220M to 11B parameters provides consistent gains across all tasks
5. **Simplicity**: The framework requires no task-specific architectural modifications

The paper's systematic exploration establishes best practices for transfer learning in NLP and provides the community with reproducible baselines through open-source code, models, and the C4 dataset. T5's success demonstrates that scale combined with a unified framework can achieve state-of-the-art results across diverse language understanding and generation tasks.
