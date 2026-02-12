# Meta Information

- URL: [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation](https://arxiv.org/abs/1607.05368)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lau, J. H., & Baldwin, T. (2016). An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation. *Proceedings of the 1st Workshop on Representation Learning for NLP*.

# Overview

doc2vec (also called Paragraph Vector) is a neural method for learning fixed-length dense vector representations of variable-length text documents, introduced by Le & Mikolov (2014). This paper answers four practical questions about how to use doc2vec effectively: which variant is better, which hyperparameters matter most, whether training on large external corpora is beneficial, and whether pre-trained word embeddings improve performance.

**Who should use this**: NLP practitioners who need document-level embeddings for tasks such as duplicate question detection, semantic similarity, and document clustering — particularly over long documents where bag-of-words averaging degrades.

# doc2vec Variants

doc2vec has two architectures, both of which learn a document embedding $\mathbf{d} \in \mathbb{R}^m$ alongside word embeddings $\mathbf{w} \in \mathbb{R}^m$:

## DBOW (Distributed Bag of Words)

- **Input**: Document token $d$ (a special ID for the document).
- **Task**: Predict randomly sampled words $w$ within the document from $d$ alone.
- **Output**: A softmax (or negative-sampling approximation) over the vocabulary.
- **Training objective** (negative sampling):

$$\mathcal{L} = \sum_{(d, w)} \left[\log \sigma(\mathbf{d}^\top \mathbf{w}) + \sum_{w' \sim P_n} \log \sigma(-\mathbf{d}^\top \mathbf{w}')\right]$$

where $P_n$ is the noise distribution, $\sigma$ is the sigmoid function, $\mathbf{d} \in \mathbb{R}^m$ is the document vector, and $\mathbf{w}, \mathbf{w}' \in \mathbb{R}^m$ are the positive and negative word vectors.

DBOW ignores word order and is conceptually similar to the skip-gram model for words. It is simpler and faster to train.

## DMPV (Distributed Memory with Paragraph Vectors)

- **Input**: Document token $d$ concatenated with a sliding window of context words $w_{t-k}, \ldots, w_{t-1}$ (window size $k$).
- **Task**: Predict the next word $w_t$ given both $d$ and the context words.
- **Output**: Softmax over vocabulary.
- **Key difference from DBOW**: DMPV incorporates local word-order information via the context window, similar to the CBOW model for words.

DMPV requires roughly 10× more training epochs than DBOW to converge due to the larger number of parameters.

> [!NOTE]
> The authors compare DBOW and DMPV across all experiments and consistently find that DBOW outperforms DMPV despite being the simpler model. DMPV's additional complexity does not pay off in the tasks studied.

# Evaluation Tasks

## Task 1: Forum Question Duplication (Q-Dup)

- **Dataset**: 12 StackExchange subforums (e.g., Android, Gaming, WordPress, Cooking).
- **Task**: Given a question, rank candidate duplicate questions by similarity.
- **Metric**: ROC AUC (Area Under the Curve) — higher is better.
- **Average document length**: ~130 words (longer documents).
- **Similarity computation**: cosine similarity between document embeddings.

## Task 2: Semantic Textual Similarity (STS)

- **Dataset**: SemEval-2015 English subtask (5 domains: Headlines, Images, Forums, Students, Belief).
- **Task**: Predict a continuous similarity score (0–5) for sentence pairs.
- **Metric**: Pearson's $r$ between predicted and gold-standard scores.
- **Average document length**: ~13 words (shorter documents).
- **Similarity computation**: cosine similarity between document embeddings.

# Hyperparameter Analysis

The authors train doc2vec on task-specific corpora and systematically vary hyperparameters. Key findings:

| Hyperparameter | DBOW (optimal) | DMPV (optimal) |
|---|---|---|
| Vector size $m$ | 300 | 300 |
| Window size $k$ | 15 | 5 |
| Min count | 1–5 | 1–5 |
| Subsampling $t$ | $10^{-5}$ | $10^{-5}$ |
| Negative samples | 5 | 5 |
| Epochs | 20–400 | 600–1000 |

- **Window size**: DBOW favors large windows (15) because it samples any word in the document regardless of position; DMPV uses smaller windows (5) because it must fit in memory and compute over local contexts.
- **Subsampling** of high-frequency words is critical: setting $t = 10^{-5}$ consistently outperforms larger values.
- **Epochs**: DMPV requires far more training due to slower convergence (more parameters per update step).

# Training with Large External Corpora

Instead of training only on task-specific documents, the authors train doc2vec on two large external corpora and then directly use the resulting embeddings at inference time:

| Corpus | Documents | Tokens |
|---|---|---|
| Wikipedia | ~35 million | ~2 billion |
| AP News | ~25 million | ~0.9 billion |

At test time, new documents are encoded by running additional gradient steps on the document vector while keeping all word vectors fixed (inference mode).

**Key finding**: Training on large corpora makes doc2vec robust, especially for Q-Dup (long documents). For STS (short documents), word2vec averaging over pre-trained embeddings is competitive or superior.

> [!IMPORTANT]
> DBOW trained on Wikipedia ("DBOW (wiki)") achieves near-best results on Q-Dup without any task-specific training, demonstrating strong off-the-shelf applicability for long-document similarity.

# Improving doc2vec with Pre-trained Word Embeddings

A critical practical finding: initializing DBOW's word vectors with pre-trained skip-gram embeddings (trained on the same large corpus) dramatically stabilizes training and improves performance.

**Procedure**:
1. Train skip-gram on the external corpus to get word embeddings $\mathbf{W} \in \mathbb{R}^{|V| \times m}$.
2. Initialize DBOW's word embedding matrix with $\mathbf{W}$ and freeze it (or fine-tune at a low learning rate).
3. Train document vectors $\{\mathbf{d}_i\}$ as usual via negative sampling.

**Why it helps**: Without pre-trained initialization, word vectors trained alongside document vectors can be noisy due to corpus size and task sparsity. Pre-trained embeddings provide a stable semantic anchor.

Typical improvement: ~0.01–0.02 AUC / Pearson $r$ across tasks, with more consistent results across random seeds.

# Comparison with Other Methods

| Method | Approach | Q-Dup (long docs) | STS (short docs) |
|---|---|---|---|
| DBOW | doc2vec, bag-of-words at doc level | **Best overall** | Competitive |
| DMPV | doc2vec, context-window at doc level | Below DBOW | Below DBOW |
| word2vec average | Average word vectors | Moderate | Strong |
| skip-thought | RNN encoder trained on sentence continuity | Weak | Moderate |
| Paragram-phrase (PP) | Word-vector averaging with paraphrase supervision | Weak on Q-Dup | Strong on STS |

> [!NOTE]
> "Vector averaging methods work best for shorter documents, while DBOW handles longer documents better." This is because DBOW's document vector is trained to predict any word in the full document, making it capture global semantics. Word averaging loses this by treating all words equally with no document-level context.

**Difference from skip-thought**: skip-thought vectors are trained by predicting adjacent sentences (sequence-to-sequence), which imposes a strong sequential structure. This makes them poorly calibrated for unordered similarity tasks. doc2vec's DBOW avoids sequential assumptions entirely.

**Difference from bag-of-words (ngrams TF-IDF)**: Classic TF-IDF vectors are sparse and high-dimensional ($\mathbb{R}^{|V|}$), do not encode semantic similarity between words, and cannot generalize to unseen documents without retraining. DBOW produces dense, fixed-size vectors ($\mathbb{R}^m$, $m=300$) with semantic smoothness.

# Qualitative Analysis

t-SNE visualization of doc2vec embeddings shows that document vectors cluster by content words (e.g., "tech", "bangalore") rather than function words ("the", "of"), indicating that the embedding space captures topical semantics rather than syntactic patterns. This contrasts with word2vec centroids, which can be pulled toward high-frequency function words if subsampling is not applied.

# Experiments

- **Datasets**:
  - Q-Dup: 12 StackExchange subforums (Android, Gaming, Webmasters, Cooking, English, GameDev, Photos, Stats, Mathematica, Unix, WordPress, Physics)
  - STS: SemEval-2015 English STS subtask (Headlines, Images, Forums, Students, Belief)
  - External training: Wikipedia (~2B tokens), AP News (~0.9B tokens)
- **Hardware**: Not explicitly stated.
- **Optimizer**: Stochastic gradient descent with negative sampling (standard word2vec/doc2vec training).
- **Implementation**: gensim doc2vec.
- **Results**:
  - DBOW consistently outperforms DMPV across both tasks.
  - doc2vec with large external corpus training matches or exceeds skip-thought on Q-Dup by a large margin (e.g., .96 vs .57 AUC on Android subforum).
  - Pre-trained word vector initialization improves DBOW results on Q-Dup (e.g., Android: .97 → .99).
  - On STS (short documents), word2vec averaging with pre-trained embeddings remains a strong baseline.
