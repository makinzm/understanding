# Meta Information

- URL: [Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling](https://arxiv.org/abs/2409.15626)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kapoor, S., Gil, A., Bhaduri, S., Mittal, A., & Mulkar, R. (2024). Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling. arXiv:2409.15626.

# Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling

## Overview

QualIT is a topic modeling pipeline that combines LLM-based key-phrase extraction with K-Means clustering and automatic cluster count selection. It is designed for Voice of Customer (VOC) analysis in organizational research contexts—specifically for HR teams, talent management researchers, and data scientists who need to analyze large corpora of open-ended survey responses or employee feedback. The pipeline addresses key weaknesses of classical methods (LDA's bag-of-words assumption and fixed topic count) and neural methods (BERTopic's single-topic-per-document constraint).

On the 20 NewsGroups dataset with 20 ground-truth categories, QualIT achieves 70% topic coherence and 95.5% topic diversity, outperforming LDA (65% / 72%) and BERTopic (57% / 85%).

## Background and Motivation

### Traditional Topic Modeling Limitations

**Latent Dirichlet Allocation (LDA)** models documents as mixtures of topics and topics as distributions over a fixed vocabulary. Its limitations:
- Bag-of-words assumption: ignores word order and semantic context.
- The number of topics $K$ must be pre-specified by the researcher.
- Struggles with short texts and polysemous words.

**BERTopic** improves semantic representation using BERT embeddings and UMAP/HDBSCAN clustering, but still imposes the constraint that each document is assigned only one topic, causing information loss when documents cover multiple themes.

### Applicability Context

QualIT is most useful when:
- Documents may cover multiple simultaneous topics (e.g., employee engagement surveys).
- Ground-truth topic count is unknown and must be inferred automatically.
- Runtime of 2–3 hours per 2500-document corpus is acceptable (compared to 30 minutes for BERTopic).
- High topic diversity is critical to avoid redundant or overlapping topics.

## Methodology

### Pipeline Overview

Input: a corpus of $N$ text documents $\{d_1, d_2, \ldots, d_N\}$.

Output: a set of labeled main topics $\{T_1, T_2, \ldots, T_K\}$, each with sub-topics $\{S_{k,1}, \ldots, S_{k,m}\}$, and document-to-topic assignments (many-to-many).

### Step 1: Key-Phrase Extraction via LLM

Each document $d_i$ is passed to a large language model (Claude 2.1 in the paper) with a prompt instructing it to extract key phrases that capture recurring themes and patterns. Unlike BERTopic, the LLM can output multiple key phrases per document, enabling multi-topic assignment.

**Pseudocode:**
```
for each document d_i in corpus:
    keyphrases_i = LLM.extract_keyphrases(d_i)
    keyphrase_pool.extend(keyphrases_i)
```

### Step 2: Hallucination Check via Cosine Similarity

Because LLMs may generate key phrases not grounded in the source document, QualIT filters hallucinated phrases using cosine similarity between the document embedding and each key-phrase embedding.

**Coherence score for document $i$:**

$$C_i = \frac{1}{n} \sum_{j=1}^{n} \frac{\mathbf{v}_{\text{input},ij} \cdot \mathbf{v}_{\text{keyphrase},ij}}{\|\mathbf{v}_{\text{input},ij}\| \cdot \|\mathbf{v}_{\text{keyphrase},ij}\|}$$

where $\mathbf{v}_{\text{input},ij} \in \mathbb{R}^d$ is the embedding of document $i$, $\mathbf{v}_{\text{keyphrase},ij} \in \mathbb{R}^d$ is the embedding of the $j$-th key phrase from document $i$, and $n$ is the number of key phrases extracted from document $i$.

Key phrases with $C_i < 0.10$ are discarded as likely hallucinations.

### Step 3: Two-Level K-Means Clustering

All retained key phrases are embedded and clustered at two levels.

**Automatic $K$ selection via Silhouette Score:**

$$s(i) = \begin{cases} \frac{b(i) - a(i)}{\max(a(i),\, b(i))} & \text{if } |C_I| > 1 \\ 0 & \text{if } |C_I| = 1 \end{cases}$$

where $a(i)$ is the mean intra-cluster distance for key phrase $i$, and $b(i)$ is the mean distance from $i$ to the nearest other cluster. The value of $K$ that maximizes the mean silhouette score across all key phrases is selected automatically—no manual specification required.

**Main topic clustering pseudocode:**
```
K_range = [2, 3, ..., K_max]
best_K, best_score = 0, -inf
for K in K_range:
    labels = KMeans(n_clusters=K).fit(keyphrase_embeddings)
    score = mean_silhouette_score(keyphrase_embeddings, labels)
    if score > best_score:
        best_K, best_score = K, score

main_clusters = KMeans(n_clusters=best_K).fit(keyphrase_embeddings)
```

**Sub-topic clustering** repeats this process within each main cluster to produce finer-grained groupings.

### Step 4: LLM Topic Labeling

For each cluster, the key phrases are passed to the LLM, which generates a human-readable topic label summarizing the cluster's theme.

```
for each cluster C_k:
    topic_label_k = LLM.summarize_to_topic(keyphrases in C_k)
```

## Comparison with Similar Algorithms

| Feature | LDA | BERTopic | QualIT |
|---|---|---|---|
| Word representation | Bag-of-words | BERT embeddings | LLM key-phrase extraction |
| Topics per document | Multiple (mixture) | Single | Multiple (multi-label) |
| Topic count $K$ | Manual | Automatic (HDBSCAN) | Automatic (Silhouette + K-Means) |
| Semantic coherence | Low | Medium | High |
| Topic diversity | Low | Medium | High |
| Runtime (2500 docs) | Fast | ~30 min | ~2–3 hours |
| Hierarchical topics | No | No | Yes (main + sub-topics) |

> [!IMPORTANT]
> The key differentiator of QualIT over BERTopic is multi-topic-per-document assignment. BERTopic embeds each document and assigns it to exactly one cluster; QualIT extracts multiple key phrases per document and can assign the document to multiple topic clusters simultaneously.

## Evaluation Metrics

### Topic Coherence (Normalized Pointwise Mutual Information, NPMI)

$$\text{NPMI}(x, y) = \frac{\ln\!\left(\frac{p(x,y)}{p(x)\,p(y)}\right)}{-\ln\,p(x,y)}$$

Ranges from $-1$ to $+1$. Higher values indicate that the top-$n$ most frequent words within a topic co-occur more than expected by chance, reflecting greater semantic relatedness.

### Topic Diversity

$$\text{Diversity} = \frac{|\text{unique words across all topics}|}{|\text{total words across all topics}|}$$

Ranges from 0 to 1. A value close to 1 means topics are distinct and non-overlapping; a value close to 0 indicates highly redundant topics.

# Experiments

- **Dataset**: 20 NewsGroups — 20,000 newsgroup articles across 20 categories (e.g., `sci.med`, `rec.sport.hockey`, `talk.politics.guns`). Preprocessing: lowercasing, special-character removal, stopword removal, short-token removal, lemmatization.
- **Hardware**: AWS SageMaker and AWS Bedrock.
- **Model**: Claude 2.1 (temperature=0, top_k=50, top_p=0) for key-phrase extraction and topic labeling.
- **Baselines**: LDA (Gensim implementation) and BERTopic.
- **Human evaluation**: 4 annotators independently mapped generated topics to ground-truth categories; consensus computed at ≥2, ≥3, and all-4 agreement levels.

**Key quantitative results (at 20 topics, matching ground truth):**

| Method | Avg Coherence | Avg Diversity |
|---|---|---|
| LDA | 51.4% | 72.7% |
| BERTopic | 61.0% | 86.3% |
| QualIT | **64.4%** | **93.7%** |

At the ground-truth $K=20$, QualIT reaches 70% coherence and 95.5% diversity.

**Human evaluation (% of generated topics correctly matched to ground-truth):**

| Consensus Level | LDA | BERTopic | QualIT |
|---|---|---|---|
| ≥2 of 4 evaluators | 50% | 45% | **80%** |
| ≥3 of 4 evaluators | 25% | 25% | **50%** |
| All 4 evaluators | 20% | 20% | **35%** |

> [!NOTE]
> Human evaluators found QualIT topics "less ambiguous" and "easier to classify into topics vs the benchmarks," suggesting that LLM-generated labels are more interpretable than LDA's top-word lists or BERTopic's c-TF-IDF labels.

## Limitations

- **Runtime**: 2–3 hours for 2,500 documents vs. ~30 minutes for BERTopic, due to repeated LLM API calls. Not suitable for real-time or large-scale batch processing without optimization.
- **Clustering algorithm**: K-Means assumes spherical clusters; replacing it with HDBSCAN could handle arbitrary cluster shapes and eliminate even the need to search over $K$.
- **Language**: Evaluated only on English text; LLM extraction quality for low-resource languages is untested.
- **LLM dependency**: Quality depends on the underlying LLM. Claude 2.1 is used; performance with smaller or open-source models may degrade.
