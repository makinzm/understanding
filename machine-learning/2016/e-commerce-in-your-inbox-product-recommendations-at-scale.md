# Meta Information

- URL: [E-commerce in Your Inbox: Product Recommendations at Scale](https://arxiv.org/abs/1606.07154)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Grbovic, M., Radosavljevic, V., Djuric, N., Bhamidipati, N., Savla, J., Bhagwan, V., & Sharp, D. (2015). E-commerce in your inbox: Product recommendations at scale. In *KDD '15: Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

---

# E-commerce in Your Inbox: Product Recommendations at Scale

## Overview

This paper presents a large-scale product recommendation system deployed in **Yahoo Mail** that exploits e-commerce receipt emails to deliver personalized product advertisements. The core contribution is the adaptation of Word2Vec's skip-gram model to learn product embeddings (prod2vec) from purchase sequences, enabling product-to-product and user-to-product recommendations at internet scale.

**Applicability:**
- E-commerce recommendation engineers needing scalable, low-latency personalization
- Researchers applying neural language models to sequential behavioral data beyond NLP
- Companies with access to transactional event logs (purchase histories, clickstreams)

---

## Problem Formulation

### Input and Output

- **Input**: A corpus of purchase receipts $\mathcal{C} = \{r_1, r_2, \ldots, r_N\}$ where each receipt $r_i = (p_1, p_2, \ldots, p_k)$ is an ordered or unordered set of product IDs purchased by a user.
- **Output**:
  - Product embeddings $\mathbf{v}_p \in \mathbb{R}^d$ for each product $p$ in vocabulary $\mathcal{V}$ (where $|\mathcal{V}| = 2.1\text{M}$, $d$ is the embedding dimension).
  - A recommendation list of top-$K$ products for a given query product or user.

---

## Model Architectures

### prod2vec

prod2vec directly applies the **skip-gram** model from Word2Vec to product purchase sequences. Each user's full purchase history is treated as a "sentence," with individual products as "words."

**Objective** — maximize log-likelihood of context products given the center product:

$$
\mathcal{L} = \sum_{r \in \mathcal{C}} \sum_{p_i \in r} \sum_{-c \leq j \leq c, j \neq 0} \log P(p_{i+j} \mid p_i)
$$

where $c$ is the context window size. The conditional probability is modeled with softmax:

$$
P(p_o \mid p_i) = \frac{\exp(\mathbf{v}_{p_o}^\top \mathbf{v}_{p_i})}{\sum_{p \in \mathcal{V}} \exp(\mathbf{v}_p^\top \mathbf{v}_{p_i})}
$$

In practice, **negative sampling** is used to approximate this softmax efficiently over the large vocabulary.

**Pseudocode:**
```
Input: Purchase sequences C, context window c, embedding dim d, neg samples k
Initialize: product embeddings V_in, V_out ∈ R^{|V| × d} randomly

for each receipt r in C:
    for each product p_i in r:
        for j in [-c, ..., -1, 1, ..., c]:
            p_context = r[i + j]  # context product
            # Positive update
            gradient = (1 - sigmoid(V_out[p_context] · V_in[p_i])) * V_out[p_context]
            V_in[p_i] += η * gradient
            # Negative sampling updates (k random products)
            for p_neg in sample(V, k):
                gradient = -sigmoid(V_out[p_neg] · V_in[p_i]) * V_out[p_neg]
                V_in[p_i] += η * gradient

Return: V_in as final product embeddings
```

### bagged-prod2vec

A key insight is that a **shopping bag** (a single receipt with multiple items) represents simultaneous purchases with no meaningful temporal order. Treating them as an ordered sequence introduces artificial ordering artifacts.

bagged-prod2vec modifies the training to treat each receipt as an **unordered bag**: for a receipt $r = \{p_1, \ldots, p_k\}$, every product in the bag serves as context for every other product in the bag (window covers all items regardless of position).

> [!NOTE]
> "Unlike prod2vec where each purchase is assigned a specific position within the sequence, in the bagged version each product is taken as context of every other product within the same shopping bag."

This more accurately reflects co-purchase semantics: buying a printer and ink cartridge together signals a strong relationship regardless of which was listed first in the receipt.

### user2vec

To generate **user-level** embeddings for cold-start and long-term preference modeling, user2vec introduces a user vector $\mathbf{u}_{\text{user}} \in \mathbb{R}^d$ that replaces the center-product vector in the skip-gram objective. The model predicts the products a user purchases from their latent user representation.

**Difference from prod2vec:** prod2vec predicts neighboring products given a product; user2vec predicts all purchased products given a user vector.

---

## Recommendation Algorithms

### Product-to-Product Recommendation (Cosine Similarity)

Given a query product $p_q$, recommend the top-$K$ products by cosine similarity in embedding space:

$$
\text{sim}(p_i, p_j) = \frac{\mathbf{v}_{p_i} \cdot \mathbf{v}_{p_j}}{\|\mathbf{v}_{p_i}\| \cdot \|\mathbf{v}_{p_j}\|}
$$

### Cluster-Based Transition (bagged-prod2vec-cluster)

To introduce **diversity** and avoid recommending only close neighbors in embedding space, products are clustered using K-means on their embeddings. A transition matrix $T$ is estimated where $T[c_i][c_j]$ is the empirical probability of a user in cluster $c_i$ next purchasing from cluster $c_j$.

**Recommendation with time decay:**
For a user with purchase history $H = \{(p_1, t_1), (p_2, t_2), \ldots\}$ (product, timestamp), the scored recommendation set is:

$$
\text{score}(p_{\text{rec}}) = \sum_{(p_i, t_i) \in H} \alpha^{\Delta t_i} \cdot \text{sim}(p_i, p_{\text{rec}}) \cdot T[c(p_i)][c(p_{\text{rec}})]
$$

where $\alpha = 0.9$ is the decay factor and $\Delta t_i$ is time since purchase $p_i$ in days. This rewards products that are (a) semantically similar to recent purchases and (b) in clusters frequently visited after current clusters.

---

## System Architecture

The production system at Yahoo Mail is a multi-stage distributed pipeline:

```
Email receipts
    ↓ (parse and normalize)
Purchase event log (Hadoop HDFS)
    ↓ (batch training every 5 days)
prod2vec / bagged-prod2vec model
    ↓ (offline inference)
Product-to-product similarity table  →  Distributed KV Store (product2product)
User purchase history                →  Distributed KV Store (user_profiles)
    ↓ (online serving, <500ms SLA)
Ad serving layer (personalized inbox ads)
```

**Key infrastructure details:**
- Training: Hadoop-based distributed Word2Vec on 5-day purchase windows
- Serving: Custom distributed key-value stores for both user profiles and precomputed product similarities
- Ad placement: "Pencil" position above inbox, displayed during email browsing sessions

---

## Cold Start Handling

Users without sufficient purchase history are served recommendations based on **popular products filtered by demographic cohorts** (combinations of state, age, gender). These cohort-based popular lists are recalculated every 3 days using a 5-day lookback window.

> [!NOTE]
> The 5-day lookback for popular products was empirically determined — shorter windows miss important trends, while longer windows introduce stale items.

---

## Experiments

### Datasets

| Dataset | Scale |
|---|---|
| Yahoo Mail purchase receipts | 280M+ purchases, 29M users, 172 commercial websites |
| Product vocabulary | 2.1M unique products |
| Training period | 5-day rolling window |

### Offline Evaluation

**Task:** Predict future purchases given past purchase history.

**Metrics:** Precision@K, evaluated at prediction horizons of 1–14 days.

**Baselines compared:**
| Method | Description |
|---|---|
| Popular products | Globally most purchased items |
| co-purchase frequency | Items most frequently bought with query product |
| prod2vec | Skip-gram on ordered purchase sequences |
| bagged-prod2vec | Skip-gram on unordered shopping bags |
| bagged-prod2vec-cluster | Cluster-based transition scoring with time decay |
| user2vec | User vector predicting all user purchases |

**Key offline findings:**
- bagged-prod2vec outperforms prod2vec, confirming that unordered bag treatment is more appropriate for receipt data
- bagged-prod2vec-cluster achieves the best precision@K at short horizons (1–3 days) by leveraging cluster diversity
- user2vec performs well at the 1-day horizon but degrades faster than product-based methods as horizon increases

### Online A/B Test

**Setup:** Live bucket test in Yahoo Mail during the 2014 holiday season.

**Results:**
- ~9.8% improvement in click-through rate (CTR) vs. control (non-personalized ads)
- ~7.63% improvement in yield rate vs. control
- The system met the <500ms serving latency SLA

---

## Comparison with Similar Methods

| Method | Training Data | Granularity | Key Difference |
|---|---|---|---|
| **Matrix Factorization (CF)** | User-item ratings | User ↔ Item | Requires explicit ratings; doesn't capture sequential/temporal dynamics |
| **Word2Vec (NLP)** | Text corpora | Word sequences | Adapted here to product purchase sequences |
| **prod2vec** (this paper) | Purchase receipts | Product sequences | Treats purchases as ordered; may introduce artificial ordering |
| **bagged-prod2vec** (this paper) | Purchase receipts | Unordered bags | Respects simultaneous co-purchase structure |
| **bagged-prod2vec-cluster** (this paper) | Purchase receipts | Cluster transitions | Adds diversity through cluster-level transition probabilities |
| **user2vec** (this paper) | Purchase receipts | User-level | Single vector per user; degrades over long prediction horizons |

> [!IMPORTANT]
> The bagged-prod2vec approach works better than standard prod2vec specifically because e-commerce receipts contain **unordered multi-item purchases**. In domains where sequences are naturally ordered (e.g., music listening history, page navigation), standard prod2vec may be more appropriate.

---

## Key Takeaways

1. **Email receipts as implicit feedback**: Purchase confirmation emails provide rich, passively collected behavioral data that sidesteps the cold-start and sparsity issues of explicit rating systems.
2. **Bag-of-products semantics**: Respecting the unordered nature of shopping receipts (bagged-prod2vec) meaningfully improves embedding quality over treating them as ordered sequences.
3. **Cluster-based diversity**: Transition probabilities between product clusters prevent recommendations from collapsing to near-identical items, improving diversity without sacrificing relevance.
4. **Time decay is critical**: Recent purchases are exponentially more predictive of future behavior ($\alpha = 0.9$ decay per day), so scoring should weight recency explicitly.
5. **Production feasibility**: A 5-day Hadoop training cycle with KV-store serving is sufficient for internet-scale recommendation at sub-500ms latency.
