# Meta Information

- URL: [[1603.04259] Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/abs/1603.04259)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Barkan, O., & Koenigstein, N. (2016). Item2Vec: Neural item embedding for collaborative filtering. In *2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP)* (pp. 1–6). IEEE.

# Background: Collaborative Filtering and Word2Vec

## Item-Based Collaborative Filtering

Item-based collaborative filtering (CF) is a recommendation technique that models user preference by measuring similarity between items based on co-occurrence patterns in user history. Given a set of users $\mathcal{U}$ and a set of items $\mathcal{I}$, for each user $u \in \mathcal{U}$ a set of consumed items $\mathcal{S}_u \subseteq \mathcal{I}$ is observed. The goal is to recommend new items to user $u$ by finding items that are similar to those already in $\mathcal{S}_u$.

Traditional approaches represent items as sparse high-dimensional vectors (e.g., columns of a user-item matrix) and compute cosine or Pearson similarity. A core limitation is that these representations do not capture latent semantic relationships: two items that are never directly co-consumed by the same user but are conceptually related may score zero similarity.

## Skip-gram with Negative Sampling (Word2Vec SGNS)

Word2Vec's Skip-gram model, introduced by Mikolov et al. (2013), learns dense vector embeddings for words by predicting the context of a target word. Given a sequence of words $w_1, w_2, \ldots, w_T$, the objective is to maximize:

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c,\ j \neq 0} \log p(w_{t+j} \mid w_t)$$

where $c$ is the context window size and $p(w_{t+j} \mid w_t)$ is approximated using the softmax over all vocabulary words.

Negative Sampling (SGNS) replaces the expensive softmax with a binary classification objective. For each (target, context) word pair $(w, c)$ observed in the corpus, the model maximizes:

$$\log \sigma(\mathbf{v}_{w}^T \mathbf{v}_{c}) + \sum_{k=1}^{K} \mathbb{E}_{c_k \sim P_n(w)}\left[\log \sigma(-\mathbf{v}_{w}^T \mathbf{v}_{c_k})\right]$$

where:
- $\mathbf{v}_{w}, \mathbf{v}_{c} \in \mathbb{R}^d$ are the embedding vectors of target word $w$ and context word $c$
- $\sigma(\cdot)$ is the sigmoid function
- $K$ is the number of negative samples
- $P_n(w) \propto f(w)^{3/4}$ is the noise distribution over vocabulary words (frequency raised to 3/4 power)

> [!TIP]
> [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

# Item2Vec Method

## Core Insight: Items as Words, Sets as Sentences

Item2Vec reframes item-based CF as a word embedding problem. The key observation is:

> [!NOTE]
> "item-based CF can be cast in the same framework of neural word embedding."

The analogy is:
| NLP (Word2Vec) | Item2Vec (Collaborative Filtering) |
|---|---|
| Word | Item |
| Sentence / document | User's consumed item set $\mathcal{S}_u$ |
| Word co-occurrence within window | Item co-occurrence within same set |
| Word vocabulary $\mathcal{V}$ | Item catalog $\mathcal{I}$, $\lvert\mathcal{I}\rvert = V$ |

## Input and Output

- **Input**: A collection of user-item sets $\{\mathcal{S}_u\}_{u \in \mathcal{U}}$, where $\mathcal{S}_u = \{i_1, i_2, \ldots, i_{|\mathcal{S}_u|}\}$ is the set of items consumed by user $u$.
- **Output**: Item embedding matrix $\mathbf{W} \in \mathbb{R}^{V \times d}$, where $V = |\mathcal{I}|$ is the number of items and $d$ is the embedding dimension. Each row $\mathbf{v}_i \in \mathbb{R}^d$ is the embedding vector for item $i$.

## Objective Function

For an item set $\mathcal{S}_u = \{i_1, \ldots, i_K\}$, Item2Vec treats all pairs $(i_j, i_k)$ with $j \neq k$ as positive (target, context) pairs—unlike Word2Vec which uses a sliding window of fixed size $c$. This design choice is appropriate because item sets have no temporal ordering, so no positional bias should be introduced.

The objective to maximize over all users is:

$$\frac{1}{K} \sum_{j=1}^{K} \sum_{\substack{k=1 \\ k \neq j}}^{K} \log p(i_k \mid i_j)$$

Using Negative Sampling, $\log p(i_k \mid i_j)$ is approximated as:

$$\log \sigma(\mathbf{u}_{i_j}^T \mathbf{v}_{i_k}) + \sum_{m=1}^{M} \mathbb{E}_{i_m \sim P_n(i)}\left[\log \sigma(-\mathbf{u}_{i_j}^T \mathbf{v}_{i_m})\right]$$

where:
- $\mathbf{u}_{i} \in \mathbb{R}^d$: "target" embedding of item $i$ (rows of matrix $\mathbf{U} \in \mathbb{R}^{V \times d}$)
- $\mathbf{v}_{i} \in \mathbb{R}^d$: "context" embedding of item $i$ (rows of matrix $\mathbf{W} \in \mathbb{R}^{V \times d}$)
- $M$: number of negative samples per positive pair
- $P_n(i) \propto f(i)^{3/4}$: noise distribution over items, where $f(i)$ is the frequency of item $i$

After training, item $i$ is represented by the sum (or average) $\mathbf{u}_i + \mathbf{v}_i$, or equivalently by only $\mathbf{u}_i$, depending on implementation.

## Algorithm (Training Procedure)

```
Input:  item sets {S_u}, embedding dim d, neg samples M, epochs E, learning rate η
Output: item embedding matrix U ∈ R^(V×d)

1. Initialize U, W ∈ R^(V×d) with small random values
2. For epoch = 1, ..., E:
   3. For each user set S_u in {S_u}:
      4. For each item i_j in S_u:
         5. For each item i_k in S_u, k ≠ j:
            6. Compute positive score: s_pos = σ(u_{i_j}^T v_{i_k})
            7. Sample M negative items i_m ~ P_n(i)
            8. Compute negative scores: s_neg_m = σ(-u_{i_j}^T v_{i_m}) for m=1..M
            9. Compute gradient of SGNS loss w.r.t. u_{i_j}, v_{i_k}, v_{i_m}
           10. Update: u_{i_j} += η * ∂L/∂u_{i_j}
           11. Update: v_{i_k} += η * ∂L/∂v_{i_k}
           12. Update: v_{i_m} += η * ∂L/∂v_{i_m}  (for each negative m)
13. Return U (or U + W averaged)
```

> [!IMPORTANT]
> A critical difference from Word2Vec: since item sets are **unordered** (unlike text sequences), all within-set pairs $(i_j, i_k)$ are used as (target, context) pairs regardless of position. This is equivalent to using an infinite context window, but only within the set boundary.

## Item Similarity

After training, item similarity is computed using cosine similarity between embedding vectors:

$$\text{sim}(i, j) = \frac{\mathbf{v}_i^T \mathbf{v}_j}{\|\mathbf{v}_i\| \cdot \|\mathbf{v}_j\|}$$

Items with high cosine similarity are recommended as alternatives or complements to a queried item.

# Comparison with SVD-Based Collaborative Filtering

| Aspect | SVD (Matrix Factorization) | Item2Vec |
|---|---|---|
| Model | Decomposes user-item matrix $R \approx U \Sigma V^T$ | Neural embedding via SGNS |
| Objective | Minimize reconstruction error of $R$ | Maximize co-occurrence likelihood |
| User information | Required (rows of $R$) | Not required during inference |
| Cold-start | User cold-start problem exists | Can work without user data |
| Complexity | $O(|\mathcal{U}| \cdot |\mathcal{I}| \cdot d)$ | $O(\sum_u |\mathcal{S}_u|^2 \cdot M \cdot d)$ |
| Training | Batch (ALS, SGD on explicit matrix) | SGD with negative sampling |

> [!NOTE]
> "item2vec is capable of inferring item-item relations even when user information is not available."

This is a significant practical advantage: item2vec only requires item co-occurrence data (e.g., purchase sessions, playlist compositions, document co-citations), not explicit user-item ratings.

# Applicability

Item2Vec is applicable to any domain where:
- Items can be grouped into **sets** (unordered) or **sequences** (ordered, using windowed context)
- The goal is item-item similarity for recommendation, retrieval, or exploration
- User identity information may be sparse, noisy, or unavailable

Typical use cases:
- Music playlist generation (songs appearing in the same playlist)
- E-commerce recommendations (products purchased together)
- App recommendation (apps co-installed on the same device)
- Document similarity (papers cited together in references)

> [!CAUTION]
> Item2Vec captures **co-occurrence similarity**, not necessarily semantic causality. Highly popular items may dominate the embeddings even with frequency subsampling, and items with very few co-occurrences will have poorly trained embeddings (the cold-start item problem).

# Experiments

- **Dataset**: Microsoft Xbox music service dataset — user listening histories used as item sets (songs as items, listening sessions as sets)
- **Baseline**: SVD-based item-item CF (cosine similarity in SVD latent space)
- **Evaluation Metric**: Artists similarity evaluation using artist labels as ground truth; human-rated similarity for qualitative evaluation
- **Results**: Item2Vec achieves competitive or superior performance to SVD-based CF on item-item similarity tasks, with AUC scores close to or exceeding SVD baselines depending on configuration
- **Hyperparameters**: Embedding dimension $d \in \{100, 200\}$; negative samples $M = 15$; subsampling threshold applied; training via SGD

> [!NOTE]
> The paper's primary contribution is demonstrating the feasibility and competitiveness of applying word2vec-style neural embeddings directly to item CF without requiring explicit user-item matrix decomposition.

# Differences from Related Methods

| Method | Key Difference |
|---|---|
| Word2Vec (Mikolov 2013) | Uses ordered word sequences with fixed context window $c$; Item2Vec uses unordered sets (all pairs within set) |
| Matrix Factorization / SVD | Decomposes user-item matrix; Item2Vec only needs item co-occurrence, no user matrix needed |
| BPR (Bayesian Personalized Ranking) | Optimizes user-specific pairwise ranking; Item2Vec is user-agnostic at embedding time |
| Prod2Vec (Grbovic et al. 2015) | Applies Word2Vec to e-commerce purchase sequences; Item2Vec generalizes to unordered sets and any domain |
