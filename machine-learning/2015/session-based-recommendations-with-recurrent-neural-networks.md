# Meta Information

- URL: [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based Recommendations with Recurrent Neural Networks. ICLR 2016.

# Session-based Recommendations with Recurrent Neural Networks

## Problem Setting

Traditional collaborative filtering and matrix factorization techniques require persistent user profiles built from long interaction histories. However, many real-world scenarios lack reliable user identification:

- Anonymous browsing sessions in e-commerce
- Short-lived sessions on media streaming platforms
- Cold-start users with no prior history

**Session-based recommendation** addresses this gap by predicting the next item a user will interact with, given only the sequence of items clicked within the current session. Prior work relied on item-to-item similarity (Item-KNN) or first-order Markov chains, which discard most of the session context.

> [!NOTE]
> "Most session-based recommendation systems deployed for e-commerce are based on relatively simple methods, such as comparing co-occurrences of items."

## Architecture: GRU4Rec

The authors apply **Gated Recurrent Units (GRU)** to model the entire sequence of items within a session, allowing the model to capture long-range dependencies across clicks.

### Input Representation

Each item at step $t$ is represented as a one-hot vector $\mathbf{x}_t \in \{0,1\}^{|I|}$, where $|I|$ is the number of unique items. No user ID is used.

> [!NOTE]
> The authors also experimented with item embeddings, but the 1-of-N one-hot encoding performed better, especially for rare items.

### GRU Cell (Standard Formulation)

Given input $\mathbf{x}_t \in \mathbb{R}^{|I|}$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{d}$ (where $d$ is the hidden size):

$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1})$$
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1})$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \mathbf{x}_t + \mathbf{U}(\mathbf{r}_t \odot \mathbf{h}_{t-1}))$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Where:
- $\mathbf{r}_t \in \mathbb{R}^d$: reset gate
- $\mathbf{z}_t \in \mathbb{R}^d$: update gate
- $\sigma$: sigmoid function
- $\odot$: element-wise product

### Output Layer

The final hidden state $\mathbf{h}_t \in \mathbb{R}^d$ is projected to item scores:

$$\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{W}_o \mathbf{h}_t) \in \mathbb{R}^{|I|}$$

The model outputs a ranking score $\hat{r}_{s,i}$ for each item $i$, used to rank items for recommendation.

### Network Depth

While multi-layer GRUs were tested, a single GRU layer consistently outperformed stacked layers on both datasets. The recommended configuration is a single GRU layer with $d = 1000$ hidden units.

## Session-Parallel Mini-Batches

Standard NLP sequence batching (sliding window) is ill-suited for recommendation because sessions vary greatly in length. The authors propose **session-parallel mini-batches**:

**Algorithm: Session-Parallel Mini-Batch Construction**

```
Input: set of sessions S = {s_1, s_2, ..., s_N}
Output: sequence of mini-batches B_1, B_2, ...

1. Sort sessions by length (optional)
2. Initialize slots: slot[i] = s_i for i = 1..B  (B = batch size)
3. At each step t:
   a. For each slot i in {1..B}:
      - Read item x_{t,i} from session slot[i]
      - If session slot[i] ends at step t:
          * Reset hidden state h[i] = 0
          * Assign next unprocessed session to slot[i]
   b. Form mini-batch X_t = [x_{t,1}, ..., x_{t,B}]
   c. Forward pass GRU with X_t
   d. Compute loss on next-item targets Y_t = [x_{t+1,1}, ..., x_{t+1,B}]
```

This allows the model to process $B$ sessions simultaneously, efficiently using GPU parallelism while maintaining correct hidden state continuity within each session.

## Ranking Loss Functions

Cross-entropy loss over all items is computationally expensive for large item catalogs. The authors introduce **output sampling**: negative examples are drawn from the other items in the same mini-batch (approximating popularity-weighted negative sampling).

Let $\hat{r}_{s,i}$ be the predicted score for positive item $i$ and $\hat{r}_{s,j}$ for sampled negative item $j$. Two loss functions are proposed:

### BPR (Bayesian Personalized Ranking)

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{N_S} \sum_{j=1}^{N_S} \log \sigma(\hat{r}_{s,i} - \hat{r}_{s,j})$$

BPR maximizes the probability that the positive item is ranked above each sampled negative item.

### TOP1 (Custom Ranking Loss)

$$\mathcal{L}_{\text{TOP1}} = \frac{1}{N_S} \sum_{j=1}^{N_S} \left[ \sigma(\hat{r}_{s,j} - \hat{r}_{s,i}) + \sigma(\hat{r}_{s,j}^2) \right]$$

The first term pushes positive items above negatives. The second term acts as a regularizer, penalizing large scores for negative items and preventing score collapse.

> [!NOTE]
> TOP1 was designed to approximate the relative rank of the positive item while avoiding degenerate solutions where all scores grow unboundedly.

### Comparison of Loss Functions

| Loss | Type | Description |
|------|------|-------------|
| Cross-entropy | Pointwise | Score each item independently; unstable for large catalogs |
| BPR | Pairwise | Standard ranking loss; stable but no explicit regularization |
| TOP1 | Pairwise | Novel loss with built-in score regularization |

Empirically, both BPR and TOP1 outperform cross-entropy, with TOP1 winning on the RSC15 dataset and BPR on VIDEO.

## Comparison with Similar/Previous Methods

| Method | Session Modeling | Long-range Dependencies | Personalization |
|--------|-----------------|------------------------|-----------------|
| Item-KNN | Last item only | No | No |
| Markov Chain (FPMC) | Last item only | No | Yes (latent factors) |
| BPR-MF | Session average | No | Yes |
| GRU4Rec (this work) | Full session sequence | Yes (via GRU gates) | No (session-only) |

> [!IMPORTANT]
> Unlike FPMC (Factorized Personalized Markov Chains), GRU4Rec does not use user embeddings. It is purely session-based and designed for anonymous or cold-start users.

## Experiments

- **Dataset 1: RSC15** (RecSys Challenge 2015 — e-commerce click data)
  - Training: ~6 million sessions, 31.6 million clicks, 37,483 items
  - Test: 15,324 sessions, 71,222 events
  - Items with fewer than 5 training occurrences removed

- **Dataset 2: VIDEO** (proprietary streaming service watch data)
  - Training: ~3 million sessions, 13 million watch events, ~330,000 videos
  - Test: ~37,000 sessions, ~180,000 events

- **Metrics:**
  - **Recall@20**: Fraction of test cases where the target item appears in the top-20 recommendations
  - **MRR@20**: Mean Reciprocal Rank, truncated at rank 20

- **Key Results:**
  - GRU4Rec (TOP1, 1000 units) achieves Recall@20 = 0.6206 on RSC15, vs. Item-KNN = 0.5065 (+22.5%)
  - GRU4Rec (BPR, 1000 units) achieves Recall@20 = 0.6322 on RSC15 (+24.8% over Item-KNN)
  - On VIDEO, TOP1 gains +20.3% Recall@20 over Item-KNN
  - Single-layer GRU outperforms 2-layer stacked GRU on both datasets

- **Hardware:** GPU training; model iteration takes "a few hours"
- **Optimizer:** RMSProp with momentum (standard GRU training)

## Applicability

**Who**: Engineers and researchers building recommendation systems where persistent user profiles are unavailable or unreliable (anonymous users, cold-start users).

**When**: Session-based scenarios — e-commerce browsing, news reading, short video streaming — where only within-session item interactions are available.

**Where**: Large-scale production systems with tens of thousands to millions of items; GPU training makes this practical within a few hours per iteration.

> [!TIP]
> GRU4Rec was the foundation for subsequent work like GRU4Rec+ (Hidasi & Karatzoglou, 2018) which improved the loss function, and NARM, STAMP, and SASRec which further improved session-based recommendation using attention mechanisms.
