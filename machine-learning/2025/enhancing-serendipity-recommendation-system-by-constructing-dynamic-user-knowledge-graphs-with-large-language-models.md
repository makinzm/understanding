# Meta Information

- URL: [Enhancing Serendipity Recommendation System by Constructing Dynamic User Knowledge Graphs with Large Language Models](https://arxiv.org/abs/2508.04032)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yong, Q., Li, Y., Shi, J., Dou, Y., & Qi, T. (2025). Enhancing Serendipity Recommendation System by Constructing Dynamic User Knowledge Graphs with Large Language Models. arXiv:2508.04032.

# Overview

## Problem: Filter Bubble in Recommendation Systems

Recommendation systems deployed at scale suffer from a feedback loop problem called the **filter bubble**: the system learns to recommend only items similar to what users have already engaged with, causing the recommendation distribution to converge toward a narrow, homogeneous set of items. This reduces serendipity—the discovery of unexpectedly relevant content—and long-term user satisfaction even as short-term engagement metrics appear healthy.

The paper formalizes serendipity as a combination of **novelty** (items outside the user's historical consumption pattern) and **relevance** (alignment with the user's actual interests). The challenge is that latent interests not yet expressed in interaction history are unobservable from standard collaborative filtering signals.

## Solution Overview

The paper proposes a two-stage framework deployed at Dewu (得物), a trendy e-commerce platform with tens of millions of users:

1. **Stage 1 — Dynamic Knowledge Graph Construction**: Use LLMs to infer latent user interests by performing two-hop semantic reasoning over a user knowledge graph built from static profile attributes and 30-day search history.
2. **Stage 2 — Nearline Interest-Augmented Retrieval**: Cache LLM-generated interest nodes for 7 days and serve them to a dual-tower retrieval model that incorporates both user-to-item and item-to-item matching signals with contrastive learning.

This framework enables serendipitous recommendations without sacrificing the latency constraints of industrial-scale serving.

# Stage 1: Dynamic User Knowledge Graph Construction

## Input Representation

Each user is represented by:
- **Static profile**: age group, gender (categorical attributes)
- **Behavioral history**: search queries over the past 30 days (free-text)

These are composed into a natural language prompt and fed to an LLM.

## Two-Hop Semantic Reasoning

Rather than directly predicting items or categories, the LLM performs two hops on a semantic graph:

**Hop 1 — Hypernym extraction (core demand identification)**

The LLM generalizes from surface-level search terms to abstract user demands via hypernym relationships. For example, a user who repeatedly searches for "Nike Air Max" may have the core demand "sports shoes" or "streetwear aesthetics."

**Hop 2 — Hyponym/co-hyponym expansion (latent interest discovery)**

From the core demands identified in Hop 1, the LLM then generates more specific subcategories or laterally related categories (co-hyponyms). For example, from "streetwear aesthetics," the model may expand to "skateboard accessories," "vintage caps," or "graphic tees"—categories the user has not yet searched for but which are semantically adjacent to their expressed interests.

> [!NOTE]
> "Multiple language model instances engage in multi-round proposing and debating" to improve the accuracy and diversity of generated interest nodes. This multi-agent debate scheme reduces single-model hallucination by having agents challenge each other's outputs.

The output of Stage 1 is a set of **interest node embeddings** $\{e_1, e_2, \ldots, e_K\} \in \mathbb{R}^{K \times d}$, where $K$ is the number of generated interest categories and $d$ is the embedding dimension produced by CLIP's text encoder.

## Model Distillation: InterestGPT

Running a large LLM at inference time for every user is computationally infeasible. The authors use **model distillation**:

1. Use **DeepSeek-R1** (large, high-quality reasoning model) to generate 30,000 labeled (user profile, behavioral history) → (interest nodes) training examples.
2. Fine-tune **QWQ-32B** on this labeled dataset to produce a smaller, deployable model called **InterestGPT**.

InterestGPT retains the two-hop reasoning capability of DeepSeek-R1 while fitting within latency budgets.

> [!IMPORTANT]
> The nearline deployment strategy caches InterestGPT's outputs for each user for a 7-day window. Interest nodes are refreshed asynchronously, so online serving reads from cache rather than calling the LLM in real-time. This is the key design choice that makes LLM-augmented retrieval feasible at industrial scale.

# Stage 2: Nearline Interest-Augmented Retrieval Model

## Dual-Tower Architecture

The retrieval model follows the standard dual-tower (two-tower) design for approximate nearest-neighbor retrieval at scale:

- **User tower**: Takes user features $u \in \mathbb{R}^{d_u}$ and interest embeddings $E_{\text{interest}} \in \mathbb{R}^{K \times d}$ as input. Concatenates and processes through fully connected layers to produce user representation $h_u \in \mathbb{R}^{d_h}$.
- **Item tower**: Takes item features $v \in \mathbb{R}^{d_v}$ and item-side interest embeddings as input, processed similarly to produce item representation $h_v \in \mathbb{R}^{d_h}$.

Relevance score: $s(u, v) = h_u^\top h_v$

## Multi-Task Training Objective

The model is trained with two losses:

**1. Binary Cross-Entropy Loss (standard retrieval task)**

```math
\begin{align}
  \mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
\end{align}
```

where $y_i \in \{0, 1\}$ is the click label and $\hat{y}_i = \sigma(s(u_i, v_i))$.

**2. Interest-Aligned Contrastive Loss (novelty-aware task)**

The contrastive loss aligns user-item pairs that share the same LLM-generated interest node:

```math
\begin{align}
  \mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(h_{u_i}, h_{v_i^+}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(h_{u_i}, h_{v_j}) / \tau)}
\end{align}
```

where $v_i^+$ is a positive item (an item whose interest embedding matches the user's LLM-generated interest), $\tau$ is a temperature hyperparameter, and $\text{sim}(\cdot, \cdot)$ is cosine similarity.

**Combined loss**:

```math
\begin{align}
  \mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{contrast}}
\end{align}
```

where $\lambda$ balances the two objectives.

## Hybrid Retrieval: Balancing Serendipity and Conversion

The model combines two retrieval paradigms:
- **User-to-item (U2I)**: Traditional collaborative filtering signal based on user-item interaction history. High conversion rate but low novelty.
- **Item-to-item (I2I)**: Interest-node-guided retrieval that finds items semantically related to the LLM-generated interests. Higher novelty but potentially lower direct conversion.

The final retrieval pool merges candidates from both U2I and I2I channels, with downstream ranking models scoring and reranking for final exposure.

## Difference from Standard Collaborative Filtering

| Aspect | Standard CF | This Work |
|---|---|---|
| Interest signals | Observed click/purchase history only | LLM-inferred latent interests via knowledge graph |
| Interest coverage | Restricted to in-distribution items | Can retrieve out-of-distribution novel categories |
| Latency | Low (embedding lookup) | Managed via nearline caching (7-day window) |
| Reasoning depth | Single-hop (user → item) | Two-hop (user → core demand → latent interest) |
| Training signal | Click labels only | Click labels + interest-aligned contrastive pairs |

> [!TIP]
> The two-hop knowledge graph reasoning is conceptually related to knowledge graph embedding methods (e.g., TransE, RotatE) but here the graph is *dynamically generated per user* by an LLM rather than pre-built from a static knowledge base.

# Experiments

- **Platform**: Dewu (得物) — a trendy e-commerce platform serving tens of millions of users
- **Offline evaluation dataset**: 1,000 labeled test samples scored by human evaluators (novelty scale 0–2)
- **SFT dataset for InterestGPT**: 30,000 labeled (user profile + history) → (interest nodes) examples generated by DeepSeek-R1
- **Text encoder**: CLIP (for encoding interest node text into embeddings $\in \mathbb{R}^d$)
- **Online A/B test**: 10% traffic split on production system (~tens of millions of users)

**Offline Results (novelty quality of LLM-generated interests)**:

| Score | Proportion |
|---|---|
| 2 (maximum novelty) | 96% |
| 1 (intermediate) | 3% |
| 0 (minimum) | 1% |

**Online A/B Test Results (serendipity channel vs. control group)**:

| Metric | Change |
|---|---|
| Exposure Novelty Rate (ENR) | +4.62% |
| Click Novelty Rate (CNR) | +4.85% |
| Avg. View Duration per User (AVDU) | +0.15% |
| Unique Visitor CTR (UVCTR) | +0.07% |
| User Interaction Penetration | +0.30% |

> [!NOTE]
> The serendipity channel achieves a 26.53% novelty rate compared to 16.17% for other recommendation channels and 14.24% for the control group baseline—a statistically meaningful improvement in out-of-distribution discovery.

# Applicability

- **Who**: Recommendation system engineers at e-commerce, content, or media platforms facing filter bubble degradation and declining long-term retention.
- **When**: Applicable when (a) user interaction logs are available for 30+ days, (b) static user profile attributes exist, and (c) latency requirements allow for nearline (async) LLM inference with caching.
- **Where**: Industrial-scale retrieval stages that precede ranking layers. The framework plugs in as an additional retrieval channel (alongside existing U2I/I2I channels) rather than replacing the entire recommendation pipeline.
- **Limitation**: The 7-day cache window means interest nodes are not updated in real-time, which may lag sudden changes in user taste. The quality of InterestGPT's outputs depends heavily on the quality of DeepSeek-R1 distillation data.
