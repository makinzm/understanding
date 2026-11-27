# Meta Information

- URL: [A Comprehensive Review on Harnessing Large Language Models to Overcome Recommender System Challenges](https://arxiv.org/abs/2507.21117)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Raja, R., Vats, A., Vats, A., & Majumder, A. (2025). A Comprehensive Review on Harnessing Large Language Models to Overcome Recommender System Challenges. arXiv:2507.21117.

---

# Terminologies

| Term | Symbol | Description |
|------|--------|-------------|
| User embedding | $\mathbf{p}_u \in \mathbb{R}^d$ | Latent representation of user $u$ with $d$ dimensions |
| Item embedding | $\mathbf{q}_i \in \mathbb{R}^d$ | Latent representation of item $i$ with $d$ dimensions |
| Rating matrix | $R \in \mathbb{R}^{|U| \times |I|}$ | Observed interaction matrix over users $U$ and items $I$ |
| Observed interactions | $\mathcal{O}$ | Set of $(u, i)$ pairs with known ratings |
| Sparsity | — | $1 - \|\mathcal{R}_{obs}\|_0 / (|U| \cdot |I|)$; typically $<0.1\%$ density in industrial datasets |
| Cold-start | — | Condition where a new user or item has zero or minimal historical interactions |
| Implicit feedback | $o_{ui}$ | Binary signal (click, view, purchase) that proxies true preference $r^*_{ui}$ |

---

# 1. Introduction

Modern recommender systems face several persistent structural challenges that cannot be solved by pure collaborative filtering or matrix factorization alone:

1. **Cold-start problem**: New users and items have no interaction history.
2. **Data sparsity**: The interaction matrix $R$ has $<0.1\%$ density in most industrial settings.
3. **Noisy implicit feedback**: Clicks and views do not reliably reflect true preference.
4. **Temporal drift**: User preferences and item relevance change over time.
5. **Multimodal heterogeneity**: Items have diverse modalities (text, image, audio, metadata).
6. **Personalization vs. generalization tension**: Deep personalization leads to overfitting; over-regularization causes underfitting.

This survey systematically maps LLM-based solutions onto each of these six challenge categories, covering architectures from traditional collaborative filtering through graph-based models to modern instruction-tuned LLMs.

> [!NOTE]
> The authors' thesis is that LLMs function as *foundational enablers* rather than auxiliary components—they provide zero-shot generalization, semantic reasoning without explicit supervision, interpretable justifications, and adaptive personalization through dialogue.

---

# 2. Traditional Recommendation Methods (Baseline Architectures)

## 2.1 Collaborative Filtering

**User-based CF** computes a predicted rating for user $u$ on item $i$ by aggregating the ratings of the $k$ most similar users $\mathcal{N}_k(u)$:

$$\hat{r}_{u,i} = \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v) \cdot r_{v,i}}{\sum_{v \in \mathcal{N}_k(u)} |\text{sim}(u, v)|}$$

**Item-based CF** aggregates over items previously rated by the user $\mathcal{I}_u$:

$$\hat{r}_{u,i} = \frac{\sum_{j \in \mathcal{I}_u} \text{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in \mathcal{I}_u} |\text{sim}(i, j)|}$$

Similarity functions used:
- Cosine: $\text{sim}_{\cos}(i,j) = \frac{\mathbf{r}_{+,i} \cdot \mathbf{r}_{+,j}}{\|\mathbf{r}_{+,i}\| \cdot \|\mathbf{r}_{+,j}\|}$
- Pearson: $\text{sim}_P(i,j) = \frac{\sum_u (r_{u,i} - \bar{r}_i)(r_{u,j} - \bar{r}_j)}{\sqrt{\sum_u (r_{u,i} - \bar{r}_i)^2} \cdot \sqrt{\sum_u (r_{u,j} - \bar{r}_j)^2}}$

**Limitation**: Similarity degrades at large scale; cold-start users have no neighbors.

## 2.2 Matrix Factorization

Approximate $R \approx UV^\top$ where $U \in \mathbb{R}^{|U| \times d}$, $V \in \mathbb{R}^{|I| \times d}$:

$$\min_{U,V} \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \mathbf{u}_u^\top \mathbf{v}_i)^2 + \lambda(\|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2)$$

**SVD++** incorporates implicit feedback items $\mathcal{I}_u$:

$$\hat{r}_{u,i} = \mu + b_u + b_i + \left(\mathbf{u}_u + \frac{\sum_{j \in \mathcal{I}_u} \mathbf{y}_j}{\sqrt{|\mathcal{I}_u|}}\right)^\top \mathbf{v}_i$$

where $b_u, b_i$ are user and item biases and $\mu$ is the global mean.

**Content-based filtering** replaces the collaborative user profile with a hand-crafted feature vector $\mathbf{u}$:
$$\hat{r}_{u,i} = \text{sim}(\mathbf{u}, \mathbf{x}_i), \quad \text{sim}_{\cos} = \frac{\mathbf{u}^\top \mathbf{x}_i}{\|\mathbf{u}\| \cdot \|\mathbf{x}_i\|}$$

**LightFM** (hybrid) combines user/item latent vectors and feature vectors:
$$\hat{r}_{u,i} = \mathbf{u}_u^\top \mathbf{v}_i + \mathbf{u}_u^\top \mathbf{f}_i + \mathbf{f}_u^\top \mathbf{v}_i$$

## 2.3 Deep Learning Architectures

**Neural Collaborative Filtering (NCF)** replaces the inner product with an MLP:
$$\hat{r}_{u,i} = \text{MLP}([\mathbf{u}_u; \mathbf{v}_i])$$

**Wide & Deep** combines a linear (wide) component with a deep network:
$$\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + \mathbf{a}^\top f_{\text{deep}}(\mathbf{x}))$$

## 2.4 Sequence-Aware Models

**GRU4Rec** (RNN-based session recommendation):
- Input at step $t$: item one-hot or embedding $x_t$
- Hidden state: $h_t = \text{GRU}(x_t, h_{t-1})$
- Output: $\hat{y}_{t+1} = \text{softmax}(Wh_t + b)$

**SASRec** (Transformer-based sequential):
$$\mathbf{H} = \text{SelfAttention}(\mathbf{X} + \mathbf{P}), \quad \hat{y}_{t+1} = \text{softmax}(Wh_t)$$
where $\mathbf{P}$ is a positional embedding matrix.

**TiSASRec** (Temporal self-attention):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top + T_{\text{abs}} + T_{\text{rel}}}{\sqrt{d_k}}\right)V$$
$T_{\text{abs}}$ and $T_{\text{rel}}$ encode absolute and relative timestamps between items.

**BERT4Rec**: Bidirectional transformer with masked item prediction (15% of items masked per sequence).

## 2.5 Graph-Based Models

**GCN layer** aggregates neighborhood representations:
$$\mathbf{h}_v^{(k)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}} \mathbf{W}^{(k)} \mathbf{h}_u^{(k-1)}\right)$$

**NGCF** adds interaction terms:
$$\mathbf{e}_v^{(k)} = \sigma\left(\mathbf{W}_1^{(k)} \mathbf{e}_v^{(k-1)} + \mathbf{W}_2^{(k)} \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}} \mathbf{e}_u^{(k-1)}\right)$$

**LightGCN** removes non-linear activation and feature transformation:
$$\mathbf{e}_v^{(k)} = \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}} \mathbf{e}_u^{(k-1)}, \quad \mathbf{e}_v = \sum_{k=0}^K \alpha_k \mathbf{e}_v^{(k)}$$

## 2.6 Large-Scale Retrieval

**Two-Tower Model** trains user and item encoders independently:
- $\mathbf{e}_u = f_\theta(u) \in \mathbb{R}^d$, $\mathbf{e}_i = g_\phi(i) \in \mathbb{R}^d$
- Similarity: $s(u, i) = \langle \mathbf{e}_u, \mathbf{e}_i \rangle$

Candidate generation: $\mathcal{C}(u) = \{i \in \mathcal{I} \mid \text{sim}(f_u, f_i) \geq \tau\}$

Used in YouTube DNN, DLRM, Pinterest, and Spotify at production scale.

---

# 3. LLM-Based Solutions by Challenge Category

## 3.1 Cold-Start Problem

**Challenge**: New users/items have $\mathbf{p}_u = \mathbf{0}$ or $\mathbf{q}_i = \mathbf{0}$ in CF systems, making $\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i = 0$.

**LLM Solutions**:

1. **Zero-shot content-conditioned scoring**:
   $$\hat{r}_{ui} = \text{LLM}(\texttt{"User profile: "} + x_u + \texttt{" Item: "} + x_i)$$
   No interaction history required; LLM uses linguistic priors.

2. **Representation bootstrapping** for new items:
   $$\hat{\mathbf{q}}_{i'} = f_{\text{LLM}}(x_{i'}) \in \mathbb{R}^d$$
   Immediately integrates new catalog entries without retraining.

3. **Retrieval-Augmented Generation (RAG)**:
   $$P(i \mid x_u) = \text{DecoderLLM}(x_u, \mathcal{R}(x_u))$$
   where $\mathcal{R}(x_u)$ is a retrieved support set from ANN/semantic search index.

4. **Dialogue-based cold-start**: The LLM elicits preferences through natural language conversation, building a profile from explicit user responses.

> [!NOTE]
> Amazon reported ~5–10% hit-rate improvement for products with fewer than 5 historical interactions using LLM content-conditioned generation.

## 3.2 Data Sparsity

**Challenge**: Sparsity $= 1 - \|\mathcal{R}_{obs}\|_0 / (|U| \cdot |I|) > 99.9\%$ in most industrial datasets.

**LLM Solution — Semantic Embedding Matching**:
$$\hat{\mathbf{p}}_u = f_u(x_u) \in \mathbb{R}^d, \quad \hat{\mathbf{q}}_i = f_i(x_i) \in \mathbb{R}^d$$
$$\hat{r}_{ui} = \hat{\mathbf{p}}_u^\top \hat{\mathbf{q}}_i$$

LLM-derived embeddings capture semantic similarity from text descriptions, substituting for missing interaction signals.

**Multimodal Imputation** via generative completion:
$$p_\theta(x^{i,\text{miss}} \mid x^{i,\text{obs}}) = \prod_{t=1}^T p_\theta(w_t \mid w_{<t}, x^{i,\text{obs}})$$
Missing modalities (e.g., item descriptions) are auto-completed by the LLM conditioned on observed attributes.

## 3.3 Noisy Implicit Feedback

**Challenge**: The mismatch $\mathbb{P}(o_{ui} \mid r^*_{ui}) \neq \delta(r^*_{ui})$ means clicks do not deterministically indicate true preferences.

**LLM Denoising**:
$$\tilde{r}_{ui} = \text{sigmoid}(\text{LLM}(x_i, c_u))$$
where $c_u$ is the user's contextual session signal (search query, dwell time, device). The LLM interprets context to estimate intent rather than relying solely on the binary click.

**Counterfactual Reasoning**: LLMs generate contrastive explanations ("the user clicked because X but probably doesn't prefer Y") to re-weight training labels.

## 3.4 Temporal Dynamics

**Challenge**: User preference vector $\mathbf{p}_u^{(t)}$ and item relevance both shift over time, but retraining is expensive.

**Time-Aware Representation**:
$$\mathbf{p}_u^{(t)} = f(\mathbf{p}_u^{(t-1)}, \Delta t, \mathbf{x}_u^{(t)})$$

**LLM-augmented variant** (no full retraining):
$$\mathbf{p}_u^{(t)} = \text{MLP}(\mathbf{p}_u^{(t-1)} \| \text{LLM}(\mathbf{x}_u^{(t)}))$$

The LLM encodes recent behavioral text into a context vector, concatenated ($\|$) with the existing latent profile and passed through a lightweight MLP.

**TiSASRec** addresses temporal drift at the attention level through absolute/relative timestamp encoding in the attention matrix.

## 3.5 Multimodal Fusion

**Challenge**: Items have heterogeneous modalities (text description, images, audio, behavioral signals, metadata) with no natural alignment.

**Feature Combination**:
$$\mathbf{q}_i = f_{\text{text}}(x_i^{\text{text}}) + f_{\text{img}}(x_i^{\text{img}}) + f_{\text{meta}}(x_i^{\text{meta}}) + f_{\text{behav}}(x_i^{\text{behav}})$$

**Modal-Aware Gating** (soft attention over modalities):
$$\mathbf{q}_i = \boldsymbol{\gamma}_i^\top [f_{\text{text}}; f_{\text{img}}; f_{\text{meta}}], \quad \gamma_{i,m} = \frac{\exp(g_m)}{\sum_{m'} \exp(g_{m'})}$$

**Instruction-Tuned LLM Fusion** (unified encoding):
$$\mathbf{q}_i = g_{\text{pool}}(\text{LLM}(\text{Prompt}_i)) \in \mathbb{R}^d$$
A single instruction-tuned LLM processes a combined prompt containing all modalities, producing a single pooled representation.

> [!TIP]
> This approach is deployed at Pinterest and JD.com for cross-modal product recommendation, and at Spotify for audio-text-metadata fusion.

## 3.6 Personalization vs. Generalization

**Challenge**: Minimizing per-user loss risks overfitting:
$$\mathcal{L}_{\text{overfit}} = \sum_{u \in \mathcal{U}} \sum_{i \in \mathcal{I}_u} (\hat{r}_{ui} - r_{ui})^2$$

**LLM Solution — Instruction Tuning + Prompt Tuning**:
- A single LLM backbone is instruction-tuned on diverse user tasks.
- At inference, a per-user prompt adapts behavior without weight updates.
- This separates general knowledge (backbone weights) from personalization (prompt tokens).

---

# 4. Full Recommendation Pipeline

## 4.1 Multi-Stage Architecture

```
[Retrieval/Candidate Generation]
  Candidate set: C(u) = {i ∈ I | sim(e_u, e_i) ≥ τ}, |C(u)| ~ O(hundreds)

[Scoring/Ranking]
  Shared encoder: h = Encoder_θ(φ_u, φ_i, ψ_{u,i})
  Multi-task heads:
    ŷ_click  = σ(W_click · h)
    ŷ_dwell  = ReLU(W_dwell · h)
    ŷ_engage = σ(W_engage · h)
  Loss: L = λ₁·L_click + λ₂·L_dwell + λ₃·L_engage

[Post-Ranking]
  Re-ranking score: s'(u, i_j) = g(s(u, i_j), δ(i_j), γ(i_j))
  where δ(i_j) = diversity adjustment, γ(i_j) = freshness factor
```

## 4.2 LLM as Ranker

For smaller candidate sets ($|\mathcal{C}| \leq 20$), the LLM can directly rank items via prompting:
```
Prompt: "Given user history: {history}, rank these items: {C(u)} by relevance."
Output: Ordered list i_1 > i_2 > ... > i_k
```

This is deployed in dialogue-based and conversational recommendation systems (FLAN-T5, GPT-4).

---

# 5. Comparison: Traditional vs. LLM-Based Approaches

| Challenge | Traditional Approach | LLM-Based Solution | Key Advantage |
|-----------|---------------------|--------------------|---------------|
| Cold Start | Popularity heuristics, content-based heuristics | Zero-shot prompting, representation bootstrapping, RAG, dialogue | No interaction history required |
| Data Sparsity | Matrix factorization, regularization | Semantic embedding via LLM text encoders | Linguistic priors compensate for missing interactions |
| Noisy Feedback | Loss weighting, EM denoising | Contextual prompt interpretation, counterfactual reasoning | Intent-aware re-weighting |
| Temporal Drift | Periodic retraining, time-aware MF | Prompt-based adaptation, LLM-augmented temporal MLP | Adapts without full retraining |
| Multimodal Fusion | Early/late feature fusion, attention | Unified instruction-tuned LLM encoding, modal-aware gating | Implicit cross-modal alignment |
| Personalization | Separate per-user models | Instruction tuning + prompt tuning | Single backbone with per-user prompt |

---

# 6. Experiments

- **Datasets**: Netflix Prize (MF baseline), MovieLens (CF benchmarks), session-based e-commerce datasets (GRU4Rec), Amazon product reviews, Spotify (audio), Pinterest (image+text), JD.com (proprietary), YouTube (video watch history)
- **Hardware**: Production systems at Amazon, Pinterest, Spotify, Google Play, TikTok, LinkedIn (scale not specified for individual experiments)
- **Key Results**:
  - Amazon multimodal RecSys: ~5–10% hit-rate improvement for items with $<5$ interactions (cold-start regime)
  - Pinterest/JD.com: Production-scale content-conditioned generation deployed successfully
  - FLAN-T5 and GPT-4: Zero-shot recommendation demonstrated without task-specific fine-tuning
  - YouTube DNN and DLRM: Industrial baselines for two-tower retrieval, widely reproduced

> [!IMPORTANT]
> This is a survey paper, not a single-system empirical evaluation. Quantitative results cited are from referenced production systems, not a unified benchmark. Specific NDCG, MAP, or HR@K scores for individual LLM models are not presented in a consolidated table.

---

# 7. Key Takeaways for Practitioners

**Who**: ML engineers and researchers building industrial recommender systems (e-commerce, streaming, social platforms).

**When**: When facing cold-start, sparsity, or multimodal challenges; when semantic understanding of item content is important.

**Where**: Large-scale two-stage pipelines (retrieval → ranking → re-ranking) where LLMs serve at each stage with different computational budgets.

**Practical considerations**:
- RAG balances accuracy with latency better than full LLM re-ranking at scale.
- Prompt-based methods reduce compute vs. fine-tuning but may sacrifice precision.
- Hybrid pipelines (LLM representations + collaborative signals) outperform either alone.
- Federated learning variants address privacy constraints for LLM-based personalization.
