# Meta Information

- URL: [Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning](https://arxiv.org/abs/2602.09229)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Feng, X., & Watanabe, T. (2026). Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning. arXiv:2602.09229.

---

# Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning

## Overview

This paper challenges the long-standing assumption in contrastive learning that embedding magnitude is noise and should be discarded via L2-normalization. Through systematic ablation studies on both text and vision tasks, the authors demonstrate that magnitude carries task-relevant information and that whether to retain or suppress it depends on the **symmetry** of the task — specifically, whether the two sides of a pair (query and document) play distinct versus interchangeable roles.

**Applicability**: Practitioners building dense retrieval systems (text search, RAG pipelines) or cross-modal embeddings (text–image alignment) who must choose between cosine similarity and dot product. The findings directly affect model design decisions for sentence embedding libraries, bi-encoder architectures, and CLIP-style models.

---

## Background: Contrastive Learning and Similarity Metrics

### Standard Setup

Contrastive learning trains an encoder $f_\theta$ by pulling positive pairs together and pushing negative pairs apart in an embedding space. The similarity function $s(\mathbf{q}, \mathbf{d})$ between query embedding $\mathbf{q} \in \mathbb{R}^d$ and document embedding $\mathbf{d} \in \mathbb{R}^d$ can be defined in two canonical ways:

| Metric | Formula | Magnitude Effect |
|--------|---------|-----------------|
| Cosine similarity | $s = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}$ | Discards magnitude (projects onto unit hypersphere $S^{d-1}$) |
| Dot product | $s = \mathbf{q} \cdot \mathbf{d}$ | Magnitude directly scales the score |

The dominant practice in methods like SimCLR, MoCo, SimCSE, DPR, Contriever, and CLIP is to normalize embeddings to the unit hypersphere before computing similarity, treating $\|\mathbf{e}\| = 1$ as a constraint.

### The InfoNCE / NT-Xent Loss

With a batch of $N$ queries, the standard InfoNCE loss for a query $\mathbf{q}_i$ with positive document $\mathbf{d}_i^+$ and in-batch negatives $\{\mathbf{d}_j\}_{j \neq i}$ is:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s(\mathbf{q}_i, \mathbf{d}_i^+) / \tau)}{\sum_{j=1}^{N} \exp(s(\mathbf{q}_i, \mathbf{d}_j) / \tau)}$$

where $\tau > 0$ is a temperature hyperparameter and $s$ is either cosine or dot product. When $s$ = cosine, all embeddings lie on $S^{d-1}$; when $s$ = dot product, the magnitude $\|\mathbf{e}\|$ is free to vary.

---

## Key Contributions

### Three Core Findings

1. **Document magnitude correlates with relevance in text retrieval**: In asymmetric retrieval tasks (short query vs. long document), the L2-norm of relevant document embeddings is significantly larger than that of non-relevant documents. The effect size reaches Cohen's $d = 1.80$ — a very large effect — and is most pronounced on reasoning-heavy benchmarks where document complexity strongly predicts relevance.

2. **Input magnitude and output magnitude play distinct roles**: The paper decouples normalization into two sites:
   - *Input normalization*: normalizing the raw model input before the encoder
   - *Output normalization*: normalizing the final embedding vector before computing similarity

   These two affect different aspects of training: output magnitude directly scales similarity scores (affecting ranking), while input magnitude modulates gradient flow and training dynamics without directly affecting the similarity surface.

3. **Task symmetry determines whether magnitude helps or hurts**: The paper proposes a **task symmetry principle**:
   - **Asymmetric tasks** (query ≠ document in role or distribution): dot product with free magnitudes improves performance — e.g., passage retrieval (MS MARCO), retrieval-augmented generation (RAG)
   - **Symmetric tasks** (both inputs drawn from the same distribution): magnitude learning hurts — e.g., semantic textual similarity (STS), text–image alignment (CLIP-style)

---

## Normalization Ablation Framework

The authors systematically vary whether normalization is applied at the input and output sides for both the query encoder and the document encoder. This creates a $2 \times 2 \times 2$ ablation grid (input/output × query/document), yielding ablations such as:

| Configuration | Query Input Norm | Query Output Norm | Doc Input Norm | Doc Output Norm |
|---------------|-----------------|------------------|----------------|-----------------|
| Cosine (baseline) | ✗ | ✓ | ✗ | ✓ |
| Dot product (full) | ✗ | ✗ | ✗ | ✗ |
| Output-only dot product | ✗ | ✗ | ✗ | ✓ |
| Asymmetric output | ✗ | ✓ | ✗ | ✗ |

> [!NOTE]
> "Input and output magnitudes play asymmetric roles in model training: output magnitude directly scales similarity scores while input magnitude modulates training dynamics."

---

## Task Symmetry Principle

### Definition

A **symmetric task** is one where the two input sides (query and document, or image and text) are drawn from the same or equivalent distribution and play interchangeable roles. A **asymmetric task** is one where query and document come from different distributions or serve structurally different functions.

| Task | Symmetry | Recommended Similarity |
|------|----------|----------------------|
| Passage retrieval (MS MARCO) | Asymmetric | Dot product |
| Retrieval-Augmented Generation | Asymmetric | Dot product |
| Semantic Textual Similarity (STS) | Symmetric | Cosine |
| Text–image alignment (CLIP) | Symmetric | Cosine |

> [!IMPORTANT]
> The key design question is: "Does the task have distinct input roles?" If yes (asymmetric), allow magnitude to be learned via dot product. If no (symmetric), constrain to unit hypersphere via cosine similarity.

### Why Symmetry Matters

In a symmetric task, the optimal embedding of input $x_1$ and input $x_2$ should have the same magnitude if they are equally relevant. Allowing free magnitude introduces a degree of freedom that has no signal, leading the optimizer to exploit magnitude differences in ways that overfit to spurious correlations. In an asymmetric task, documents that are more relevant, longer, or more information-dense may legitimately deserve higher magnitude, which the optimizer can learn to exploit.

---

## Relationship to Prior Work

| Method | Similarity | Task Type | Magnitude Handling |
|--------|-----------|-----------|-------------------|
| SimCLR | Cosine | Symmetric (augmentation pairs) | Suppressed |
| MoCo | Cosine | Symmetric | Suppressed |
| SimCSE | Cosine | Symmetric (STS) | Suppressed |
| DPR | Dot product | Asymmetric (retrieval) | Free |
| Contriever | Cosine | Asymmetric (retrieval) | Suppressed |
| E5, GTE | Cosine | Mixed | Suppressed |
| CLIP | Cosine | Symmetric (image–text) | Suppressed |
| **This work** | Principled choice | Task-dependent | Task-symmetry principle |

> [!TIP]
> Contriever (Izacard et al., 2022) uses cosine similarity for retrieval (asymmetric task). This paper suggests that switching to dot product could yield improvements, consistent with the findings.

**Difference from DPR**: DPR (Karpukhin et al., 2020) uses dot product empirically for retrieval without systematic justification. This paper provides the theoretical and empirical grounding for *why* dot product works better in asymmetric retrieval — the magnitude acts as a relevance prior.

**Difference from CLIP**: CLIP normalizes embeddings to unit sphere, which this paper validates as correct for the symmetric text–image alignment task, since both modalities play interchangeable roles in the contrastive objective.

---

## Experimental Setup

### Datasets

| Domain | Dataset / Benchmark | Task Type | Notes |
|--------|-------------------|-----------|-------|
| Text retrieval | MS MARCO Passage Ranking | Asymmetric | Standard dense retrieval benchmark |
| Text retrieval | BEIR (multiple sub-datasets) | Asymmetric | Includes reasoning-heavy tasks (e.g., SciFact, TREC-COVID, ArguAna) |
| Text similarity | STS benchmarks (STS-B, SICK-R, etc.) | Symmetric | Standard STS evaluation via Spearman $\rho$ |
| Vision-language | CLIP-style text–image alignment | Symmetric | Zero-shot retrieval on MS COCO / Flickr30K |

> [!NOTE]
> The BEIR benchmark spans 18 heterogeneous IR tasks. The authors find that reasoning-heavy BEIR tasks (where document complexity most strongly predicts relevance) show the largest Cohen's $d$ for magnitude–relevance correlation, and thus benefit most from dot product.

### Baselines

- **Cosine similarity** (unit-norm constrained): the standard baseline matching SimCSE, Contriever, E5
- **Dot product** (unconstrained magnitude): ablation of no normalization
- **Asymmetric normalization variants**: combinations from the ablation grid above

### Evaluation Metrics

- **MRR@10** and **nDCG@10**: for retrieval tasks (MS MARCO, BEIR)
- **Spearman $\rho$**: for STS benchmarks
- **Cohen's $d$**: effect size for magnitude–relevance correlation analysis

---

## Experiments

- Dataset: MS MARCO Passage Ranking, BEIR (18 sub-tasks), STS benchmarks (STS-B, SICK-R, STS12–16), CLIP-style text–image benchmarks
- Hardware: Not specified (preliminary work under review)
- Optimizer: Not specified in available abstract; standard contrastive learning setup
- Results:
  - Cohen's $d = 1.80$ for document magnitude vs. relevance correlation on reasoning-heavy BEIR tasks
  - Dot product improves MRR/nDCG on MS MARCO and asymmetric BEIR subsets
  - Cosine outperforms dot product on STS benchmarks and text–image alignment
  - Input magnitude ablation reveals it modulates training dynamics rather than directly affecting ranking quality

---

## Practical Implications

For practitioners:

1. **Building a retrieval system** (passage search, RAG): prefer dot product similarity; do not force L2-normalization on the output layer
2. **Building a symmetric similarity system** (STS, paraphrase detection): prefer cosine similarity; L2-normalization is beneficial
3. **Text–image (CLIP-style) systems**: keep cosine similarity; the symmetric task structure means magnitude adds noise
4. **Diagnostic tool**: measure Cohen's $d$ for the magnitude vs. relevance correlation in your domain to quantify how much magnitude signal is available before deciding on the similarity metric

> [!CAUTION]
> This is preliminary work under review (as of February 2026). The experimental results, especially on vision domains, should be treated with caution until peer-reviewed. The dataset coverage for vision tasks is not fully described in available pre-publication materials.
