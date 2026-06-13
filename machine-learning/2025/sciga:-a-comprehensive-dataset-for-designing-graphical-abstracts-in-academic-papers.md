# Meta Information

- URL: [SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers](https://arxiv.org/abs/2507.02212)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html); Dataset licensed under [C-UDA 1.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/c-uda.md)
- Reference: Kawada, T., Kitada, S., Nemoto, S., & Iyatomi, H. (2025). SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers. arXiv:2507.02212.

# SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers

Graphical abstracts (GAs) — single figures that visually summarize a scientific paper — have become ubiquitous in academic communication. Publishers and social media platforms embed GAs as thumbnails to attract readers, yet creating an effective GA is labour-intensive and understudied computationally. This paper introduces **SciGA-145k**, a large-scale dataset with approximately 145,000 arXiv papers and 1.14 million extracted figures, and defines two recommendation tasks (Intra-GA and Inter-GA) together with a new evaluation metric (CAR@k) to benchmark them.

**Who would use this:** Researchers building vision-language models for scientific document understanding; developers of academic search engines or social media publishing tools; scientists who want automated suggestions for which figure to highlight as a GA.

## 1. Background: Graphical Abstracts in Science

A graphical abstract is the single figure that represents an entire paper. Journals such as *Cell* require one; arXiv papers include them voluntarily. Three types of GA exist in practice:

| Type | Description | Share in SciGA |
|------|-------------|----------------|
| Original | Newly designed specifically as a GA | 20.9% |
| Reused | One existing figure selected verbatim | 64.5% |
| Modified | Combination or alteration of existing figures | 14.5% |

Because most GAs are *Reused* figures, the **Intra-GA Recommendation** task — automatically selecting the best figure from a paper to serve as its GA — is both tractable and practically valuable.

## 2. Dataset Construction

### 2.1 Paper and Figure Collection

Papers were crawled from arXiv for the period **January 2021 – March 2024**. HTML versions (via ar5iv) provided clean full text; figures were extracted from TeX source archives to avoid compression artefacts. GA images submitted to open-access journals were collected separately when available.

**Coverage:**
- 8 top-level arXiv categories, 155 sub-categories
- Cross-referenced with 11 ACM-CCS top-level categories (330 sub-categories) and 64 MSC 2022 top-level categories (5,171 sub-categories)
- Dominant fields: computer science (39.55%), mathematics (16.61%)

### 2.2 Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total papers | 144,883 |
| Total figures | 1,148,191 |
| Average figures per paper | 6.16 ± 5.86 |
| Average figures (incl. subfigures) | 7.92 ± 10.45 |
| Max figures in one paper | 700 |

**Text statistics (mean ± SD):**

| Field | Length (tokens) |
|-------|----------------|
| Title | 9.98 ± 3.57 |
| Abstract | 164.72 ± 66.32 |
| Full text | 6,403.98 ± 4,304.17 |
| Caption | 48.11 ± 44.13 |

### 2.3 Experimental Subset and Splits

The benchmark experiments use a **computer science subset** of papers with confirmed GAs or representative figures:

- **Total:** 20,520 papers
- **Train / Val / Test split:** 8 : 1 : 1

> [!IMPORTANT]
> Only papers with an identified GA (or manually-selected representative figure) are included in the benchmark subset. The full 145k dataset contains many papers without explicit GAs and is intended for pre-training and exploration.

## 3. Task Definitions

### 3.1 Intra-GA Recommendation

**Goal:** Given a paper $i$ with text $T^{(i)}$ and figures $\{I_j^{(i)}\}_{j=1}^{N_i}$, rank all figures and return the one most suitable as a GA.

- **Input:** Abstract $T^{(i)} \in$ text, figure set $\{I_j^{(i)}\} \in \mathbb{R}^{H \times W \times 3}$
- **Output:** Ranked list of figures; ideally the ground-truth GA at rank 1

**Motivation:** When a paper is shared on social media, the platform must select one thumbnail. This task asks a model to make that choice automatically.

### 3.2 Inter-GA Recommendation

**Goal:** Given a query paper $i$, retrieve GAs from *other* papers that could inspire the design of a new GA for paper $i$.

- **Input:** Abstract $T^{(i)}$ or existing GA $I_{GA}^{(i)}$
- **Output:** Ranked list of GAs from the corpus $\{I_{GA}^{(i')}\}_{i' \neq i}$

**Motivation:** Researchers designing a GA for the first time can benefit from seeing visually compelling precedents in the same field.

## 4. Evaluation Metric: CAR@k

Standard retrieval metrics (Recall@k, MRR) assume only one correct answer exists among $k$ candidates. For GAs this is overly strict — any visually informative figure could plausibly serve as a GA. CAR@k (Confidence Adjusted top-1 GT Ratio) relaxes this assumption while penalising low-confidence predictions.

### 4.1 Formulation

Let $P \in \mathbb{R}^k$ be the predicted relevance scores for the top-$k$ candidates, z-score normalised and then converted to probabilities via softmax. Let $p_\text{top-1}$ be the probability of the model's top-ranked candidate and $p_\text{GT}$ be the probability of the highest-ranked ground-truth candidate.

```math
\begin{align}
  \text{CAR@k} = \frac{p_\text{GT}}{p_\text{top-1}} \times C(P, k)
\end{align}
```

**Confidence term:**

```math
\begin{align}
  C(P, k) = 1 - \frac{1}{2} \max\!\left(0,\; \frac{H(P) - h}{H_\text{max}(P) - h}\right)
\end{align}
```

where:
- $H(P)$ — Shannon entropy of $P$
- $H_\text{max}(P) = \log k$ — entropy of the uniform distribution over $k$ items
- $h = \log(k)/2$ — confidence threshold (midpoint between uniform and peaked distributions)
- $C(P,k) \in [0.5, 1.0]$: equals 1.0 when the model is highly confident (peaked distribution), 0.5 when near-uniform

> [!NOTE]
> When the model places the ground-truth GA at rank 1, $p_\text{GT} = p_\text{top-1}$ and the first ratio equals 1; the score then reduces to the confidence term $C(P,k)$. When a different figure is ranked first, the score is scaled down proportionally.

### 4.2 Interpretation

| Scenario | CAR@k value |
|----------|-------------|
| GT at rank 1, confident prediction | Close to 1.0 |
| GT at rank 1, uncertain (uniform) prediction | Close to 0.5 |
| GT not at rank 1, confident wrong prediction | Low (penalty for overconfidence) |

## 5. Baseline Models

### 5.1 Abs2Cap — Lexical Matching Baseline

Ranks figures by TF-IDF similarity between each figure's caption $C_j^{(i)}$ and the paper abstract $T^{(i)}$. No learning involved; serves as a non-parametric lower bound.

### 5.2 GA-BC — Binary Classification

A vision-language classifier trained with binary labels (GA=1, non-GA=0). Ignores the abstract; ranks figures by the predicted GA probability.

### 5.3 Abs2Fig — Contrastive Abstract-to-Figure Retrieval

A dual-encoder model with a text encoder $f(\cdot)$ and image encoder $g(\cdot)$.

**Intra-GA ranking score:**

```math
\begin{align}
  \text{score}(T^{(i)}, I_j^{(i)}) = \rho\!\left(f(T^{(i)}),\; g(I_j^{(i)})\right)
\end{align}
```

where $\rho(\cdot, \cdot)$ denotes cosine similarity.

**Intra-GA training loss** (InfoNCE):

```math
\begin{align}
  \mathcal{L}_\text{Intra} &= \frac{1}{|B|} \sum_i \mathcal{L}_C\!\left(f(T^{(i)}),\; g(I_\text{GA}^{(i)}),\; \{g(I_{j \neq \text{GA}}^{(i)})\}\right)
\end{align}
```

**Inter-GA training loss** (bidirectional InfoNCE):

```math
\begin{align}
  \mathcal{L}_\text{Inter} = \frac{1}{2|B|} \sum_i &\mathcal{L}_C\!\left(f(T^{(i)}),\; g(I_\text{GA}^{(i)}),\; \{g(I_\text{GA}^{(i' \neq i)})\}\right) \\
  + \frac{1}{2|B|} \sum_i &\mathcal{L}_C\!\left(g(I_\text{GA}^{(i)}),\; f(T^{(i)}),\; \{f(T^{(i' \neq i)})\}\right)
\end{align}
```

The standard InfoNCE loss $\mathcal{L}_C$ with temperature $\tau$:

```math
\begin{align}
  \mathcal{L}_C(z^q, z^+, \{z_i^-\}) = -\log \frac{\exp(\rho(z^q, z^+)/\tau)}{\exp(\rho(z^q, z^+)/\tau) + \sum_i \exp(\rho(z^q, z_i^-)/\tau)}
\end{align}
```

**Backbone models tested:** CLIP, OpenCLIP, Long-CLIP, BLIP-2, X2-VLM.

### 5.4 Abs2Fig w/cap — Caption-Augmented Retrieval

Enriches the image representation by fusing the figure embedding with its caption embedding using the Hadamard (element-wise) product:

```math
\begin{align}
  \text{score}(T^{(i)}, I_j^{(i)}) = \rho\!\left(f(T^{(i)}),\; g(I_j^{(i)}) \odot f(C_j^{(i)})\right)
\end{align}
```

where $C_j^{(i)}$ is the caption of figure $j$. The caption encoder reuses the same text encoder $f(\cdot)$.

**Input/Output summary:**

| Component | Input | Output |
|-----------|-------|--------|
| Text encoder $f(\cdot)$ | Abstract $T^{(i)}$ or caption $C_j^{(i)}$ (token sequence) | $\mathbb{R}^d$ embedding |
| Image encoder $g(\cdot)$ | Figure $I_j^{(i)} \in \mathbb{R}^{H \times W \times 3}$ | $\mathbb{R}^d$ embedding |
| Fused image repr. | $g(I_j^{(i)}) \odot f(C_j^{(i)}) \in \mathbb{R}^d$ | $\mathbb{R}^d$ embedding |
| Ranking score | Abstract embed. vs. fused figure embed. | Scalar $\in [-1, 1]$ |

## 6. Comparison with Prior Work

| Aspect | SciGA-145k | FigureQA / DVQA / ChartQA | DocFigure |
|--------|-----------|--------------------------|-----------|
| Focus | GA selection & inspiration | Figure understanding/QA | Figure classification |
| Scale | 145k papers, 1.14M figures | Thousands (synthetic or small) | ~34k figures |
| Tasks | Intra/Inter-GA recommendation | Question answering | Classification |
| Text modality | Full abstract + full text | Limited captions | Captions only |
| GA annotation | Manual 3-type taxonomy | Not applicable | Not applicable |

> [!NOTE]
> No prior dataset was specifically constructed to benchmark graphical abstract selection. SciGA-145k is the first at this scale with explicit GA annotations.

# Experiments

- **Dataset:** SciGA-145k (arXiv, Jan 2021 – Mar 2024); benchmark subset of 20,520 CS papers; 8:1:1 train/val/test split
- **Hardware:** Not specified in the available text
- **Optimizer:** Not specified in the available text
- **Backbone models evaluated:** CLIP, OpenCLIP, Long-CLIP, BLIP-2, X2-VLM

**Intra-GA Recommendation results (top entries):**

| Method | Backbone | R@1 | MRR | CAR@5 |
|--------|----------|-----|-----|-------|
| Abs2Fig w/cap | Long-CLIP | **0.637** | **0.778** | **0.615** |
| Abs2Fig | BLIP-2 | 0.578 | 0.737 | 0.577 |
| Abs2Fig | CLIP | 0.573 | 0.735 | 0.573 |
| GA-BC | CLIP | ~0.5 | — | — |
| Abs2Cap | — | baseline | — | — |

**Inter-GA Recommendation results (top entries):**

| Method | Backbone | Field-P@5 | Abs2Abs SBERT@5 | GA2GA CLIP-S@5 |
|--------|----------|-----------|-----------------|----------------|
| Abs2Fig w/cap | CLIP | **0.755** | 0.493 ± 0.098 | 0.614 ± 0.067 |
| Abs2Fig w/cap | Long-CLIP | 0.753 | **0.498 ± 0.098** | 0.614 ± 0.070 |
| Abs2Fig | Long-CLIP | 0.726 | 0.456 ± 0.108 | **0.648 ± 0.056** |

A user study with 15 machine learning researchers confirmed that methods balancing visual clarity with semantic relevance were preferred over pure retrieval accuracy.

> [!CAUTION]
> Results are reported only on the computer science subset. Performance on other arXiv domains (math, physics, biology) may differ substantially given domain-specific figure styles.
