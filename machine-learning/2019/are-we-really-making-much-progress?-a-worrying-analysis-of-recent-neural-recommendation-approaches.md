# Meta Information

- URL: [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/abs/1907.06902)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. *Thirteenth ACM Conference on Recommender Systems (RecSys '19)*.

# Overview

This paper performs a large-scale reproducibility study of 18 deep learning-based top-N recommendation algorithms published at top-tier venues (KDD, SIGIR, WWW/TheWebConf, RecSys) between 2015 and 2018. The central finding is that only 7 of 18 papers (39%) could be reproduced with reasonable effort, and 6 of those 7 reproducible neural methods failed to consistently outperform carefully tuned simple baselines—including non-personalized popularity-based and classical neighborhood-based collaborative filtering methods.

**Applicability:** This work is relevant to researchers building or evaluating recommendation systems, reviewers assessing claims of progress in deep learning for RecSys, and practitioners selecting algorithms for production deployment.

# Problem Setting

## Task: Top-N Recommendation

- **Input:** A user-item interaction matrix $R \in \{0,1\}^{|U| \times |I|}$ where $R_{ui} = 1$ if user $u$ has interacted with item $i$ (implicit feedback).
- **Output:** A ranked list of the top-$N$ items not yet seen by user $u$, ordered by predicted relevance.

The evaluation is performed in ranking mode: for each user, the model scores all unobserved items, returns the top-$N$, and the hit against a held-out test set is measured. This differs from rating prediction, yet many papers conflate the two tasks.

# Reproducibility Study Design

## Selection Criteria

Papers were included if they:
1. Proposed a **deep learning-based technique** specifically for **top-N recommendation**.
2. Were published in KDD, SIGIR, WWW/TheWebConf, or RecSys between 2015 and 2018.
3. Used at least one publicly available dataset with reconstructible train/test splits.

18 papers were identified. Authors were contacted when code was unavailable, with a 30-day response window.

## Result

| Venue | Papers Analyzed | Reproducible |
|-------|----------------|-------------|
| KDD | 4 | 3 (75%) |
| SIGIR | 6 | 2 (33%) |
| WWW | 5 | 1 (20%) |
| RecSys | 3 | 1 (33%) |
| **Total** | **18** | **7 (39%)** |

Common reasons for non-reproducibility: missing preprocessing code, no published train/test splits, proprietary datasets, and missing hyperparameter details.

# Baseline Algorithms

The authors compare reproduced neural methods against the following carefully tuned classical baselines:

## TopPopular

Recommends the globally most popular (most-interacted) items to all users regardless of personal history. Serves as a sanity-check non-personalized baseline.

## ItemKNN (Collaborative Filtering)

Item-based nearest-neighbor CF. For items $i$ and $j$, cosine similarity with a shrink term $h$:

$$s_{ij} = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\| \cdot \|\mathbf{r}_j\| + h}$$

where $\mathbf{r}_i \in \mathbb{R}^{|U|}$ is the interaction vector of item $i$. Recommendations for user $u$ aggregate the $k$ most similar items to those already consumed.

**Hyperparameters:** neighborhood size $k \in [5, 800]$, shrink term $h \in [0, 1000]$.

## UserKNN

User-based variant: similarities are computed between user interaction vectors $\mathbf{r}_u \in \mathbb{R}^{|I|}$ using the same formula. Predictions aggregate ratings from the $k$ most similar users.

## P3$\alpha$ (Pseudo-Personalized Propagation)

A graph-based random-walk algorithm. The transition probability from user $u$ to item $i$ is:

$$p_{ui} = \left(\frac{R_{ui}}{N_u}\right)^\alpha$$

where $N_u$ is the number of items user $u$ has interacted with and $\alpha \in [0, 2]$ is a damping factor controlling popularity bias. Item-to-item similarity is derived by a two-hop random walk (user → item → user → item).

## RP3$\beta$

Extends P3$\alpha$ by dividing item similarities by each item's popularity raised to power $\beta \in [0, 2]$, further suppressing popular items:

$$\tilde{s}_{ij} = \frac{s_{ij}}{N_i^\beta}$$

This soft de-popularity correction often improves diversity and accuracy simultaneously.

## SLIM (Sparse LInear Method)

Learns a sparse item-item weight matrix $W \in \mathbb{R}^{|I| \times |I|}$ by solving an Elastic Net regression:

$$\min_W \frac{1}{2}\|R - RW\|_F^2 + \lambda_1 \|W\|_1 + \frac{\lambda_2}{2}\|W\|_F^2 \quad \text{s.t.} \quad W \geq 0,\ \text{diag}(W) = 0$$

Predictions: $\hat{R} = RW$. SLIM is a linear model but competitive with or superior to many deep learning approaches.

## ItemKNN-CBF and ItemKNN-CFCBF

Content-based and hybrid variants of ItemKNN that incorporate item-side features (e.g., TF-IDF or BM25 weighted representations). The hybrid combines CF and CBF similarity matrices via a weighting parameter $\alpha$:

$$s^{\text{hybrid}}_{ij} = \alpha \cdot s^{\text{CF}}_{ij} + (1 - \alpha) \cdot s^{\text{CBF}}_{ij}$$

## Hyperparameter Tuning

All baselines were optimized using **Bayesian optimization** (via Scikit-Optimize) over 35 configurations per algorithm-dataset combination (5 random initial points, then model-guided search). Neural methods used the hyperparameters reported by original authors, ensuring a fair comparison (baselines are not advantaged by more tuning).

# Evaluated Neural Methods

The 7 reproducible papers and their main findings:

| Neural Method | Venue & Year | Outperforms Baselines? | Notes |
|---------------|-------------|----------------------|-------|
| CMN (Collaborative Memory Networks) | WWW 2018 | No | ItemKNN superior on all datasets; TopPopular superior on Epinions |
| MCRec (Meta-path Collaborative Rec.) | KDD 2018 | No | ItemKNN surpasses on all metrics |
| CVAE (Collaborative VAE) | WWW 2017 | No | Hybrid ItemKNN-CFCBF superior |
| CDL (Collaborative Deep Learning) | KDD 2015 | No | Hybrid ItemKNN-CFCBF superior at standard cutoffs |
| NeuMF (Neural Collaborative Filtering) | WWW 2017 | No | Simple baselines win on Pinterest; SLIM wins on MovieLens |
| SpectralCF | RecSys 2018 | No | All baselines including TopPopular win; data split issues found |
| Mult-VAE | WWW 2018 | **Yes** | Consistently outperforms by 10–20%; SLIM competitive on some metrics |

> [!NOTE]
> "In the large majority of the investigated cases (6 out of 7) the proposed deep learning techniques did not consistently outperform the simple, but fine-tuned, baseline methods." — Ferrari Dacrema et al. (2019)

# Evaluation Metrics

All evaluations are in **ranking mode** (top-N recommendation), not rating prediction. The paper employs:

| Metric | Formula | Sensitivity |
|--------|---------|------------|
| Hit Rate (HR@N) | $\frac{1}{|U|}\sum_u \mathbb{1}[\text{test item} \in \text{top-}N]$ | Position-insensitive |
| NDCG@N | $\frac{1}{|U|}\sum_u \frac{\log 2}{\log(\text{rank}+1)}$ | Position-sensitive, higher reward for top ranks |
| Recall@N | $\frac{1}{|U|}\sum_u \frac{|\text{relevant} \cap \text{top-}N|}{|\text{relevant}|}$ | Coverage-focused |
| Precision@N | $\frac{1}{|U|}\sum_u \frac{|\text{relevant} \cap \text{top-}N|}{N}$ | Fraction correct |
| MAP@N | Mean of per-user average precisions | Averaged across cutoffs |

> [!IMPORTANT]
> The paper finds that some neural methods implement NDCG in non-standard ways (e.g., MCRec), and evaluation cutoffs range from 3 to several hundred across papers—making direct comparison across studies unreliable without re-evaluation under a common protocol.

# Root Causes of Phantom Progress

## 1. Weak Baseline Selection

- Baselines often chosen from the same neural algorithm family (e.g., MF-based), excluding strong non-neural methods.
- Classical methods (ItemKNN, SLIM, P3$\alpha$) are routinely omitted without justification.

## 2. Insufficient Hyperparameter Tuning

- Baselines are frequently reported with default or weakly tuned parameters.
- Neural methods receive extensive tuning while baselines do not—artificially inflating the gap.

## 3. Inconsistent Evaluation Protocols

- Over 20 public datasets exist; different papers use different splits of the same dataset.
- Some papers evaluate **implicit feedback** tasks with metrics designed for **rating prediction** (e.g., RMSE).
- Leave-one-out vs. random splits vs. temporal splits produce incomparable results.

## 4. Data Leakage and Splitting Errors

- SpectralCF was found to have data split issues (test interactions leaked into training), artificially inflating results.

## 5. Publication Incentive Misalignment

> [!NOTE]
> "Research focuses on incremental accuracy improvements on narrow datasets rather than solving actual user problems or advancing understanding of why methods work."

# Comparison with Similar Work

| Aspect | Neural CF Methods (e.g., NeuMF, CMN) | Classical Methods (ItemKNN, SLIM, RP3β) |
|--------|---------------------------------------|----------------------------------------|
| Model capacity | High (non-linear, embedding-based) | Low–Medium (linear or shallow) |
| Computation | High (GPU training required) | Low (CPU, minutes) |
| Tuning effort | High (many hyperparameters) | Low–Medium (2–3 key parameters) |
| Average accuracy vs. tuned baselines | Often **lower** | Often **higher** |
| Reproducibility | Poor (39%) | High (published decades ago) |
| Interpretability | Low | High (similarity scores directly inspectable) |

> [!TIP]
> The SLIM paper: Ning & Karypis (2011), "SLIM: Sparse Linear Methods for Top-N Recommender Systems," ICDM 2011. A classic strong baseline often absent from neural RS comparisons.

# Recommendations for Future Research

The authors call for community-wide changes:

1. **Mandatory code and data sharing** with complete preprocessing scripts, not just raw data links.
2. **Rigorous baseline optimization:** simple methods must be optimized with the same effort as the proposed method.
3. **Justified metric selection:** explain why chosen metrics fit the application context.
4. **Hypothesis-driven research:** propose why a new method should work, not just that it achieves higher numbers.
5. **Broader evaluation:** test on multiple datasets with varying characteristics (density, domain, interaction type).

# Experiments

- **Datasets:** MovieLens (multiple versions), CiteULike-a, Epinions, Pinterest, Netflix Prize dataset
  - Gini index for Epinions: 0.69 (highly skewed popularity) vs. CiteULike-a: 0.37 (less skewed)
- **Hardware:** Not specified
- **Optimizer for baselines:** Bayesian optimization via Scikit-Optimize (35 configurations, 5 random seeds)
- **Optimizer for neural methods:** Authors' original settings preserved
- **Key result:** Mult-VAE is the only neural method that consistently outperforms all simple baselines; 6/7 others are matched or beaten by ItemKNN, RP3β, or SLIM.
