# Meta Information

- URL: [Interestingness First Classifiers](https://arxiv.org/abs/2508.19780)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sato, R. (2025). Interestingness First Classifiers. arXiv:2508.19780.

# Interestingness First Classifiers

## Overview

This paper proposes a machine learning framework that prioritizes discovering *interesting* classifiers — those that use unexpected or non-obvious features — over maximizing predictive accuracy. The key insight is that in knowledge discovery settings (scientific research, exploratory data analysis), a classifier using a surprising feature (e.g., predicting room occupancy from humidity) can be more valuable than one using an obvious feature (e.g., predicting from light intensity), even if the former is less accurate.

**Applicability:** Researchers and data scientists in exploratory settings where generating hypotheses and novel insights matter more than peak prediction performance. Particularly suited for scientific domains, social science, and any scenario where interpretable and thought-provoking rules are the primary deliverable.

## Problem Formulation

Given a dataset $(X, y)$ where $X \in \mathbb{R}^{n \times d}$ is the feature matrix ($n$ samples, $d$ features) and $y \in \{0,1\}^n$ are binary labels, the goal is to find a classifier $\theta$ that:

$$\max_{\theta} \text{Acc}(\theta;\, X_{:, S_K},\, y) \quad \text{s.t.} \quad S_K = \{\text{top } K \text{ features by interestingness}\}$$

where $S_K \subseteq [d]$ is the selected feature subset ranked by interestingness, and $K$ is incrementally increased until the classifier achieves sufficient predictive accuracy (above a significance threshold $\alpha$).

> [!NOTE]
> The standard machine learning objective (maximize accuracy over all features) is replaced by a constrained objective that first filters features by interestingness, then optimizes accuracy within that constraint.

## Methodology: EUREKA

**EUREKA** (Exploring Unexpected Rules for Expanding Knowledge boundAries) consists of three components:

### 1. Interestingness Ranking via LLM Pairwise Comparison

An LLM is prompted to compare pairs of features $(f_i, f_j)$ and judge which would produce a more *interesting* classification rule for the given target variable. Rather than direct ranking (asking the LLM to rank all features at once), EUREKA uses **pairwise Borda counting**:

**Borda Score Estimation:**

$$\hat{b}_i = \frac{1}{n-1} \sum_{j \neq i} \Pr[i \succ j]$$

where $\Pr[i \succ j]$ is the empirical probability that feature $i$ is ranked above feature $j$ across multiple LLM queries. Features are then sorted by $\hat{b}_i$ in descending order.

> [!IMPORTANT]
> Pairwise Borda counting outperforms direct ranking in stability. Kendall $\tau$ for pairwise ranking: $0.854 \pm 0.128$ vs. direct ranking: $0.615 \pm 0.369$ on the Occupancy Detection dataset. The method is information-theoretically optimal up to constant factors (Shah & Wainwright, 2018).

### 2. Classifier Construction

EUREKA trains interpretable base models using only the top-$K$ features selected by interestingness rank:

- **Logistic Regression** (L2-regularized): $\hat{y} = \sigma(X_{:,S_K} w + b)$ where $w \in \mathbb{R}^K$, $b \in \mathbb{R}$
- **Shallow Decision Trees** (depth ≤ 3): for non-linear but still human-readable rules

Both options maintain interpretability, which is essential since the purpose is human-understandable knowledge discovery.

### 3. Interestingness-First Selection (Incremental $K$ Expansion)

**Algorithm (EUREKA):**

```
Input: Feature matrix X ∈ R^{n×d}, labels y ∈ {0,1}^n,
       significance threshold α, LLM oracle
Output: Trained classifier θ on most interesting sufficient feature subset S_K

1. Rank all d features by interestingness: [f_{(1)}, f_{(2)}, ..., f_{(d)}]
   using pairwise Borda counting with LLM comparisons

2. For K = 1, 2, ..., d:
   a. S_K ← {f_{(1)}, ..., f_{(K)}}   // top K most interesting features
   b. Train classifier θ_K on X_{:, S_K}, y
   c. Evaluate accuracy Acc(θ_K) on held-out set
   d. If Acc(θ_K) > chance level (with Bonferroni-corrected p < α):
      Return θ_K, S_K   // smallest interesting feature set with predictive power
```

> [!NOTE]
> The Bonferroni correction accounts for multiple comparisons across $K$ iterations. The chance level test ensures the selected features are genuinely predictive, not just interesting but useless.

## Differences from Similar Methods

| Method | Selection Criterion | Interpretability | Goal |
|--------|-------------------|-----------------|------|
| **EUREKA** | Interestingness (LLM-judged) | High (logistic/shallow tree) | Knowledge discovery |
| Group LASSO | Sparsity + accuracy | Medium | Feature selection for prediction |
| L2 Logistic Regression | Feature weight magnitude | Medium | Accuracy maximization |
| Validation-based selection | Single-feature accuracy | Medium | Accuracy maximization |

All conventional methods (Group LASSO, L2 logistic regression, validation-based selection) converge on the same highly-predictive features, missing the "surprising but still predictive" features that EUREKA targets.

> [!TIP]
> Borda count is a classical social choice method — see [Wikipedia: Borda count](https://en.wikipedia.org/wiki/Borda_count). Applied here to aggregate LLM preferences across pairwise feature comparisons.

## Experiments

- **Datasets:**
  - *Occupancy Detection*: Predict room occupancy from sensor readings (temperature, humidity, light, CO2, humidity ratio)
  - *Twin Papers*: Predict which of two simultaneous papers receives more future citations (~17,000 test instances); features include title characteristics (length, colons, etc.)
  - *Mammographic Mass*: Predict malignancy of mammographic masses
  - *Breast Cancer Wisconsin*: Predict breast cancer diagnosis
  - *Adult (Census Income)*: Predict whether income exceeds $50k/year; features include age, education, capital gain/loss, etc.
  - *Website Phishing*: Predict whether a website is a phishing site
- **Hardware:** Not specified
- **Optimizer:** Standard L2 logistic regression and CART decision trees
- **Preprocessing:** 80/20 stratified train/test split; numerical features standardized; categorical features one-hot encoded
- **LLM:** `gpt-4o-mini` (referred to as `gpt-5-nano` in some descriptions)
- **Significance threshold:** $\alpha = 0.05$ with Bonferroni correction

**Key Results:**

- *Occupancy Detection*: EUREKA selects `HumidityRatio` (rank 1) and `Humidity` (rank 2) over `Light` (rank 4, which is the most accurate single predictor at ~99%). Humidity-based rule achieves ~85% accuracy — lower but more *interesting*.
- *Twin Papers*: Selects features related to title punctuation (e.g., title contains colon → correlated with citation count), achieving ~52% accuracy — above chance and revealing a non-obvious bibliometric pattern.
- *Adult Income*: Selects capital losses as predictive of high income — counterintuitive since capital losses are associated with wealthier investors who actively manage portfolios.

## Limitations

- **No feature interactions:** EUREKA cannot capture interaction effects between features without explicit preprocessing (e.g., polynomial features)
- **Requires semantic feature names:** The LLM pairwise ranking relies on meaningful, human-readable feature names — numeric column IDs would not work
- **Subjectivity of interestingness:** The notion of "interesting" is inherently subjective and may vary across users, domains, or LLM models
- **Spurious correlations:** Interesting features may reflect confounding variables rather than causal mechanisms; the authors suggest post-hoc causal analysis when needed
