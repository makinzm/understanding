# Meta Information

- URL: [Bias Correction For Paid Search In Media Mix Modeling](https://arxiv.org/abs/1807.03292)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Chen, A., Chan, D., Perry, M., Jin, Y., Sun, Y., Wang, Y., & Koehler, J. (2018). Bias Correction For Paid Search In Media Mix Modeling. Google Inc.

# Bias Correction For Paid Search In Media Mix Modeling

## Overview

Media Mix Modeling (MMM) estimates the contribution of each advertising channel to sales using aggregated observational time-series data. A fundamental problem with paid search advertising is **selection bias**: advertisers increase search ad spend when consumer demand is already high, causing naive MMM models to conflate demand-driven sales spikes with ad-driven ones and thereby **overestimate** Return On Ad Spend (ROAS).

This paper applies Pearl's causal inference framework—specifically the back-door criterion—to derive a statistically principled **Search Bias Correction (SBC)** method. Instead of relying on expensive randomized experiments for every campaign, SBC leverages organic search query volume data as a proxy for unobserved demand, achieving unbiased ROAS estimates in common practical scenarios.

**Applicability:**
- Who: Data scientists and analysts running MMM for brands with significant paid search budgets.
- When: When selection bias is suspected (i.e., ad spend correlates with demand, not just marketing decisions).
- Where: Aggregate-level time-series MMM (weekly or daily), not user-level attribution.

---

## 1. Problem: Selection Bias in Search Advertising

In search advertising, the advertiser's bid and spend level depend on observed search query volume, which itself reflects consumer demand. Let:

- $Y$ = sales (outcome)
- $X$ = paid search spend (treatment)
- $V$ = search query volume (proxy for demand / confounder)
- $\varepsilon$ = unobserved demand shock

The **naive MMM** fits:

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

Since $\text{Cov}(X, \varepsilon) \neq 0$ (ad spend rises with demand), $\hat{\beta}_1$ is biased upward. Empirically, naive estimates can be 3–15× the true experimental ROAS.

---

## 2. Causal Framework: Pearl's Back-Door Criterion

### 2.1 Causal Diagram

The causal structure for paid search is represented as a Directed Acyclic Graph (DAG):

```
Demand (D) ──→ Query Volume (V) ──→ Organic Results ──→ Sales (Y)
                    │                                        ↑
                    └──→ Ad Spend (X) ─────────────────────→┘
```

- $D$ (unobserved demand) drives both $V$ and $Y$.
- $V$ drives both $X$ (advertiser reacts to query volume) and $Y$ (more queries → more organic conversions).
- $X$ causally affects $Y$ (the paid search effect we want to measure).

### 2.2 Back-Door Criterion

A set $\mathbf{Z}$ satisfies Pearl's **back-door criterion** for identifying the effect of $X$ on $Y$ if:
1. No element of $\mathbf{Z}$ is a descendant of $X$.
2. $\mathbf{Z}$ blocks every back-door path between $X$ and $Y$ (paths with an arrow into $X$).

**Key result:** Conditioning on query volume $V$ satisfies the back-door criterion in this DAG. Therefore:

$$\Pr(Y \mid \hat{x}) = \sum_v \Pr(Y \mid x, v)\, \Pr(v)$$

This do-calculus identity allows computing the interventional distribution $\Pr(Y \mid \hat{x})$ (the true causal effect) from observational data, provided $V$ is observed.

---

## 3. Methodology: Search Bias Correction (SBC)

### 3.1 Simple Scenario (Search-Dominant Mix)

**Assumption:** Search advertising is the primary media channel; no other channels significantly confound the model.

**Corrected model:**

$$Y = \beta_0 + \beta_1 X + f(V) + \eta$$

where:
- $Y \in \mathbb{R}^T$ — daily/weekly sales time series
- $X \in \mathbb{R}^T$ — paid search spend
- $V \in \mathbb{R}^T$ — aggregated search query volume (controls for demand)
- $f(\cdot)$ — unknown smooth function (estimated nonparametrically)
- $\eta$ — zero-mean noise uncorrelated with $X$ after conditioning on $V$

**Theorem 1:** Under the above causal diagram and the assumption that budget is unconstrained (so $X$ is determined only by $V$), $\beta_1$ is consistently estimated by OLS applied to the corrected model.

> [!NOTE]
> The "unconstrained budget" assumption means the advertiser bids freely based on query volume without a hard cap. If a cap exists, spend variation independent of demand is reduced, making identification harder.

### 3.2 Complex Scenario (Multi-Channel Mix)

**Extended setting:** Multiple channels $X_1$ (search), $X_2, \ldots, X_k$ (display, TV, etc.) all influence sales. The model extends to:

$$Y = \beta_0 + \beta_1 X_1 + g(X_2, \ldots, X_k) + f(V) + \eta$$

**Theorem 2:** Provided search spend $X_1$ is not directly constrained by the overall media budget (i.e., search spend decisions are made independently of other channel spend), $\beta_1$ remains identifiable by conditioning on $V$.

> [!IMPORTANT]
> The key distinction is between **budget-constrained** multi-channel scenarios (where reducing one channel frees budget for search, creating spurious correlations) and **independent search budget** scenarios. SBC is valid only in the latter.

---

## 4. Query Volume Summarization

Raw query data consists of millions of individual search terms. To make $V$ operationally useful, queries are classified into three aggregated volume series:

| Category | Definition | Rationale |
|---|---|---|
| **Target-favoring** ($V_1$) | Queries where advertiser's domain appears in >50% of organic results | Directly captures brand/product demand |
| **Competitor-favoring** ($V_2$) | Queries where competitor domains dominate organic results | Captures competitive demand dynamics |
| **General-interest** ($V_3$) | Category-relevant queries without clear brand dominance | Captures broad category demand |

Classification uses the actual organic search result compositions (which URLs appear), requiring access to query-level organic result data.

---

## 5. Implementation: Generalized Additive Model

The smooth function $f(V)$ and $g(\cdot)$ are estimated using a **Generalized Additive Model (GAM)** in R's `mgcv` library:

$$Y \sim \beta_0 + \beta_1 X + s(V_1) + s(V_2) + s(V_3)$$

where $s(\cdot)$ denotes a penalized regression spline (thin-plate spline by default in `mgcv`).

**Algorithm:**

```
Input:  Time series Y (sales), X (search spend), V1, V2, V3 (query volumes)
Output: Estimated β₁ (unbiased ROAS), confidence intervals

1. Classify all search queries into V1, V2, V3 by organic result composition
2. Aggregate query volumes to match the time granularity of Y and X
3. Fit GAM: Y ~ β0 + β1*X + s(V1) + s(V2) + s(V3)
   - Use REML for smoothing parameter selection
   - β1 is estimated by mgcv::gam() with fixed linear term for X
4. Extract β1 and its standard error from the fitted model
5. Compare to naive OLS (no V terms) to assess bias magnitude
```

**Software:** R package `mgcv`, REML estimation for both $\beta_1$ and the smoothing parameters.

---

## 6. Experiments

### Datasets

- **Case 1:** US retailer, 65 days of daily data, search-dominant scenario.
- **Case 2:** US advertiser, 135 days (4 months), search-dominant scenario.
- **Case 3:** US advertiser, 88 days (3 months), search-dominant scenario.
- **Case 4:** Large advertiser, 3 years (2013–2015) of daily data, 12+ media channels.

All cases were validated by comparing SBC estimates to **randomized geo-experiments** conducted by Google, which serve as ground truth ROAS measurements.

### Hardware / Software

- R with `mgcv` library (REML-based GAM fitting)
- No specific hardware requirements noted (aggregate time-series, small data)

### Key Results

**Simple scenario cases (Cases 1–3):**

| Case | Naive ROAS (bias factor) | Demand-Adjusted (bias factor) | SBC ROAS | Experiment ROAS |
|------|--------------------------|-------------------------------|----------|-----------------|
| 1    | 14.7× experiment         | 7.1× experiment               | ≈ experiment | Ground truth |
| 2    | 8.4× experiment          | 7.3× experiment               | 1.9 (SE=0.14) | ≈1.9 |
| 3    | 2.9× experiment          | 1.4× experiment               | 0.8 | ≈0.8 |

**Complex scenario (Case 4, normalized to 2013 SBC = 1.00):**

| Year | Naive | Demand-Adjusted | SBC  |
|------|-------|-----------------|------|
| 2013 | 3.43  | 2.09            | 1.00 |
| 2014 | 3.57  | 1.66            | 1.29 |
| 2015 | 3.54  | 3.03            | 1.80 |

Naive and demand-adjusted methods show no year-over-year improvement trend, while SBC reveals consistent ROAS improvement from 2013 to 2015, consistent with the advertiser's account management investments.

---

## 7. Comparison with Related Methods

| Method | Data Required | Bias Addressed | Limitation |
|---|---|---|---|
| **Naive MMM** | Aggregate spend + sales | None | Severe upward bias for search |
| **Demand-adjusted MMM** | Aggregate + macro demand signals | Partial | Does not fully block back-door path |
| **Propensity score matching** | User-level data | Selection bias | Requires user-level data; not aggregate |
| **Randomized geo-experiments** | None (runs experiment) | All confounding | Expensive, slow, not continuous |
| **SBC (this paper)** | Aggregate + query volumes | Selection bias via back-door | Requires query data; assumes unconstrained budget |

> [!TIP]
> For practitioners without access to query-level organic data (which requires search engine partnerships), approximate query volume proxies from tools like Google Trends may substitute $V$, though this reduces precision.

---

## 8. Limitations and Assumptions

- **Unconstrained budget assumption:** If search spend has a hard budget cap, the independence between $X$ and $V$ given $D$ breaks down. SBC becomes unreliable when budget constraints are binding.
- **No competitor ad confounding:** If a competitor's paid search activity simultaneously changes organic results and affects sales, this creates an unmeasured confounder not captured by $V$.
- **No significant lag effects:** The method assumes search ad effects are immediate (within the time aggregation period). Long carry-over effects violate the causal diagram.
- **Query classification accuracy:** Misclassification of queries into the wrong volume category ($V_1, V_2, V_3$) attenuates the bias correction.
- **Media synergies:** Strong interaction effects between search and other channels (e.g., TV drives search queries) can violate the causal independence assumptions in the complex scenario.

---

## Summary

SBC addresses a long-standing problem in marketing analytics: selection bias in paid search ROAS estimation from observational MMM data. By grounding the method in Pearl's causal theory and using organic query volumes as back-door adjustment variables, the approach achieves near-experimental accuracy on aggregate data without running new experiments. The GAM-based implementation in R is straightforward to apply when query volume data is available, making SBC practically viable for brands with significant search advertising budgets.
