# Meta Information

- URL: [It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](https://arxiv.org/abs/2602.12147)
- LICENSE: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Qiao, Z., Pan, S., Wang, A., Zhukova, V., Liu, Y., Jiang, X., Wen, Q., Long, M., Jin, M., & Liu, C. (2026). It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks. arXiv:2602.12147.

# It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks

## Overview

TIME is a comprehensive evaluation framework for assessing Time Series Foundation Models (TSFMs) under zero-shot forecasting conditions. The benchmark addresses four critical gaps in existing evaluation practice: (1) reliance on legacy datasets that risk data contamination, (2) insufficient quality control leading to inconsistent data integrity, (3) misaligned task design that ignores real-world operational contexts, and (4) limited analytical perspectives that aggregate performance at the dataset level without capturing temporal-pattern-level insights.

The framework is applicable to researchers developing or comparing TSFMs, practitioners selecting models for production deployment, and benchmark designers who need principled, reproducible evaluation pipelines.

## Problem Formulation

Given a multivariate time series $\mathbf{X} \in \mathbb{R}^{M \times L}$ with $M$ variates and context length $L$, the forecasting task is to predict the next $H$ timesteps:

```math
\begin{align}
  \hat{\mathbf{X}}_{L+1:L+H} = f_\theta(\mathbf{X}_{1:L})
\end{align}
```

where $f_\theta$ is the TSFM being evaluated and $H$ is the prediction horizon. Models are evaluated in a zero-shot regime, meaning no fine-tuning occurs on any of the TIME benchmark datasets, explicitly preventing data leakage.

## Benchmark Construction

The benchmark contains **50 newly curated datasets** organized into **98 forecasting tasks** spanning nine application domains: Nature, Energy, Transportation, Healthcare, Finance, Economics, Sales, CloudOps, and Industry. All datasets were sourced from government portals, industry partnerships, and publicly available repositories not previously used in mainstream benchmarks, ensuring genuine data novelty.

### Stage 1: Manual Eligibility Assessment

Human experts first verify:
- **Format validity**: continuous time series with valid timestamps and regular sampling intervals
- **Context validation**: meaningful forecasting processes exist (e.g., not purely random noise)
- **Metadata filtering**: sufficient frequency and temporal span to support multiple evaluation windows
- **Semantic relevance**: variable labels carry domain-meaningful interpretations
- **Visual inspection**: pattern visualization to detect gross anomalies

### Stage 2: Automated Screening Pipeline

Five sequential automated steps clean the candidate datasets:

1. **Timestamp Rectification**: Detect and correct missing or misaligned timestamps to ensure uniform sampling
2. **Rule-based Validation**: Flag series violating thresholds on missing rate, minimum length, or value dominance ratio
3. **Statistical Testing (Ljung-Box)**: Identify white-noise series that carry no forecastable signal
4. **Extreme Outlier Removal**: Apply local IQR filtering with a window parameter $k = 9$ to remove transient extreme values while preserving domain-specific variation
5. **Correlation Assessment**: Compute pairwise correlations among variates to flag near-duplicate channels that would inflate effective dataset size

### Stage 3: Human Decision-Making

Domain experts review automated quality summaries enriched with LLM-generated commentary. For borderline cases, experts determine whether observed irregularities (e.g., prolonged zeros, sudden spikes) reflect genuine domain behavior or data corruption, and decide the granularity of pruning—at the full series, individual variate, or temporal-segment level.

### Stage 4: Context-Aligned Task Formulation

Rather than applying a fixed set of prediction horizons (e.g., 96, 192, 336, 720 steps as used in legacy benchmarks), TIME tailors horizons to each dataset's frequency and operational context:

- **High-frequency datasets**: three horizons—Short ($H_S$), Medium ($H_M$), Long ($H_L$)—defined to correspond to meaningful operational cycles (e.g., hour/day/week for hourly data)
- **Low-frequency datasets**: a single horizon aligned with a single operationally meaningful period (e.g., one quarter for monthly economic data)

## Pattern-Level Evaluation

The most novel analytical contribution is **pattern-level evaluation**, which stratifies models' performance by the intrinsic temporal structure of each series rather than simply aggregating across datasets.

### Temporal Feature Extraction via STL

Each time series is decomposed as:

```math
\begin{align}
  x_t = T_t + S_t + R_t
\end{align}
```

where $T_t$ is the trend component, $S_t$ is the seasonal component, and $R_t$ is the remainder, estimated via STL (Seasonal-Trend decomposition using LOESS). Seven scalar features are then computed:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Trend Strength | $F_1$ | Proportion of variance explained by trend: $1 - \mathrm{Var}(R) / \mathrm{Var}(T + R)$ |
| Trend Linearity | $F_2$ | Degree to which trend approximates a linear function (correlation of $T_t$ with a linear fit) |
| Seasonality Strength | $F_3$ | Proportion of variance explained by seasonality: $1 - \mathrm{Var}(R) / \mathrm{Var}(S + R)$ |
| Seasonality Correlation | $F_4$ | Cycle-to-cycle stability measured as correlation between consecutive seasonal cycles |
| Residual ACF-1 | $F_5$ | First-lag autocorrelation of the remainder $R_t$ |
| Complexity | $F_6$ | Spectral entropy of $x_t$; higher values indicate less structured spectra |
| Stationarity | $F_7$ | Binary indicator from the ADF test ($1$ = stationary, $0$ = non-stationary) |

### Binary Pattern Encoding

Each continuous feature $F_k$ is binarized using the population median $\tilde{F}_k$ as threshold:

```math
\begin{align}
  b_k = \begin{cases} 1 & \text{if } F_k > \tilde{F}_k \\ 0 & \text{otherwise} \end{cases}
\end{align}
```

This produces a seven-bit binary pattern code $\mathbf{b} \in \{0, 1\}^7$ per series, grouping series by their structural temporal properties rather than by source dataset. Models can then be compared within each pattern group to reveal where specific architectures excel or struggle.

## Evaluation Protocol

### Metrics

**Point forecasting** uses the Mean Absolute Scaled Error (MASE), scaled by a seasonal naive baseline:

```math
\begin{align}
  \text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{SeasonalNaive}}}
\end{align}
```

**Probabilistic forecasting** uses the Continuous Ranked Probability Score (CRPS), similarly normalized by the baseline:

```math
\begin{align}
  \text{CRPS}_{\text{norm}} = \frac{\text{CRPS}_{\text{model}}}{\text{CRPS}_{\text{SeasonalNaive}}}
\end{align}
```

Normalized scores below 1.0 indicate the model outperforms the seasonal naive baseline.

### Rolling Window Evaluation

To obtain multiple evaluation samples per dataset without overlap, a non-overlapping rolling strategy generates:

```math
\begin{align}
  W = \left\lfloor \frac{L_{\text{test}}}{H} \right\rfloor
\end{align}
```

windows per series, where $L_{\text{test}}$ is the test set length and $H$ is the prediction horizon. Each window uses a fixed context length and advances by exactly $H$ steps.

### Aggregation

Performance across windows and series is aggregated using the **geometric mean**, which is more appropriate than the arithmetic mean for ratio-scaled metrics and is less sensitive to extreme outliers.

## Evaluated Models

Twelve TSFMs spanning diverse architectures and scales are benchmarked:

| Model | Architecture | Parameters |
|-------|-------------|------------|
| TimesFM-2.5 | Decoder-only Transformer | ~500M |
| TimesFM-2.0 | Decoder-only Transformer | ~200M |
| TimesFM-1.0 | Decoder-only Transformer | ~200M |
| Moirai-2 | Encoder (Unified training) | ~311M |
| Moirai-1 | Encoder (Unified training) | ~311M |
| Chronos2 | Encoder-decoder (T5-based) | ~710M |
| Chronos-Bolt | Efficient Encoder-decoder | ~200M |
| Toto | Transformer-based | ~100M |
| Sundial | Diffusion-based | ~50M |
| VisionTS++ | Vision-based (ViT) | ~300M |
| Kairos | Retrieval-augmented | ~50M |
| TiRex | Interval-based | ~11M |

## Experiments

- **Datasets**: 50 newly curated datasets with 98 forecasting tasks across 9 domains (Nature, Energy, Transportation, Healthcare, Finance, Economics, Sales, CloudOps, Industry)
- **Evaluation setting**: Strict zero-shot; no fine-tuning on benchmark datasets
- **Baseline**: Seasonal Naive (predicts the last observed seasonal cycle)
- **Metrics**: Normalized MASE and normalized CRPS; aggregated via geometric mean
- **Leaderboard**: Interactive at [Hugging Face Spaces](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)

### Key Results

- **TimesFM-2.5** achieves the lowest normalized MASE (0.662), making it the best point forecaster overall
- **Chronos2** leads in probabilistic forecasting with the best normalized CRPS
- Newer model iterations (e.g., TimesFM-2.5 > 2.0 > 1.0; Moirai-2 > Moirai-1) consistently outperform their predecessors, validating genuine capability improvements rather than benchmark overfitting

### Pattern-Level Findings

**Trend**: Models show larger performance improvements on high-trend-strength and high-linearity series, suggesting that modern TSFMs effectively exploit clear trend signals.

**Seasonality**: On high-seasonality data, top models substantially outperform weaker ones; the performance gap narrows on unstable seasonal patterns ($F_4$ low), a challenging regime for all models.

**Stationarity**: Non-stationary series ($F_7 = 0$) yield larger gains versus the seasonal naive baseline because the naive predictor struggles more on dynamic data. Model rankings shift depending on stationarity, indicating different architectural strengths.

**Complexity**: High spectral entropy (high $F_6$) compresses inter-model performance differences, whereas low-entropy (structured) series allow superior models to differentiate clearly.

> [!NOTE]
> "No single model dominates across all temporal characteristics, suggesting specialization opportunities for domain-specific TSFM variants." This is a key finding motivating pattern-level analysis.

## Comparison with Prior Benchmarks

| Aspect | Legacy Benchmarks (e.g., LTSF-Linear) | TIME |
|--------|--------------------------------------|------|
| Datasets | Fixed legacy sets (ETT, Weather, Traffic) | 50 fresh, previously unused datasets |
| Data quality | Minimal filtering | 4-stage human-in-the-loop pipeline |
| Task design | Fixed horizons (96/192/336/720) | Context-aligned, operationally motivated horizons |
| Evaluation scope | Dataset-level aggregation | Pattern-level stratification via STL features |
| Contamination risk | High (widely used in training) | Low (novel sources) |
| Model scope | Primarily supervised trained models | Zero-shot TSFMs |

> [!IMPORTANT]
> The pattern-level evaluation methodology is architecture-agnostic and can be applied to any future TSFM. The seven STL-derived features are computable from any univariate time series and require no labels or metadata.

## Limitations

- **Pattern granularity**: With $2^7 = 128$ possible pattern codes, some pattern bins contain few series, limiting statistical power for rare combinations.
- **Metric scope**: MASE and CRPS are general-purpose; task-specific metrics (e.g., peak demand accuracy for energy) are not yet included.
- **Feature completeness**: Additional temporal characteristics (e.g., intermittency, hierarchical structure) are not captured in the current seven-feature set.
