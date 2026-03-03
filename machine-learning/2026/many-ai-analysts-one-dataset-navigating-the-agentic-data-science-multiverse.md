# Meta Information

- URL: [Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse](https://arxiv.org/abs/2602.18710)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Bertran, M., Fogliato, R., & Wu, Z. S. (2026). Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse. arXiv:2602.18710.

---

# Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse

## Overview

This paper demonstrates that **fully autonomous AI analysts** built on large language models can reproduce the structured analytical diversity that classical "many-analyst" studies achieve only through months of coordination across dozens of research teams. By independently varying the underlying model and prompt framing, the authors show that AI agents produce wide dispersion in statistical effect sizes, p-values, and binary hypothesis decisions on identical datasets — and critically, that this dispersion is *steerable* through persona assignment and model selection.

**Applicability**: Researchers, data scientists, and organizations who deploy agentic AI systems for empirical analysis; auditors of AI-generated research who need to understand where analytical conclusions may be brittle; and AI governance practitioners designing multi-agent validation protocols.

> [!NOTE]
> Direct quote from the abstract: "The conclusions of empirical research depend not only on data but on a sequence of analytic decisions that published results seldom make explicit."

---

## Problem Setting

### The Many-Analyst Problem

In conventional empirical research, a well-known phenomenon is that different analyst teams given the same dataset and the same hypothesis regularly arrive at conflicting conclusions. Each analyst makes sequences of micro-decisions across:

| Decision Stage | Examples of Analytic Choices |
|---|---|
| Preprocessing | Outlier removal thresholds, missing data imputation, variable transformations |
| Model specification | Choice of regression family, covariate selection, interaction terms |
| Inference | Significance thresholds, multiple testing corrections, effect size reporting |

Traditional "many-analyst" studies (e.g., Silberzahn et al.) document this empirically but require months of coordination among dozens of independent teams, making them expensive and rarely conducted.

### Research Question

Can autonomous AI analysts — each independently executing a full analysis pipeline — replicate this structured analytical variability **cheaply and at scale**? And if so, is the resulting variability controllable (steerable) by adjusting the LLM or analyst persona?

---

## Framework: Agentic Data Science Multiverse

### Input / Output

| Component | Input | Output |
|---|---|---|
| AI Analyst | Fixed dataset $D$, pre-specified hypothesis $H$, assigned LLM and persona | Full analysis pipeline (code + results): effect size $\hat{\beta}$, p-value $p$, binary decision $d \in \{0,1\}$ |
| AI Auditor | One AI Analyst's complete analysis | Pass/fail flag for methodological validity |
| Study | $N$ independent AI Analyst runs with varied (LLM, persona) conditions | Distribution of $(\hat{\beta}, p, d)$ across the multiverse |

### AI Analyst Pipeline

Each AI analyst agent independently constructs and executes all stages of an analysis pipeline without human intervention:

```
Algorithm: AI Analyst Pipeline
──────────────────────────────────────────────────────────────────
Input:
  D         — fixed dataset (identical across all analyst runs)
  H         — pre-specified hypothesis to test
  model     — LLM backbone (varied across runs)
  persona   — analyst persona prompt framing (varied across runs)

1. Load dataset D
2. Perform data preprocessing
   (outlier handling, variable coding, missing value treatment)
3. Specify statistical model
   (choice of method, covariates, functional form)
4. Fit model to D
5. Conduct inference on H
   (compute effect size β̂, p-value p, binary decision d)
6. Report: (β̂, p, d, full analysis code)
──────────────────────────────────────────────────────────────────
```

### AI Auditor

An independent AI auditor agent reviews each analyst run for methodological validity (e.g., logical errors in code, inappropriate statistical choices). Runs that fail the auditor are labeled as methodologically deficient and can be filtered out, enabling separate analysis of the "clean" and "full" multiverse.

> [!IMPORTANT]
> Even after excluding methodologically deficient runs (those flagged by the auditor), the dispersion in outcomes persists and remains steerable. This means the variability is not merely noise from buggy analyses — it reflects genuine analytic flexibility.

---

## Experimental Design

### Datasets

The study uses three datasets spanning both experimental and observational research designs:

| Dataset | Design Type | Domain |
|---|---|---|
| ANES (American National Election Studies) | Observational (survey) | Political science |
| Dataset 2 | Experimental | TBD (not fully specified in abstract) |
| Dataset 3 | Observational or experimental | TBD (not fully specified in abstract) |

> [!CAUTION]
> The exact names of all three datasets are not explicitly stated in the abstract. The ANES dataset is confirmed from secondary sources; the other two datasets are described only as "spanning experimental and observational designs."

### Variation Axes

The authors vary two orthogonal dimensions across analyst runs:

1. **LLM backbone**: Different model families and capability tiers (e.g., varying sizes and versions of frontier models)
2. **Analyst persona**: Different prompt framings that assign the agent a particular analytical role, prior assumptions, or stylistic approach

These two axes create a **multiverse** of analyst configurations, each producing a different (but internally consistent) analysis pipeline.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Effect size $\hat{\beta}$ | Magnitude of the estimated relationship between variables |
| p-value $p$ | Statistical significance of the estimated effect |
| Binary decision $d \in \{0,1\}$ | Whether the analyst concludes the hypothesis is supported |
| Decision reversal rate | Fraction of multiverse analyses that flip $d$ relative to a reference |
| Dispersion | Variance or spread of $\hat{\beta}$ and $p$ across the multiverse |

---

## Key Findings

### 1. Wide Dispersion in Outcomes

Across the three datasets, AI analyst-produced analyses show **wide dispersion** in effect sizes, p-values, and binary support decisions. The hypothesis is frequently judged supported by some analysts and unsupported by others analyzing the identical data.

### 2. Structured (Not Random) Variability

The dispersion is not random noise — it is **structured along recognizable analytic dimensions**:

- Preprocessing choices (e.g., how outliers and missing data are handled) differ systematically between conditions
- Model specification choices (e.g., which covariates are included) cluster by LLM and persona
- Inference procedures (e.g., significance thresholds, two-tailed vs. one-tailed tests) vary predictably with the analyst configuration

This means the multiverse has identifiable decision forks, not arbitrary randomness.

### 3. Steerable Outcomes

Reassigning the analyst **persona** or switching the **LLM** systematically shifts the entire distribution of outcomes. This steerability holds even after the AI auditor filters out methodologically deficient runs, demonstrating that:

- The effect is not explained by some LLMs generating buggy code
- Legitimate, methodologically sound analyses diverge based on the analyst framing

> [!IMPORTANT]
> Steerability means that a bad actor (or an unknowing practitioner with a prior) could, by choosing the "right" LLM or persona, reliably obtain analyses that support or refute a pre-determined conclusion — without committing any overt methodological error.

---

## Comparison with Related Work

| Approach | Scale | Cost | Structured Variability | Steerable |
|---|---|---|---|---|
| Human many-analyst studies (e.g., Silberzahn et al.) | Dozens of teams | High (months of coordination) | Yes | Partially (depends on team framing) |
| Specification curve analysis (Simonsohn et al.) | Automated enumeration of choices | Low | Partial (pre-defined choices) | No |
| AI Analyst Multiverse (this work) | Hundreds of runs | Low (automated) | Yes | Yes (via model/persona) |

**vs. Specification Curve Analysis**: Specification curve analysis exhaustively enumerates a researcher-defined set of analytic choices; the AI multiverse allows the model itself to make those choices autonomously, discovering a broader (and potentially less anticipated) space of forking paths.

**vs. Human Many-Analyst Studies**: Human studies are costly, slow, and limited to the analysts' shared methodological training. AI multiverses are fast, cheap, and show that even LLM-based agents with different "personalities" recreate the diversity of human analytical practice.

> [!TIP]
> The "garden of forking paths" concept (Gelman & Loken) describes how researcher degrees of freedom inflate false discovery rates. This paper shows that AI agents traverse the same garden autonomously.

---

## Implications

### For AI-Assisted Research

Organizations using agentic AI for data analysis should be aware that:

1. **Single-run conclusions are unreliable**: A single AI analyst run represents one point in a broad multiverse of equally defensible analyses.
2. **Model and prompt choices are analytic choices**: Selecting an LLM or writing a persona prompt is equivalent to making methodological pre-commitments that steer conclusions.
3. **Multi-agent validation is necessary**: Running multiple independently-configured AI analysts and examining the spread of conclusions is analogous to conducting an in-house many-analyst study.

### For AI Governance

The steerability finding raises concerns about deliberate manipulation: a system that can reliably shift hypothesis decisions by changing persona prompts could be used to generate desired conclusions while maintaining a facade of methodological rigor.

### Recommended Protocol (derived from paper's findings)

```
Multi-Analyst Validation Protocol
──────────────────────────────────────────────────────────────
1. Pre-register hypothesis H before running any AI analyst
2. Run N analyst configurations, varying both LLM and persona
3. Apply AI auditor to filter methodologically deficient runs
4. Report the full distribution of (β̂, p, d), not just a single run
5. Report sensitivity of conclusions to LLM / persona variation
6. Flag conclusions that reverse direction across the multiverse
──────────────────────────────────────────────────────────────
```

---

# Experiments

- **Datasets**: Three datasets spanning experimental and observational designs; ANES (American National Election Studies) confirmed; two others unspecified in the abstract
- **Hardware**: Not specified in available sources
- **Evaluation Framework**: Inspect AI (AI agent evaluation infrastructure)
- **LLM Variants Tested**: Multiple frontier LLM models and capability tiers, varying across analyst runs
- **Persona Conditions**: Multiple prompt framings assigning different analyst identities or prior orientations
- **Results**: Wide dispersion in effect sizes, p-values, and binary hypothesis decisions across the multiverse; dispersion is structured by LLM and persona; conclusions frequently reverse across conditions; steerability confirmed after filtering methodologically deficient runs via AI auditor
