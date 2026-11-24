# Meta Information

- URL: [Prompt-Hacking: The New p-Hacking?](https://arxiv.org/abs/2504.14571)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kosch, T., & Feger, S. (2025). Prompt-Hacking: The New p-Hacking?. arXiv:2504.14571.

# Prompt-Hacking: The New p-Hacking?

## Overview

This position paper argues that the strategic manipulation of LLM prompts to obtain desired research outputs—termed *prompt-hacking*—poses a threat to scientific integrity analogous to *p-hacking* in classical statistics. The authors (Thomas Kosch, Humboldt University Berlin; Sebastian Feger, Rosenheim Technical University of Applied Sciences) identify LLMs' inherent biases, non-determinism, and opacity as fundamental obstacles to their use as rigorous data analysis tools.

**Applicability**: This work is relevant to any researcher or practitioner considering LLMs as replacements for traditional quantitative or qualitative data analysis methods, particularly in human-computer interaction (HCI), social science, and empirical software engineering contexts.

## Data Analysis in Empirical Research

Rigorous empirical research demands:

- **Quantitative methods**: Statistical validation through numeric data collection, hypothesis testing with defined significance thresholds (e.g., $p < 0.05$), and replication across independent samples.
- **Qualitative methods**: Systematic gathering of observational, interview, or case-study data with transparent codebook development and inter-rater reliability measures.

Both paradigms are vulnerable to conscious or unconscious manipulation through selective reporting, biased tool choices, and post-hoc analytical decisions.

## P-Hacking

P-hacking refers to the practice of tuning experimental data or the statistical analysis procedure until a desired (typically significant) p-value is obtained. Common tactics include:

| Tactic | Mechanism |
|--------|-----------|
| Selective variable reporting | Report only variables that yield $p < 0.05$ |
| Sample size inflation | Collect more data until significance is reached |
| Post-hoc hypothesis revision | Reframe hypotheses to match observed results (HARKing) |
| Multiple comparisons without correction | Run many tests and report significant ones |

Consequences of p-hacking include erosion of scientific trust and documented replication crises across psychology, medicine, and social science.

## Prompt-Hacking

### Definition

**Prompt-hacking** is the strategic adjustment of prompts—varying phrasing, structure, persona instructions, or context—until an LLM produces a desired output. Because LLM outputs depend heavily on both training data distribution and prompt phrasing, researchers can unintentionally or deliberately iterate prompts until the output aligns with a pre-existing hypothesis.

**PARKing** (*Prompt Adjustments to Reach Known Outcomes*) is introduced as the LLM-specific analogue of HARKing. A researcher who privately iterates dozens of prompt variants and reports only the one yielding the expected result is PARKing, regardless of whether the final prompt appears methodologically sound in isolation.

### Why LLMs Amplify the Risk

Unlike statistical methods, which are grounded in mathematically defined procedures with established validation frameworks, LLMs exhibit:

1. **Non-determinism**: The same prompt at temperature $T > 0$ produces stochastically varying outputs, making exact replication impossible without controlling model version, seed, and temperature.
2. **Opacity**: Internal reasoning is not interpretable; there is no equivalent to inspecting a test statistic or coefficient.
3. **Sensitivity to phrasing**: Minor lexical changes (e.g., "classify" vs. "label") can shift output distributions significantly.
4. **Hallucination**: LLMs generate plausible but factually incorrect outputs with no explicit signal of uncertainty, unlike confidence intervals or p-values.
5. **Bias inheritance**: Training data encodes cultural, demographic, and domain biases that propagate into analytical outputs without disclosure.

> [!NOTE]
> The authors cite Morris (2024): "prompting is a poor user interface for LLMs, which should be phased out as quickly as possible."

### Comparison: p-Hacking vs. Prompt-Hacking

| Aspect | p-Hacking | Prompt-Hacking |
|--------|-----------|----------------|
| **Tool type** | Statistical (neutral by design) | LLM (inherently biased by design) |
| **Manipulation method** | Data selection, test choice, stopping rules | Prompt variation, iteration, persona injection |
| **Reproducibility challenge** | Methods are transparent and auditable | Non-determinism resists exact replication |
| **Core risk** | Misuse of a sound, validated technique | Fundamental limitation of the tool itself |
| **Detectable with audit?** | Often yes (via pre-registration, code) | Difficult—failed prompts leave no trail |

> [!IMPORTANT]
> The authors distinguish between the two hacking types: p-hacking is the *misuse* of inherently neutral statistical techniques, whereas prompt-hacking exploits tools that are *not impartial by design*. This means that even a "correct" application of an LLM in analysis cannot guarantee validity.

## Are LLMs Appropriate for Data Analysis?

The paper proposes five foundational recommendations before employing LLMs in research:

### 1. Evaluate Necessity

Ask explicitly: why is an LLM being considered when traditional methods exist that carry no LLM-specific validity risks? Default to conventional tools unless a concrete, methodologically justified reason exists.

### 2. Assess Task Compatibility

Avoid LLMs for tasks requiring:
- Deep contextual understanding across long documents
- Impartial, bias-free interpretation
- Highly specialized domain knowledge without hallucination risk

LLMs may be appropriate for well-scoped tasks (e.g., structured extraction from short text) where outputs can be independently validated.

### 3. Standardize Prompt Use

Establish community-level or institutional guidelines defining:
- Which research tasks permit LLM use
- Required prompt formats and versioning
- Mandatory disclosure of all prompt variants attempted

### 4. Review Ethical Implications

Before deployment, audit whether LLM use could:
- Introduce cultural or systemic biases that skew findings
- Violate participant privacy through data sent to third-party APIs
- Compromise informed consent if automated analysis changes the scope of data use

### 5. Consider Reproducibility and Validity

Concrete practices for reproducible LLM use:

- **Preregister prompts**: Submit prompts and experimental protocols to a registry (e.g., Zenodo, Open Science Framework) *before* data collection.
- **Document prompt iteration history**: Record every prompt variant attempted, including those that failed or produced unexpected results.
- **Pin model versions**: Specify the exact model (e.g., `gpt-4o-2024-05-13`) and temperature setting; never use mutable aliases.
- **Repeat testing**: Run the same prompt multiple times and report variance, not just a single favored output.
- **Avoid iterating prompts to align with hypotheses**: Any prompt modification after seeing results must be disclosed and justified.

> [!NOTE]
> The authors call on infrastructure providers (Zenodo, Center for Open Science) to extend their platforms to capture preregistered prompts and metadata about the target LLM and its precise version.

## Moving Towards Ethical and Reliable Use

The paper's normative conclusion is that *whether* to use LLMs is a more important question than *how* to use them responsibly. Their recommended default position is "just don't do it" for most data analysis tasks; LLM use should require explicit justification demonstrating:

1. No traditional method can accomplish the task
2. The LLM's output will be independently validated
3. All prompting decisions are preregistered and transparently disclosed

## Differences from Related Concerns

| Concern | Description | Relation to Prompt-Hacking |
|---------|-------------|---------------------------|
| **p-Hacking** | Statistical manipulation to achieve significance | Conceptual template; prompt-hacking is worse because the tool itself is biased |
| **HARKing** | Hypothesizing After Results Are Known | PARKing is the LLM-specific instantiation of HARKing |
| **Adversarial prompting** | Attacks designed to bypass model safeguards | Different goal (breaking safety vs. confirming hypotheses), but same exploit surface |
| **Prompt injection** | Malicious instructions embedded in data | Security threat, not a research integrity threat |

> [!TIP]
> For context on p-hacking and the replication crisis, see Simmons et al. (2011) "False-Positive Psychology" and the Open Science Collaboration (2015) replication study in *Science*.

# Experiments

- Dataset: No empirical datasets; this is a position/opinion paper with no quantitative experiments.
- Hardware: N/A
- Optimizer: N/A
- Results: No quantitative results; the contribution is a conceptual framework (PARKing), a comparative taxonomy (p-hacking vs. prompt-hacking), and five actionable recommendations for research governance.
