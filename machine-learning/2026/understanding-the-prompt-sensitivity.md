# Meta Information

- URL: [Understanding the Prompt Sensitivity](https://arxiv.org/abs/2604.18389)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Liu, Y., & Chu, C. (2026). Understanding the Prompt Sensitivity. arXiv:2604.18389.

# Understanding the Prompt Sensitivity

## Overview

This paper provides a mathematical explanation for why LLMs produce different outputs when given semantically identical prompts with minor wording variations. The authors model LLMs as multivariate continuous functions and apply first-order Taylor expansion to derive an upper bound on the log-probability difference between outputs for meaning-preserving prompt pairs. The central finding is that LLMs *disperse* semantically equivalent inputs across hidden-state space (unlike image classifiers that *cluster* same-class inputs), making it geometrically impossible to guarantee equivalent outputs.

**Applicability:** This analysis is relevant to practitioners who need consistent LLM outputs (e.g., evaluation pipelines, production NLP systems), prompt engineers designing robust templates, and researchers studying LLM reliability. It applies across decoder-only autoregressive LLMs of varying sizes.

## 2. Clustering in Traditional Neural Networks

### 2.1 Intra-class Mean Distance as a Clustering Proxy

As a baseline, the paper measures the **intra-class mean distance** in ResNet-101 layers on CIFAR-10. For class $c$ with sample set $S_c$, this is:

```math
\begin{align}
D_{\text{intra}}(c) = \frac{1}{|S_c|^2} \sum_{i,j \in S_c} \|x_i - x_j\|_2
\end{align}
```

The overall intra-class mean distance averages over all $|C|$ classes:

```math
\begin{align}
D_{\text{intra}} = \frac{1}{|C|} \sum_{c \in C} D_{\text{intra}}(c)
\end{align}
```

$D_{\text{intra}}$ decreases monotonically through Stages 1–3 of ResNet, confirming that traditional classifiers compress same-class inputs toward each other in representation space. This convergence is the geometric basis for robustness: once inputs cluster tightly, small perturbations cannot cross the decision boundary.

**Contrast with LLMs:** When the analogous measurement is run on hidden states of LLMs given meaning-preserving prompt pairs, the hidden-state difference $\|\Delta h\|_2 = \|h_1 - h_0\|_2$ *grows* from near 0 at the input embedding layer to approximately 70 at the final layer. Clustering is entirely absent.

## 3. LLMs as Multivariable Functions

### 3.1 Formal Setup

An LLM generates the next token $y_t$ from a prompt $x = (x_1, \ldots, x_n)$. Each token is embedded into $e \in \mathbb{R}^{n \times d}$ where $d$ is the model dimension, then propagated through $L$ Transformer layers. Let $h \in \mathbb{R}^d$ denote the hidden state of the last token position at the final layer. The log-probability of next token $y_t$ is:

```math
\begin{align}
\log \pi(y_t \mid h) = \log \text{softmax}(W_{\text{out}} h)_{y_t}
\end{align}
```

where $W_{\text{out}} \in \mathbb{R}^{|V| \times d}$ is the unembedding matrix and $|V|$ is the vocabulary size.

For two meaning-preserving prompts, let $h_0, h_1 \in \mathbb{R}^d$ be their respective final-layer hidden states, and define $\Delta h = h_1 - h_0 \in \mathbb{R}^d$.

### 3.2 First-Order Taylor Expansion

The log-probability difference between the two prompts is approximated by Taylor expansion around $h_0$:

```math
\begin{align}
\Delta \log \pi(y_t \mid h) = \nabla_h \log \pi(y_t \mid h_0)^\top \cdot \Delta h + \mathcal{O}(\|\Delta h\|^2)
\end{align}
```

The first-order term captures how the gradient of the log-probability at $h_0$ is projected onto the direction of hidden-state change $\Delta h$.

### 3.3 Upper Bound via Cauchy-Schwarz

Applying the Cauchy-Schwarz inequality to the first-order term yields the key bound:

```math
\begin{align}
|\Delta \log \pi(y_t \mid h)| \leq \underbrace{\|\nabla_h \log \pi(y_t \mid h_0)\|_2}_{\text{Saliency Score: SGN}(h)} \cdot \underbrace{\|\Delta h\|_2}_{\text{hidden-state dispersion}}
\end{align}
```

**Saliency Score (SGN):** The gradient norm quantifies how sensitive the output distribution is to perturbations of the hidden state:

```math
\begin{align}
\text{SGN}(h) = \|\nabla_h \log \pi(y_t \mid h)\|_2 = \sqrt{\sum_{i=1}^{d} \left(\frac{\partial \log \pi(y_t \mid h)}{\partial h_i}\right)^2}
\end{align}
```

SGN is computed via backpropagation through the softmax and unembedding layer. Input $h \in \mathbb{R}^d$; output is a non-negative scalar.

**Why the bound cannot collapse to zero:** For the upper bound to guarantee identical outputs, either $\text{SGN}(h) \to 0$ (output becomes insensitive to hidden state) or $\|\Delta h\|_2 \to 0$ (meaning-preserving prompts map to identical hidden states). LLMs achieve neither: $\|\Delta h\|_2$ grows across layers, so the upper bound stays large. This is the core theoretical explanation for prompt sensitivity.

### 3.4 Layer-Wise Distance Computation

For L2-normalized hidden states $h_0, h_1$ (unit vectors in $\mathbb{R}^d$), the Euclidean distance reduces to a function of cosine similarity:

```math
\begin{align}
\|h_0 - h_1\|_2 = \sqrt{2 - 2\cos\theta_{01}}
\end{align}
```

where $\theta_{01}$ is the angle between the two vectors. This connects dispersion in Euclidean space directly to cosine similarity, the standard metric in NLP embedding analysis.

## 4. Research Questions and Empirical Findings

### RQ1: Why Do LLMs Exhibit Prompt Sensitivity?

Layer-by-layer measurement of $\|\Delta h\|_2$ across Pythia, GPT-2, Qwen1.5, and LLaMA3.2 models consistently shows monotonically increasing dispersion from input (≈0) to the final layer (≈70). This holds regardless of model family or scale. The dispersal pattern directly inflates the upper bound $\text{SGN}(h) \cdot \|\Delta h\|_2$, preventing it from constraining $|\Delta \log \pi|$ near zero.

The activation-steering experiment in the appendix (Appendix G) manipulates $\|\Delta h\|$ at specific layers and confirms causal influence on output consistency, strengthening the theoretical account.

### RQ2: Which Prompt Modifications Cause the Greatest Sensitivity?

Seven modification types are compared:

| Modification Type | Description | Relative $\|\Delta h\|_2$ |
|---|---|---|
| Misalignment more | Significant token reordering | Highest |
| Misalignment fewer | Minor token reordering | High |
| Typographical errors | QWERTY-adjacent character substitutions | High |
| Modification latter | Synonym replacement in second half | Moderate |
| Modification first | Synonym replacement in first half | Moderate |
| Orthographic errors | Spacing, capitalization, punctuation | Moderate |
| Paraphrases | LLM-generated rephrasing | Lowest |

Key patterns:
- Token reordering produces higher $\|\Delta h\|_2$ than lexical substitution because positional encodings amplify ordering changes throughout all layers
- Typographical errors outrank orthographic errors because character-level changes alter the entire token identity (a new subword token), not just surface form
- Smaller models (Pythia-1B) are more sensitive to latter-half modifications; larger models (Qwen1.5-4B) are more sensitive to first-half modifications, suggesting differences in how positional context is weighted by scale

### RQ3: Correlation Between Upper Bound and PromptSensiScore

**PromptSensiScore (PSS)** measures empirical prompt sensitivity as the variance in model accuracy across multiple meaning-preserving templates. The theoretical upper bound $\text{SGN}(h) \cdot \|\Delta h\|_2$, averaged across layers, shows positive correlation with PSS across all tested models and datasets: models with smaller average upper bounds have lower PSS values. This validates that the bound captures the actual mechanism driving observed sensitivity.

### RQ4: Template vs. Question Contribution to Output Variance

ANOVA decomposition of logit variance attributes variance to prompt template and question content separately:
- **Prompt templates** account for 60–80% of logit variance
- **Question content** accounts for 20–40% of logit variance

The template framing dominates over the semantic content of the question in determining the model's output distribution. Qwen1.5 models show the highest absolute variance sensitivity; Pythia models show the most consistent behavior across datasets.

> [!IMPORTANT]
> This finding implies that benchmark results can be dominated by template choice rather than the model's actual knowledge, raising concerns about evaluation reliability across NLP benchmarks.

## 5. Comparison with Related Methods

| Method / Framework | Approach | Limitation Addressed by This Work |
|---|---|---|
| **PromptSensiScore** | Empirical variance across templates | No theoretical explanation — just measurement |
| **Prompt augmentation / ensembling** | Average outputs over multiple prompts | Treats symptom, not root cause |
| **Adversarial robustness (image models)** | $\ell_p$ ball perturbation bounds | Does not transfer to discrete NLP inputs with semantic equivalence |
| **Calibration methods** | Temperature scaling / post-hoc adjustment | Does not address hidden-state dispersion |
| **This work** | Taylor expansion + Cauchy-Schwarz bound | First principled mathematical explanation linking dispersal to sensitivity |

> [!NOTE]
> The first-order approximation introduces error for highly nonlinear regions. The authors acknowledge this as a limitation and propose second-order (Hessian-based) extensions as future work.

## Limitations

- Analysis is restricted to the single next-token log probability; multi-step generation is not covered
- First-order Taylor approximation may accumulate error for large $\|\Delta h\|$ where higher-order terms are non-negligible
- Does not directly propose a mitigation strategy, only an explanation

# Experiments

- **Datasets:** ARC Challenge (MCQ), CommonSenseQA (MCQ), MMLU (MCQ), OpenBookQA (MCQ), Alpaca (open-ended generation); 500 random examples per dataset
- **Templates:** 12 meaning-preserving prompt templates per sample, covering 7 modification types
- **Models:** Pythia (410M, 1B, 1.4B), GPT-2 (small, medium, large), Qwen1.5 (0.5B, 1.8B, 4B), LLaMA3.2 (1B, 3B)
- **Baseline:** ResNet-101 on CIFAR-10 (for clustering comparison in Section 2)
- **Metrics:** Per-layer $\|\Delta h\|_2$, Saliency Score $\text{SGN}(h)$, upper bound $\text{SGN}(h) \cdot \|\Delta h\|_2$, PromptSensiScore (PSS), ANOVA logit variance decomposition
- **Code:** [ku-nlp/Understanding\_the\_Prompt\_Sensitivity](https://github.com/ku-nlp/Understanding_the_Prompt_Sensitivity)
