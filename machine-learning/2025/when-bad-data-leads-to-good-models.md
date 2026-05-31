# Meta Information

- URL: [When Bad Data Leads to Good Models](https://arxiv.org/abs/2505.04741)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, K., Chen, Y., Viégas, F., & Wattenberg, M. (2025). When Bad Data Leads to Good Models. arXiv preprint arXiv:2505.04741.

---

# When Bad Data Leads to Good Models

## Overview

This paper challenges the common practice of removing toxic content from LLM pretraining corpora. The central claim is that **incorporating a moderate proportion (~10%) of toxic data during pretraining actually makes the resulting model easier to detoxify during post-training**, ultimately yielding better toxicity-capability trade-offs than models trained only on clean data.

The authors adopt a **co-design perspective**: rather than treating pretraining and post-training alignment as independent sequential stages, they argue these should be optimized jointly as a unified system.

> [!NOTE]
> "Although toxic data increases the generational toxicity of the base model, it also makes the toxicity easier to remove."

---

## 1. Problem Setup and Motivation

### Standard Practice and Its Limitation

State-of-the-art LLMs (T5, Gopher, Chinchilla, LLaMA) universally filter toxic content before pretraining. This heuristic assumes that cleaner pretraining data yields a safer final model. The paper questions whether this assumption holds once post-training alignment is applied.

### Key Insight

Toxic content, being a minority in web-scraped data, is an **underrepresented feature**. Underrepresented features in neural networks tend to become entangled with other features due to the limited representational capacity (the superposition hypothesis). By deliberately increasing the presence of toxic content in pretraining, the model develops a **more dedicated, less entangled linear representation** of toxicity, making it straightforwardly steerable in post-training.

---

## 2. Theoretical Framework: Feature Entanglement

The analysis builds on the **superposition hypothesis** (Elhage et al., 2022): when the number of features $N$ exceeds the dimensionality $M$ of the representation space, features cannot all be orthogonal and must share representational dimensions (superposition).

### Entanglement Measure

For a feature $P_i$ with unit direction vector $v_{P_i} \in \mathbb{R}^M$, entanglement is defined as:

```math
\begin{align}
  \mathcal{E}_{P_i} = \max_{j \in [N] \setminus \{i\}} \left| v_{P_i} \cdot v_{P_j} \right|
\end{align}
```

This is the maximum absolute cosine similarity between feature $P_i$ and all other features. Lower $\mathcal{E}_{P_i}$ means the feature is more orthogonal to others and thus easier to isolate via linear probing or activation steering.

### Welch Bound (Theoretical Lower Bound on Entanglement)

When $N > M$ (more features than dimensions), the minimax entanglement satisfies:

```math
\begin{align}
  \max_i \mathcal{E}_{P_i} \geq \sqrt{\frac{N - M}{(N-1)M}}
\end{align}
```

Equality holds when feature directions are evenly distributed (equiangular tight frame). The bound decreases as $M$ increases or as $N$ decreases — implying that reducing the effective number of features competing for a fixed representational space lowers entanglement.

### How Frequency Affects Entanglement

In practice, models allocate more representational capacity to frequently occurring features. By increasing toxic data from ~0% to ~10%, toxic concepts transition from being a rare, compressed feature to a well-represented one, receiving more dedicated directions in the residual stream and fewer conflicts with unrelated concepts.

---

## 3. Toy Experiment

To verify the frequency-entanglement relationship in a controlled setting, the authors use synthetic Markov chain data.

### Setup

- **3 Markov chains**, each with a **vocabulary size of 4** distinct tokens (12 total features)
- The **residual stream has 4 dimensions** ($M = 4$), so $N = 12 \gg M = 4$ — heavy superposition is guaranteed
- One chain has its transition probability varied to simulate changing the frequency of one feature set
- Models are trained **10 times with different random seeds** for statistical reliability

### Result

As the frequency of the targeted chain increases, the measured entanglement $\mathcal{E}$ of its features decreases, consistent with the theoretical prediction. This toy experiment validates that deliberate overrepresentation of a minority concept reduces its entanglement.

---

## 4. Probing Methodology

To quantitatively measure how well toxicity is linearly represented in a pretrained model's activations, the authors train **binary linear probes** on intermediate activations.

### Probe Training

- **Input**: Hidden state $x^h_l \in \mathbb{R}^{1024}$ at attention head $h$, layer $l$, final token position
- **Label**: Binary toxicity label $y \in \{0, 1\}$ (from human annotation; threshold: Perspective API score $\geq 2.5$)
- **Dataset**: ToxiGen subset — **8,960 labeled samples**, split 4:1 train/validation
- **Classifier**: Linear binary classifier (logistic regression), evaluated by **validation accuracy**

### Interpretation

Higher probe accuracy at a given layer/head indicates that toxicity is more linearly separable in that head's activation space — i.e., less entangled with unrelated concepts, and thus more amenable to linear steering.

### Finding

Models pretrained with toxic data show **significantly higher probe accuracy** across multiple layers and heads compared to clean-only models (statistical test: $p = 0.0002$, 95% CI for the difference: $[0.67, 1.18]$). The distribution of accuracies shows a "fatter right tail" — more attention heads achieve high probe accuracy — indicating more specialized, redundant representations of toxicity.

---

## 5. Detoxification Methods (Post-Training)

The paper evaluates five detoxification approaches applied uniformly to models with varying toxic data proportions:

| Method | Type | How It Works |
|---|---|---|
| **ITI** (Inference-Time Intervention) | Activation steering | Shifts activations along identified toxicity direction during decoding |
| **Prompting** | Inference | Prepends ethical instruction to the prompt |
| **SFT** | Fine-tuning | Supervised fine-tuning on non-toxic completions |
| **DPO** | Fine-tuning | Direct Preference Optimization with toxic/non-toxic pairs |
| **MEDA / INST** | Baselines | Existing detoxification methods |

### ITI Details

ITI identifies a linear direction $v_{\text{tox}} \in \mathbb{R}^{1024}$ corresponding to toxicity by comparing mean activations of toxic vs. non-toxic samples across the **top-30 attention heads** selected by probe accuracy. During generation, activations at these heads are shifted:

```math
\begin{align}
  x'^h_l = x^h_l - \alpha \cdot v_{\text{tox}}
\end{align}
```

where $\alpha$ is the intervention strength. Three strengths are tested: **weak** ($\alpha = 4$), **medium** ($\alpha = 8$), **strong** ($\alpha = 12$).

---

## 6. Main Experiments

### Model and Data

- **Base model**: OLMo-1B (24 layers, 16 attention heads, hidden size 1024)
- **Training data**: Mix of **C4** (clean web text) and **4chan posts** (toxic content)
- **Toxic proportions tested**: 0%, 5%, 10%, 15%, 20%, 25%
- **Total training tokens**: 20.1–25.7 billion
- **Hardware**: 16 NVIDIA H100 GPUs; each run completes within ~12 hours

### Evaluation Datasets

| Dataset | Purpose | Size |
|---|---|---|
| **ToxiGen** | Toxicity evaluation (implicit hate) | 3,000 prompts sampled |
| **Real Toxicity Prompts** | Toxicity evaluation (explicit) | 3,000 prompts sampled |
| **MMLU** | General capability (57 subjects) | Standard benchmark |
| **Open Web Text** | Capability: cross-entropy loss | Subset |
| **AdvBench** | Red-teaming (GCG attack) | 200 adversarial prompts |

Toxicity is measured via **Perspective API** (scores 0–100; lower is less toxic).

### Key Results

With **10% toxic data** during pretraining and **strong ITI** ($\alpha = 12$) post-training:

| Method | ToxiGen Score ↓ | Real Toxicity Prompts ↓ | Cross-Entropy ↓ |
|---|---|---|---|
| Clean only (0% toxic) + strong ITI | 19.82 | 13.33 | — |
| **10% toxic + strong ITI** | **2.63** | **7.11** | 7.11 |
| MEDA (baseline) | ~18 | ~12 | — |
| INST (baseline) | ~15 | ~10 | — |

The **alignment tax** (capability degradation) is also lower for 10% toxic models, as measured by cross-entropy loss on general web text.

### Diminishing Returns Beyond 10%

At 15%, 20%, 25% toxic proportions, the detoxification gains level off while base toxicity increases, resulting in a worse overall trade-off. The 10% level is the empirical sweet spot.

### Red-Teaming Results

GCG adversarial attack success rate (lower is safer):
- Clean-only model: **46%** success rate
- 10% toxic model after detoxification: **38.5%** success rate

The improvement is modest, suggesting activation steering has limited robustness against adversarial prompt optimization.

---

## 7. Comparison with Similar Methods

| Approach | Where Toxicity Is Addressed | Entanglement Effect |
|---|---|---|
| **Traditional filtering** (LLaMA, Gopher) | Pretraining data removal | Toxicity remains entangled (rare feature) |
| **RLHF / DPO** | Post-training fine-tuning | Attempts to suppress entangled feature |
| **Controlled generation** (vocabulary shift) | Inference-time vocabulary manipulation | Does not address representation quality |
| **This work (co-design)** | Pretraining data composition + post-training steering | Reduces entanglement, enabling clean steering |

The key distinction is that prior methods treat pretraining and post-training as **independent stages** and optimize each separately. This paper proposes treating them as a **joint optimization problem**, where intentional data composition in pretraining creates a more steerable representation.

---

# Experiments

- **Datasets**: C4 (clean pretraining), 4chan (toxic pretraining), ToxiGen (3,000 eval prompts), Real Toxicity Prompts (3,000 eval prompts), MMLU (capability), Open Web Text (cross-entropy), AdvBench (200 red-team prompts)
- **Hardware**: 16 NVIDIA H100 GPUs
- **Optimizer**: Not explicitly specified (standard OLMo training setup)
- **Results**: 10% toxic pretraining + strong ITI reduces ToxiGen toxicity from 19.82 → 2.63 (vs. clean-only baseline), with lower alignment tax and 38.5% vs. 46% GCG attack success rate

## Limitations

- Experiments conducted only on OLMo-1B; generalization to larger scales is unconfirmed
- Toxicity operationalized exclusively via Perspective API; other bias/harm types not studied
- The co-design principle may require careful calibration per domain; the 10% sweet spot may shift with different corpora or model architectures
- Red-teaming improvements are modest, suggesting adversarial robustness is not fully addressed by this approach
