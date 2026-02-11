# Meta Information

- URL: [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](https://arxiv.org/abs/2510.22037)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Longpre, S., Kudugunta, S., Muennighoff, N., Hsu, I.-H., Caswell, I., Pentland, A., Arık, S. Ö., Lee, C.-Y., & Ebrahimi, S. (2025). ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality. arXiv:2510.22037.

# ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining

## Overview

ATLAS presents the largest multilingual scaling law investigation to date, covering 774 training experiments with models from 10M to 8B parameters, trained on 400+ languages and evaluated on 48 languages. The core contribution is a new scaling law that explicitly models cross-lingual transfer, enabling practitioners to predict optimal compute allocation for multilingual language models.

**Applicability:** Researchers and engineers building multilingual language models who need to decide: (1) how to scale model size and training data budget across languages, (2) how to select beneficial language mixtures, and (3) whether to pretrain from scratch or fine-tune from existing multilingual checkpoints.

## Background: Chinchilla and Its Limitations

The standard Chinchilla scaling law (Hoffmann et al., 2022) models loss as:

$$\mathcal{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:
- $N$ = number of model parameters
- $D$ = number of training tokens
- $E, A, B, \alpha, \beta$ are fitted constants

**Limitation for multilingual settings:** Chinchilla treats all training tokens as equivalent regardless of language. In multilingual pretraining, tokens in different languages contribute differently to target language performance due to cross-lingual transfer. Chinchilla also fails to model repeated data (multi-epoch training), which is necessary for low-resource languages.

## ATLAS Scaling Law

### Core Formulation

ATLAS replaces the single token count $D$ with an **effective data** term $\mathcal{D}_{\text{eff}}$ that decomposes by language role:

$$\mathcal{L}(N, \mathcal{D}_{\text{eff}}) = E + \frac{A}{N^\alpha} + \frac{B}{\mathcal{D}_{\text{eff}}^\beta}$$

The effective data for a target language $\ell$ is:

$$\mathcal{D}_{\text{eff}}^\ell = D_\ell + \gamma_{\text{transfer}} \cdot D_{\text{transfer}} + \gamma_{\text{other}} \cdot D_{\text{other}}$$

where:
- $D_\ell \in \mathbb{R}^+$ = tokens in the target language $\ell$
- $D_{\text{transfer}}$ = tokens from languages with positive transfer to $\ell$ (as identified by the cross-lingual transfer matrix)
- $D_{\text{other}}$ = tokens from remaining languages
- $\gamma_{\text{transfer}}, \gamma_{\text{other}} \in [0, 1]$ = fitted transfer coefficients

> [!IMPORTANT]
> The key insight is that $\gamma_{\text{transfer}} > \gamma_{\text{other}}$: data from transferable languages contributes more to effective data than data from unrelated languages.

### Multi-Epoch Saturation

To handle repeated data (necessary for low-resource languages), a saturation function is applied:

$$D_{\text{eff}} = D_{\text{unique}} \cdot S(r)$$

where $r = D_{\text{total}} / D_{\text{unique}}$ is the repetition factor, and $S(r)$ is a monotonically increasing concave function capturing diminishing returns from repeated tokens.

> [!NOTE]
> This saturation correction distinguishes ATLAS from prior scaling laws: Chinchilla assumes $r = 1$ (no repetition), which underestimates the cost of training on low-resource languages that require many epochs.

### Algorithm: Fitting ATLAS

**Input:** A set of training experiments $\{(N_i, D_i^\ell, \mathcal{L}_i^\ell)\}$ for each model size $N_i$, language data amount $D_i^\ell$, and measured loss $\mathcal{L}_i^\ell$.

**Steps:**

1. Construct cross-lingual transfer matrix $T \in \mathbb{R}^{L \times L}$ from bilingual experiments (Section: Transfer Matrix)
2. For each target language $\ell$, partition training data into $D_\ell$, $D_{\text{transfer}}$, $D_{\text{other}}$ using $T$
3. Estimate effective data $\mathcal{D}_{\text{eff}}^\ell$ using the saturation function
4. Fit parameters $\{E, A, B, \alpha, \beta, \gamma_{\text{transfer}}, \gamma_{\text{other}}\}$ via least-squares regression on log-transformed losses
5. Validate on held-out experiments with unseen model sizes ($R^2_N$), unseen data scales ($R^2_D$), and unseen language mixtures ($R^2_M$)

**Output:** Scaling law parameters enabling loss prediction $\hat{\mathcal{L}}(N, \mathcal{D}_{\text{eff}})$ for arbitrary configurations.

## Cross-Lingual Transfer Matrix

The transfer matrix $T \in \mathbb{R}^{38 \times 38}$ is constructed from 1,444 bilingual training experiments. Entry $T[\ell_s, \ell_t]$ measures how much including source language $\ell_s$ in training reduces perplexity on target language $\ell_t$, compared to monolingual training on $\ell_t$ alone.

**Key findings from the transfer matrix:**

| Factor | Effect on Transfer |
|---|---|
| Shared script (e.g., Latin, Cyrillic) | Strongest predictor ($p < 0.001$) |
| Shared language family | Significant but weaker than script |
| Geographic proximity | Minor effect |
| English as source | Benefits ~19 of 30 target languages |

**Transfer asymmetry:** $T[\ell_s, \ell_t] \neq T[\ell_t, \ell_s]$ in general. Language A benefiting language B does not imply the reverse—practitioners cannot assume mutual transfer without empirical verification.

> [!TIP]
> The transfer matrix is publicly released and can be used directly to select beneficial language mixtures without re-running experiments.

## Curse of Multilinguality Scaling Law

When increasing the number of training languages $K$, model capacity is diluted. ATLAS models this explicitly:

$$\mathcal{L}(K, N, D_t) = \mathcal{L}_\infty + A \cdot \frac{K^\phi}{N^\alpha} + B \cdot \frac{K^\psi}{D_t^\beta}$$

where:
- $K$ = number of training languages
- $\phi = 0.11$: mild capacity constraint from language expansion
- $\psi = -0.04$: slight positive transfer effect partially offsets capacity loss

**Practical implication:** Expanding from 1 language to 4,000 languages requires approximately:
- $2.74\times$ increase in total training tokens
- $1.4\times$ increase in model size

to maintain equivalent per-language performance.

> [!NOTE]
> The small magnitude of $\psi = -0.04$ (negative sign means adding languages slightly helps data efficiency) quantifies why multilingual models can be surprisingly competitive with monolingual ones when scaled sufficiently.

## Pretrain vs. Fine-Tune Decision

Given a compute budget $C$ (in FLOPs), the decision of whether to pretrain from scratch or fine-tune from a multilingual checkpoint depends on:

$$\log(C) = 10{,}283{,}128 \times N^{1.65}$$

**Decision boundaries:**
- $C < 144\text{B tokens}$: Fine-tuning from a Unimax multilingual checkpoint is more efficient
- $144\text{B} \leq C \leq 283\text{B tokens}$: Transition zone; optimal strategy is language-dependent
- $C > 283\text{B tokens}$: Pretraining from scratch becomes superior

**Intuition:** Fine-tuning leverages multilingual representations already encoded in the checkpoint, but this advantage is eventually outweighed by the flexibility of optimizing the full training distribution from scratch.

## Comparison to Prior Scaling Laws

| Method | Multi-lingual? | Handles Transfer? | Multi-epoch? | $R^2$ (unseen mixtures) |
|---|---|---|---|---|
| Chinchilla (Hoffmann et al., 2022) | No | No | No | 0.69 |
| Muennighoff et al. (2023) | No | No | Yes | — |
| ATLAS (this work) | Yes | Yes | Yes | **0.82** |

> [!IMPORTANT]
> The improvement from 0.69 → 0.82 $R^2$ on unseen language mixtures is the key validation: ATLAS generalizes to mixture configurations not seen during fitting, which is the practical use case for mixture design.

# Experiments

- **Dataset:** MADLAD-400 (Cassano et al., 2023) — 400+ languages, web-crawled multilingual corpus
  - 50 languages selected for diversity in family, script, and resource level
  - Vocabulary-insensitive loss computed to ensure fair cross-lingual comparison (avoids bias from tokenization efficiency differences)
- **Model sizes:** 10M, 30M, 85M, 250M, 750M, 2.5B, 8B parameters
- **Architecture:** Transformer decoder (GPT-style), 64K SentencePiece vocabulary
- **Optimizer:** AdamW, base learning rate $2 \times 10^{-4}$, WSD (Warmup-Stable-Decay) learning rate schedule
- **Training experiments:** 280 monolingual + 240 bilingual + 120 multilingual mixture + 130 fine-tuning = **774 total experiments**
- **Hardware:** Not specified in the paper excerpt
- **Key results:**
  - ATLAS monolingual $R^2 = 0.88$ on unseen model sizes (vs. 0.68 for Chinchilla)
  - ATLAS multilingual $R^2 = 0.82$ on unseen language mixtures (vs. 0.69 for prior methods)
  - Transfer matrix explains ~27% of variance in cross-lingual transfer scores via script and family features alone
