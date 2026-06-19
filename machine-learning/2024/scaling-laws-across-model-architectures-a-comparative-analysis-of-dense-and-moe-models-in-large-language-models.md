# Meta Information

- URL: [Scaling Laws Across Model Architectures: A Comparative Analysis of Dense and MoE Models in Large Language Models](https://arxiv.org/abs/2410.05661)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Siqi Wang, Zhengyu Chen, Bei Li, Keqing He, Min Zhang, Jingang Wang (2024). Scaling Laws Across Model Architectures: A Comparative Analysis of Dense and MoE Models in Large Language Models. arXiv:2410.05661.

# Scaling Laws Across Model Architectures: Dense vs. MoE Models

## Background and Motivation

Scaling laws — the empirical observation that model loss decreases as a power-law function of compute, data, and parameter count — were originally established for dense Transformer models (Kaplan et al. 2020; Hoffmann et al. 2022). Mixture of Experts (MoE) architectures activate only a sparse subset of parameters per token, making them parameter-efficient but introducing new degrees of freedom (e.g., number of experts $E$). This paper asks: **do the same power-law principles transfer to MoE models, and if so, how do the optimal hyperparameter strategies differ?**

This work targets LLM practitioners deciding between dense and MoE scaling strategies — particularly those allocating fixed compute budgets across model size, data volume, and expert count.

## Unified Scaling Law Framework

### Loss Scaling Formula

The authors extend the standard Chinchilla-style loss formula to include expert count $E$:

```math
\begin{align}
  \hat{L}(N, D, E) = \frac{A}{N^{\alpha} \cdot E^{\gamma}} + \frac{B}{D^{\beta}} + \sigma
\end{align}
```

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of (non-embedding) parameters |
| $D$ | Number of training tokens |
| $E$ | Number of experts |
| $A, B$ | Fitted constants |
| $\alpha, \beta, \gamma$ | Power-law exponents |
| $\sigma$ | Irreducible loss floor |

For dense models, $E = 1$ reduces this to the standard two-term form. Both architectures exhibit power-law behaviour: as $N$, $D$, or $E$ increases, loss decreases predictably.

### Optimal Compute Allocation

Given a fixed FLOPs budget $C$, the IsoFLOP optimal model size and token count follow:

```math
\begin{align}
  \hat{N}_{\text{opt}}(C) &= k_N \cdot C^{\alpha_N} \\
  \hat{D}_{\text{opt}}(C) &= k_D \cdot C^{\alpha_D}
\end{align}
```

Key finding: MoE models have a larger $\alpha_N$ (0.590) than dense models (0.507), meaning MoE architectures benefit more from scaling parameters than from scaling data. Practitioners training MoE models should therefore bias compute allocation toward larger models rather than more tokens, compared to equivalent dense scaling strategies.

## Hyperparameter Scaling Laws

### Optimal Batch Size

The critical batch size $B_{\text{crit}}$ scales inversely with loss:

```math
\begin{align}
  B_{\text{opt}} \approx \frac{\lambda_B}{L^{\alpha_B}}
\end{align}
```

MoE models exhibit **lower gradient noise scales** than dense models at the same loss level, meaning their stochastic gradients have smaller variance. The practical consequence is that MoE models can be trained stably with **smaller batch sizes**, reducing memory requirements per iteration.

### Optimal Learning Rate

```math
\begin{align}
  \varepsilon_{\text{opt}} \approx \frac{\lambda_\varepsilon}{L^{\alpha_\varepsilon}}
\end{align}
```

MoE models support **larger optimal learning rates** at equivalent loss values. This is consistent with the lower noise scale: when gradient variance is small, larger steps can be taken without destabilising training.

> [!NOTE]
> Both $B_{\text{opt}}$ and $\varepsilon_{\text{opt}}$ are functions of the current loss $L$, so they change during training. The authors recommend using the predicted loss at the end of training to set these hyperparameters at the start.

## Why MoE Models Are More Efficient

The authors attribute MoE data efficiency advantages (~16.37% lower test loss under equal compute) to three mechanisms:

1. **Specialisation**: Each expert sub-network learns to handle a distinct subset of input patterns, increasing effective model capacity without proportional compute.
2. **Ensemble effect**: The gating mechanism aggregates predictions from multiple specialised networks, reducing overfitting similar to ensemble methods.
3. **Implicit regularisation**: Sparse activation and load-balancing auxiliary losses act as regularisers, improving generalisation.

> [!CAUTION]
> The 16.37% figure is an average across the experimental range; the advantage varies by task and model scale. Treat it as a directional claim rather than a precise constant.

## Algorithm: IsoFLOP Profiling Procedure

1. Fix a FLOPs budget $C$ (e.g., $6 \times 10^{21}$).
2. Train multiple models with different $(N, D)$ pairs satisfying $C \approx 6ND$.
3. Record final validation loss for each configuration.
4. Fit a parabola to the IsoFLOP loss curve to identify $N^*$ and $D^*$.
5. Repeat across several values of $C$ to obtain $(C, N^*)$ and $(C, D^*)$ pairs.
6. Fit power laws $N^* = k_N C^{\alpha_N}$ and $D^* = k_D C^{\alpha_D}$.

This procedure is applied separately to dense and MoE model families, yielding architecture-specific exponents.

## Differences from Similar Work

| Aspect | Kaplan et al. (2020) | Hoffmann et al. (2022, Chinchilla) | This Work |
|--------|---------------------|-----------------------------------|-----------|
| Architecture | Dense Transformer | Dense Transformer | Dense + MoE |
| Expert scaling | Not modelled | Not modelled | Explicit $E^{\gamma}$ term |
| Hyperparameter scaling | Not covered | Not covered | $B_{\text{opt}}$, $\varepsilon_{\text{opt}}$ scaling |
| Compute-optimal ratio | $N \propto C^{0.73}$ | $N \approx D$ (50/50 split) | MoE: $\alpha_N = 0.590 > 0.507$ (dense) |

> [!TIP]
> Chinchilla's compute-optimal ratio assumes equal scaling of $N$ and $D$. This paper shows that for MoE models, parameters should be scaled more aggressively relative to data.

## Experiments

- **Dataset**: The Pile — 825 GiB English text corpus comprising 22 subsets (books, web, code, academic, etc.)
- **Hardware**: Not specified in the paper
- **Model scales**:
  - Dense: 50M–7B parameters
  - MoE: 200M–3B active parameters (8 experts per layer)
- **Training tokens**: Up to 100B tokens per run
- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay applied)
- **Learning rate schedule**: Cosine decay with 10× reduction from peak; peak LR ranged from $1.5 \times 10^{-3}$ to $2 \times 10^{-4}$ depending on model size
- **Key quantitative results**:
  - MoE models achieve ~16.37% lower test loss than dense counterparts at equal FLOPs
  - On downstream benchmarks (TriviaQA, MATH, MMLU, CMMLU, MATH401), MoE-3B (active params) outperforms Dense-7B across all tasks
  - MoE-specific exponent $\alpha_N = 0.590$ vs. dense $\alpha_N = 0.507$

## Limitations

- Experiments use at most 8 experts; behaviour with hundreds of experts (as in Switch Transformer or GLaM) is not validated
- Training beyond 100B tokens shows diminishing returns; very long training runs may shift optimal allocation
- Evaluation focuses on English-language benchmarks; multilingual or code-heavy regimes may differ
