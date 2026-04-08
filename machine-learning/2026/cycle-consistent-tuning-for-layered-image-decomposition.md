# Meta Information

- URL: [Cycle-Consistent Tuning for Layered Image Decomposition](https://arxiv.org/abs/2602.20989)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Gu, Z., Lu, M., Sun, Z., Lischinski, D., Cohen-Or, D., & Huang, H. (2026). Cycle-Consistent Tuning for Layered Image Decomposition. arXiv:2602.20989.

# Cycle-Consistent Tuning for Layered Image Decomposition

## Overview

This paper proposes an in-context image decomposition framework that separates visual layers in a composite image (e.g., a logo overlaid on an object surface) into their constituent layers. The key innovation is a **cycle-consistent tuning strategy** that jointly trains a decomposition model and a composition model under mutual supervision, without requiring large amounts of densely annotated ground-truth triplets.

**Target users**: Computer vision researchers and practitioners working on image editing, asset extraction, or intrinsic decomposition tasks.

**Applicability conditions**: Works on real-world composite images containing two visually distinct overlaid layers (e.g., logo on object, albedo/shading, foreground/background). Currently limited to two-layer decomposition.

## Background: Visual In-Context Learning with Diffusion Models

Prior work on visual in-context learning arranges input-output pairs as image grids and conditions a generative model to complete a new query in the same style. This paper builds on FLUX.1-Fill-dev, a Diffusion Transformer for image inpainting, and extends it to learn decomposition by formulating the task as in-context inpainting.

The FLUX.1-Fill-dev forward pass uses flow-matching (Eq. 1):

```math
\begin{align}
x_{t-1} = \phi\!\left(v_\theta\!\left([x_t,\, \varepsilon_{\text{img}}(X),\, M],\, t,\, \varepsilon_{\text{txt}}(T)\right)\right)
\end{align}
```

where $v_\theta$ is the Diffusion Transformer, $\varepsilon_{\text{img}}$ and $\varepsilon_{\text{txt}}$ are image and text encoders, $M \in \{0,1\}^{H \times W}$ is a binary inpainting mask (0 = preserve, 1 = generate), and $\phi$ is the flow-matching scheduler. The masked input to the model is $M \odot X$.

## Method

### 1. Visual In-Context Format (1×3 Grid)

Each training sample is a triplet $\langle I, A, B \rangle$ arranged as a three-panel horizontal grid:

| Left panel | Middle panel | Right panel |
|---|---|---|
| Composite image $I$ (logo on object) | Isolated logo $A$ | Clean object $B$ (logo removed) |

**Decomposition mask** $M_D$: the left cell is visible context; middle and right cells are masked — the model must generate $A$ and $B$ from $I$.

**Composition mask** $M_C$: middle and right cells are visible context; the left cell is masked — the model must reconstruct $I$ from $\langle A, B \rangle$. This is the complementary task used to enforce cycle consistency.

### 2. Lightweight LoRA Fine-Tuning

The base FLUX.1-Fill-dev weights $W \in \mathbb{R}^{d \times k}$ are frozen. Only LoRA adapters are trained (Eq. 2):

```math
\begin{align}
W' = W + UV
\end{align}
```

where $U \in \mathbb{R}^{d \times r}$ is the down-projection and $V \in \mathbb{R}^{r \times k}$ is the up-projection, with rank $r = 32$ and alpha $= 32$.

> [!IMPORTANT]
> The decomposition model $\mathcal{F}_D$ and composition model $\mathcal{F}_C$ share the **same single LoRA parameter space** $\theta = (U, V)$. They are differentiated only by their input mask $M$ and text prompt embedding $\tau$.

### 3. Dual-Function Framework

The shared model operates in two modes (Eq. 4):

```math
\begin{align}
\mathcal{F}_D(I) &= \langle A, B \rangle \quad \text{(decompose composite into layers)} \\
\mathcal{F}_C(\langle A, B \rangle) &= I \quad \text{(compose layers back into composite)}
\end{align}
```

Two cycle tracks enforce mutual consistency:

**Track 1 — Decompose then Compose:**

```math
\begin{align}
\langle A', B' \rangle = \mathcal{F}_D(I) \quad \rightarrow \quad I' = \mathcal{F}_C(\langle A', B' \rangle)
\end{align}
```

**Track 2 — Compose then Decompose:**

```math
\begin{align}
I^* = \mathcal{F}_C(\langle A, B \rangle) \quad \rightarrow \quad \langle A^*, B^* \rangle = \mathcal{F}_D(I^*)
\end{align}
```

### 4. Loss Functions

**Reconstruction loss** (Eq. 3): standard flow-matching mean-squared error between the predicted velocity $v_\theta$ and the target velocity $\partial x_t / \partial x = x_1 - x_0$, where $x_t = (1-t) \cdot x_0 + t \cdot x_1$ is the noisy latent at time $t$:

```math
\begin{align}
\mathcal{L}_{\text{rec}} = \mathbb{E}_{x,t}\!\left[\|v_\theta(x_t, M, t, \tau) - (x_1 - x_0)\|_2^2\right]
\end{align}
```

**Cycle-consistency loss** (Eq. 5): aligns velocity predictions across the two cycle tracks at independently sampled timesteps $t_1$ and $t_2$:

```math
\begin{align}
\mathcal{L}_{\text{cyc}} &= \mathbb{E}_{x,t_1}\!\left[\|v_\theta(x_{t_1}^I, M_D, t_1, \tau_D) - v_\theta(x_{t_1}^{I^*}, M_D, t_1, \tau_D)\|_2^2\right] \\
&+ \mathbb{E}_{x,t_2}\!\left[\|v_\theta(x_{t_2}^{\langle A,B\rangle}, M_C, t_2, \tau_C) - v_\theta(x_{t_2}^{\langle A',B'\rangle}, M_C, t_2, \tau_C)\|_2^2\right]
\end{align}
```

- The first term penalizes discrepancy between velocity predictions for the original composite $I$ and the recomposed $I^*$.
- The second term penalizes discrepancy between velocity predictions for the original layers $\langle A,B \rangle$ and the decomposed-then-recomposed $\langle A', B' \rangle$.

**Total loss:**

```math
\begin{align}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{cyc}}
\end{align}
```

### 5. Clean Latent Recovery for Cross-Track Passing

To pass the output of one track as input to the next without running full denoising inference, the paper estimates the clean latent directly from the velocity prediction (Eq. 6), derived by rearranging the linear mixing $x_t = (1-t) x_0 + t x_1$:

```math
\begin{align}
\bar{x}^0 \approx x_t - t \cdot v_\theta\!\left([x_t,\, \varepsilon_{\text{img}}(X),\, M_d],\, t,\, \varepsilon_{\text{txt}}(T_d)\right)
\end{align}
```

This avoids running 50-step denoising during training and enables efficient gradient flow across the cycle.

### 6. Algorithm: Cycle-Consistent Training Step

```
function get_pred(X, M, t, τ):
    x0 ← encode(X)
    x1 ← sample_noise()
    x_t ← (1 - t) * x0 + t * x1
    c ← [ε_img(M ⊙ X), M]          # masked context
    pred ← transformer(x_t, c, t, τ)
    tgt ← x1 - x0                    # flow-matching target
    x̄0 ← x_t - t * pred              # approximate clean latent
    return pred, tgt, x̄0, x1

function cycle_step(I, A, B, τ_D, τ_C):
    t1, t2 ← sample independently from sigmoid-transformed Normal

    # Track 1: Decompose then Compose
    p_d, g_d, Ā', B̄' ← get_pred(I, M_D, t1, τ_D)
    p_c, g_c, _       ← get_pred(⟨Ā', B̄'⟩, M_C, t2, τ_C)

    # Track 2: Compose then Decompose
    p̃_c, g̃_c, Ī*      ← get_pred(⟨A, B⟩, M_C, t2, τ_C)
    p̃_d, g̃_d, _       ← get_pred(Ī*, M_D, t1, τ_D)

    L_rec ← mse(p_d, g_d) + mse(p_c, g_c)
    L_cyc ← mse(p_d, p̃_d) + mse(p_c, p̃_c)
    return L_rec + L_cyc
```

## Iterative Self-Improving Training Pipeline

The training data is expanded iteratively from a tiny seed set using the model's own outputs, filtered by a VLM judge.

| Stage | Description | Steps | Samples |
|---|---|---|---|
| Seed | 100 manually curated triplets (GPT-4o assisted) | — | 100 |
| IC-LoRA (~5 rounds) | Train IC-LoRA on seed; generate ~1K/round; filter with Qwen-VL | 4K/round, lr=1 | ~5K total |
| Cycle-consistent (~5 rounds) | Full cycle-consistent training; add high-quality new samples with 2× weight | 5K–10K/round, lr: 1→0.5 | ~5K additional |
| **Final corpus** | | | **~10K triplets** |

> [!NOTE]
> Qwen-VL serves as the filter, checking visual plausibility and decomposition consistency of generated samples before they are added to training.

## Comparison with Similar Methods

| Method | Approach | Key Difference |
|---|---|---|
| ICEdit | General image editing with in-context prompting | Not trained for decomposition; produces layer leakage |
| Flux-Kontext | Diffusion-based context conditioning | No cycle consistency; no explicit layer separation training |
| Gemini | Multimodal LLM for image understanding | Black-box API; no fine-tuning for this task |
| AssetDropper | Specialized logo asset extraction | Task-specific priors; no cycle consistency; no object reconstruction |
| **Ours** | Cycle-consistent LoRA on FLUX.1-Fill-dev | Joint decomposition+composition training; generalizes without task-specific priors |

> [!IMPORTANT]
> Unlike AssetDropper which focuses only on extracting the logo layer, this method simultaneously outputs both the isolated logo $A$ and the cleaned object surface $B$, maintaining full layer fidelity.

## Generalization to Other Decomposition Tasks

The identical cycle-consistent LoRA framework (no architectural changes) is applied to:

1. **Intrinsic image decomposition** (albedo vs. shading) — trained on the Hypersim dataset, evaluated on MAW dataset.
2. **Foreground-background separation** — trained on ~5K triplets generated via synthetic composite construction.

This demonstrates that cycle-consistent tuning is a general strategy for any two-layer visual decomposition task, provided appropriate triplet data $\langle I, A, B \rangle$ can be constructed or collected.

# Experiments

- **Datasets**:
  - Logo-object decomposition: 100 seed triplets (manually curated), ~10K final training triplets (self-generated), 1,500 synthetic test triplets
  - Intrinsic decomposition: Hypersim (training), MAW (evaluation)
  - Foreground-background separation: ~5K synthetic triplets
- **Hardware**: NVIDIA L40 GPU
- **Inference**: 50 denoising steps, ~35 seconds per image
- **Evaluation metrics**: VQAScore (logo and object), VLMScore (Qwen-VL / GPT-4o / Gemini as judges), user study (20 questions, 30 participants)
- **Key results**:
  - Achieves highest logo VQAScore (0.43) and highest average VLMScore (4.22) among all baselines (ICEdit, Flux-Kontext, Gemini, AssetDropper)
  - Ranked top-1 in over 50% of user study comparisons
  - Ablation confirms cycle-consistency loss provides the largest single improvement in layer fidelity
