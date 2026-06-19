# Meta Information

- URL: [InfoNCE: Identifying the Gap Between Theory and Practice](https://arxiv.org/abs/2407.00143)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Rusak, E., Reizinger, P., Juhos, A., Bringmann, O., Zimmermann, R. S., & Brendel, W. (2024). InfoNCE: Identifying the Gap Between Theory and Practice. arXiv:2407.00143.

# InfoNCE: Identifying the Gap Between Theory and Practice

This paper analyzes the gap between theoretical identifiability guarantees in contrastive learning and the behavior observed in practice. The central claim is that prior theory (Zimmermann et al., 2021; von Kügelgen et al., 2021) assumes either isotropic or binary (invariant/variant) changes across latent factors in positive pairs, but practical augmentation pipelines produce a continuum of anisotropic changes. The paper introduces **AnInfoNCE**, a generalized contrastive loss achieving identifiability under this realistic anisotropic setting.

## Background: Contrastive Learning and InfoNCE

Self-supervised contrastive learning trains an encoder $f: \mathcal{X} \to \mathcal{Z}$ without labels by pulling together representations of positive pairs (two augmented views of the same image) and pushing apart negative pairs (views from different images). The standard **InfoNCE loss** is:

```math
\begin{align}
  \mathcal{L}_\text{INCE}(f) = \mathbb{E}\!\left[
    -\ln \frac{
      \exp\!\left(f(x)^\top f(x^+) / \tau\right)
    }{
      \exp\!\left(f(x)^\top f(x^+) / \tau\right)
      + \sum_{i=1}^{M} \exp\!\left(f(x)^\top f(x_i^-) / \tau\right)
    }
  \right]
\end{align}
```

- $x \in \mathcal{X}$: anchor observation
- $x^+ \in \mathcal{X}$: positive sample (augmentation of $x$)
- $x_i^- \in \mathcal{X}$: $M$ negative samples drawn i.i.d. from $p(x)$
- $\tau > 0$: temperature scalar
- Expectation is over positive pairs and negative samples

The encoder is trained end-to-end with InfoNCE; the inner product $f(x)^\top f(x^+)$ measures alignment in the learned representation space.

## Data Generating Process (DGP)

The theoretical analysis is grounded in a latent variable model. Observations are generated as:

```math
\begin{align}
  x = g(z)
\end{align}
```

where $z \in \mathcal{Z} \subseteq \mathbb{R}^d$ is the latent vector and $g: \mathcal{Z} \to \mathcal{X}$ is an invertible generative function. The training data is sampled as:

1. Draw an anchor $z \sim p(z)$
2. Draw a positive $z^+ \sim p(z^+ | z)$
3. Observe $x = g(z)$ and $x^+ = g(z^+)$

**Identifiability** is the goal: the composition $h = f \circ g$ should be a trivial transformation (e.g., permutation or componentwise scaling), meaning the encoder $f$ recovers the true latent factors up to simple ambiguities.

## Prior Identifiability Results and Their Assumptions

### Isotropic Setting (Zimmermann et al., 2021)

Zimmermann et al. proved identifiability of InfoNCE under the assumption that all latent dimensions change uniformly in positive pairs. That is, the positive conditional $p(z^+ | z)$ is isotropic — every latent factor $z_j$ varies to the same degree. This is violated in practice because different augmentations affect different semantic attributes to very different extents.

### Content-Style Partitioning (von Kügelgen et al., 2021)

Von Kügelgen et al. relaxed this by partitioning latent space into **content** $z_c$ (invariant across positive pairs) and **style** $z_s$ (variant):

```math
\begin{align}
  p(z_c^+ | z_c) &= \delta(z_c^+ - z_c) \\
  p(z_s^+ | z_s) &\text{ is non-degenerate}
\end{align}
```

This yields **block-identifiability** (content recovered up to affine transformation) but assumes a hard binary split rather than a continuum of variability. In practice, strong crops change some style factors far more than others, and even nominally "invariant" content factors can shift.

## The Anisotropic Setting

The key insight of this paper is that real augmentation pipelines produce **anisotropic changes**: each latent dimension $z_j$ changes by a different amount across positive pairs. Concretely, the authors model the positive conditional as a Gaussian with dimension-specific parameters:

```math
\begin{align}
  p(z_j^+ | z_j) = \mathcal{N}(\alpha_j z_j,\; \sigma_j^2)
\end{align}
```

- $\alpha_j \in [0, 1]$: retention coefficient for dimension $j$; $\alpha_j = 1$ means the factor is nearly preserved, $\alpha_j \approx 0$ means it changes drastically
- $\sigma_j^2 > 0$: noise variance for dimension $j$

This models a continuum between fully invariant ($\alpha_j = 1, \sigma_j \to 0$, the content regime) and highly variable ($\alpha_j$ small, $\sigma_j$ large). For example:
- **Strong random crop**: drastically alters spatial location factors ($\alpha_j \approx 0$)
- **Gaussian blur**: moderately reduces sharpness-related factors
- **Class label**: largely preserved across crops ($\alpha_j \approx 1$)

When $\sigma_j^2$ varies widely across dimensions, the standard InfoNCE objective loses information in high-variance dimensions — a phenomenon the authors call **information collapse** — because the loss treats all dimensions equally through the shared temperature $\tau$.

## AnInfoNCE: Generalized Identifiable Contrastive Loss

To address the anisotropic setting, the paper introduces **AnInfoNCE**, which generalizes the temperature parameter to be dimension-specific. Instead of a single scalar $\tau$, AnInfoNCE applies per-dimension inverse-variance weighting so that noisy (high-$\sigma_j$) dimensions are down-weighted and stable (low-$\sigma_j$) dimensions are up-weighted:

```math
\begin{align}
  \mathcal{L}_\text{AnInfoNCE}(f) = \mathbb{E}\!\left[
    -\ln \frac{
      \exp\!\left(\sum_j w_j f_j(x) f_j(x^+)\right)
    }{
      \exp\!\left(\sum_j w_j f_j(x) f_j(x^+)\right)
      + \sum_{i=1}^{M} \exp\!\left(\sum_j w_j f_j(x) f_j(x_i^-)\right)
    }
  \right]
\end{align}
```

- $f_j(x)$: the $j$-th component of the encoder output $f(x) \in \mathbb{R}^d$
- $w_j \propto 1/\sigma_j^2$: weight inversely proportional to the positive pair variance of dimension $j$ (higher weight for more stable factors)
- When all $w_j = 1/\tau$ are equal, AnInfoNCE reduces to standard InfoNCE

The dimension-specific weighting $w_j$ ensures that each latent dimension contributes to the loss in proportion to its actual signal-to-noise ratio in the positive pairs, recovering the identifiability property even when factor variabilities are heterogeneous.

**Identifiability theorem (informal)**: Under the anisotropic Gaussian DGP above, minimizing $\mathcal{L}_\text{AnInfoNCE}$ with the appropriate weights $w_j$ recovers the ground-truth latent factors $z$ up to a trivial transformation (permutation and componentwise scaling).

> [!NOTE]
> The standard InfoNCE with uniform temperature fails to identify latent factors when factor variabilities differ significantly, because high-$\sigma_j$ dimensions dominate the loss and their representations collapse — the encoder stops encoding that information.

## Extensions

### Hard Negative Mining

In practice, training is accelerated by selecting informative (hard) negatives rather than sampling uniformly. The paper extends AnInfoNCE to provably handle hard negative mining, where negatives are drawn from a non-uniform distribution $q(x^-)$ that concentrates on samples close to the anchor. The key result is that identifiability is maintained as long as the negative distribution satisfies certain coverage conditions.

### Loss Ensembles

The theory is also extended to **loss ensembles** that combine multiple contrastive objectives (e.g., objectives operating at different augmentation strengths or scales). Combining AnInfoNCE terms with different weight profiles $\{w_j^{(k)}\}$ across ensemble members can further improve factor recovery.

## Theory-Practice Gap Analysis

Beyond the anisotropy contribution, the paper systematically catalogs remaining gaps between identifiability theory and practice:

| Gap | Theory Assumption | Practice Reality |
|---|---|---|
| Augmentation model | Simple parametric $p(z^+\|z)$ | Complex pipelines (crop + jitter + blur + ...) |
| Latent dimensionality | Known $d$ | Unknown; encoders are over/under-parameterized |
| Sample complexity | Asymptotic (infinite data) | Finite batches with $M \ll \infty$ negatives |
| Optimization | Global minimum of $\mathcal{L}$ | SGD converges to local minima |
| Architecture | Arbitrary invertible $f$ | Fixed ResNet/ViT with bottleneck projector |

The paper argues that AnInfoNCE addresses the first gap (augmentation anisotropy) but the other gaps remain open research questions.

## Comparison with Related Methods

| Method | Positive Pair Assumption | Identifiability |
|---|---|---|
| InfoNCE (van den Oord, 2019) | Not specified | No formal guarantee |
| Zimmermann et al. (2021) | Isotropic variance | Yes (full) |
| von Kügelgen et al. (2021) | Binary content/style split | Yes (block) |
| **AnInfoNCE (this work)** | Anisotropic continuum | Yes (full, under anisotropy) |

> [!TIP]
> For background on nonlinear ICA and identifiability in self-supervised learning, see Hyvarinen & Morioka (2016) on temporal contrastive learning and Khemakhem et al. (2020) on iVAE.

# Experiments

- **Datasets**: Synthetic latent variable models (ground-truth known), VAE-MNIST, CIFAR-10, ImageNet
- **Metrics**: Mean Correlation Coefficient (MCC) for latent recovery quality; linear probe accuracy for downstream task performance
- **Baselines**: Standard InfoNCE, SimCLR
- **Key findings**:
  - On synthetic benchmarks with known generative models, AnInfoNCE achieves higher MCC than standard InfoNCE, confirming the theoretical identifiability improvement
  - On CIFAR-10 and ImageNet, AnInfoNCE recovers latent factors that had collapsed under standard InfoNCE
  - **Trade-off**: Improved latent recovery comes at the cost of downstream linear probe accuracy, because standard augmentations are empirically optimized for discriminative performance rather than faithful latent recovery
  - The accuracy-identifiability trade-off reflects that aggressive augmentations which discard some visual information are beneficial for learning invariant features useful for classification, but are incompatible with recovering those discarded factors

> [!IMPORTANT]
> The trade-off finding implies that practitioners face a choice: (1) standard InfoNCE with aggressive augmentations maximizes downstream accuracy but collapses information about some latent factors; (2) AnInfoNCE with dimension-aware weighting recovers more complete latent structure at the cost of discriminative performance. The right choice depends on whether the goal is classification or disentangled representation learning.
