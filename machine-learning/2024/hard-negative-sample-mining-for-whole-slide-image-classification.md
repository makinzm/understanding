# Meta Information

- URL: [Hard Negative Sample Mining for Whole Slide Image Classification](https://arxiv.org/abs/2410.02212)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Huang, W., Hu, X., Abousamra, S., Prasanna, P., & Chen, C. (2024). Hard Negative Sample Mining for Whole Slide Image Classification. arXiv:2410.02212.

# Hard Negative Sample Mining for Whole Slide Image Classification

This paper addresses weakly supervised classification of Whole Slide Images (WSIs) in computational pathology using Multiple Instance Learning (MIL). WSIs are gigapixel images where only slide-level labels (tumor/normal) are available—no patch-level annotations exist. The core contribution is to mine **hard negative patches**: patches from normal (negative) slides that are visually similar to tumor patches and thus most likely to confuse the classifier. By focusing training on these hard cases and introducing a patch-level ranking loss, the method improves classification accuracy while reducing training time by 70–80%.

**Applicability**: Computational pathology researchers and clinical AI engineers building slide-level cancer detection or mutation prediction systems on datasets with bag-level labels only and no patch-level annotation budget.

> [!NOTE]
> "Hard negatives" refer to negative patches (from tumor-free slides) that have high similarity to positive patches in feature space. They are the most informative negatives for training a discriminative classifier, because they push the decision boundary toward harder, more fine-grained distinctions.

## Background: Multiple Instance Learning for WSI Classification

A WSI $X_i$ is treated as a **bag** of patches (instances). Given only a bag-level label $Y_i \in \{0, 1\}$ (normal/tumor), MIL assumes:
- If $Y_i = 1$ (tumor), at least one patch in $X_i$ is positive.
- If $Y_i = 0$ (normal), all patches in $X_i$ are negative.

**Pipeline**:
1. A pre-trained feature extractor encodes each patch $p_{ij}$ into a feature vector $h_{ij} \in \mathbb{R}^d$.
2. An aggregator combines all $\{h_{ij}\}$ into a bag-level prediction $\hat{Y}_i$.

Standard aggregators (ABMIL, DSMIL) rely on attention weights computed globally over all patches, which is computationally expensive for slides with 10k–100k patches and fails to distinguish subtle negative patches from genuinely positive ones.

## Hard Negative Sample Mining

### Instance Bank Construction

During fine-tuning, the method maintains two **instance banks** from the current epoch's predictions:

| Bank | Source slides | Selection criterion | Size |
|------|--------------|---------------------|------|
| Positive bank $\mathcal{B}^+$ | Tumor slides ($Y_i=1$) | Top $r_p = 20\%$ patches by predicted score $\hat{s}$ | — |
| Negative bank $\mathcal{B}^-$ | Normal slides ($Y_i=0$) | Top $r_n = 5\%$ patches by predicted score $\hat{s}$ | — |

Hard negative patches are the top-scored patches from normal slides—patches the current model incorrectly assigns high tumor probability. Using only 5% of negatives dramatically reduces the number of instances without sacrificing training signal.

### Hard Negative Identification via Feature Similarity

For each positive patch $p^+ \in \mathcal{B}^+$, a hard negative $p^-$ is retrieved from $\mathcal{B}^-$ by nearest-neighbor search in feature space:

```math
\begin{align}
  p^-_{\text{hard}} = \arg\min_{p^- \in \mathcal{B}^-} \| h^+ - h^- \|_2
\end{align}
```

where $h^+ \in \mathbb{R}^d$ and $h^- \in \mathbb{R}^d$ are the feature vectors of positive and negative patches respectively. This ensures each training step uses the most challenging negative counterpart for every positive anchor.

### Pseudo-label Assignment

After sampling, each patch in $\mathcal{B}^+$ and $\mathcal{B}^-$ receives a pseudo-label (1 or 0). The feature extractor is then fine-tuned on these pseudo-labeled patches using standard cross-entropy loss $\mathcal{L}_{CE}$:

```math
\begin{align}
  \mathcal{L}_{CE} = -\frac{1}{N}\sum_{j=1}^{N} \left[ y_j \log \hat{s}_j + (1-y_j)\log(1-\hat{s}_j) \right]
\end{align}
```

where $y_j \in \{0,1\}$ is the pseudo-label and $\hat{s}_j \in [0,1]$ is the predicted score for patch $j$.

## Multiple Instance Ranking Loss (MI-Rank)

A new ranking loss enforces that the average score of the top-$K$ positive patches exceeds the average score of the top-$K$ hard negative patches by a margin of 1:

```math
\begin{align}
  \mathcal{L}_{\text{MIRank}} = \max\!\left(0,\; 1 - \frac{1}{K}\sum_{\text{top-}K} \hat{s}^+_j + \frac{1}{K}\sum_{\text{top-}K} \hat{s}^-_j\right)
\end{align}
```

where $\hat{s}^+_j$ and $\hat{s}^-_j$ are the predicted scores of top-$K$ patches from the positive and negative banks, respectively. $K = 10$ in experiments.

> [!NOTE]
> This is a **margin ranking loss** operating at the bag level: it compares aggregate statistics of positive vs. negative patches rather than individual pairs. This makes it compatible with MIL's bag-level supervision while still encouraging patch-level discrimination.

The total fine-tuning loss combines cross-entropy and ranking loss:

```math
\begin{align}
  \mathcal{L} = w_b \cdot \mathcal{L}_{CE} + w_r \cdot \mathcal{L}_{\text{MIRank}}
\end{align}
```

with $w_b = 0.5$ and $w_r = 0.1$.

## Instance Aggregator (Bag-level Prediction)

After fine-tuning, the bag-level prediction $\hat{Y}_i$ is computed by combining instance-level and attention-based bag-level classifiers:

```math
\begin{align}
  \hat{Y}_i = \frac{1}{2}\!\left(\phi_{\text{ins}}\,h_m + \phi_{\text{bag}}\sum_j U(h_j, h_m)\,h_j\right)
\end{align}
```

where:
- $h_m \in \mathbb{R}^d$: feature of the most-suspicious patch (argmax of $\hat{s}_j$ within slide $i$).
- $U(h_j, h_m) \in [0,1]$: attention weight of patch $j$ relative to $h_m$, computed via dot-product attention.
- $\phi_{\text{ins}}, \phi_{\text{bag}}$: linear classifiers for instance-level and bag-level branches.

The aggregator averages both branches, balancing local (single-patch) and global (all-patch) evidence.

## Algorithm: Full Training Procedure

```
Inputs: WSI bags {X_i, Y_i}, pre-trained feature extractor f_θ, MIL aggregator g_φ
Outputs: Fine-tuned f_θ, trained g_φ

Phase 1 — Warm-up MIL Training (350 epochs):
  for each epoch:
    for each bag X_i:
      features = {f_θ(p_ij) for p_ij in X_i}
      Ŷ_i = g_φ(features)
      update g_φ via L_CE(Ŷ_i, Y_i)

Phase 2 — Hard Negative Fine-tuning (25 epochs):
  for each epoch:
    // Build instance banks from current predictions
    B+ = top r_p% patches from positive slides by ŝ_j
    B- = top r_n% patches from negative slides by ŝ_j

    // Mine hard negatives via nearest-neighbor in feature space
    for each p+ in B+:
      p-_hard = argmin_{p- in B-} ||f_θ(p+) - f_θ(p-))||_2

    // Assign pseudo-labels and update feature extractor
    update f_θ via L = w_b * L_CE + w_r * L_MIRank

Phase 3 — Final MIL Inference:
  for each bag X_i:
    features = {f_θ(p_ij) for p_ij in X_i}
    Ŷ_i = aggregator combining instance + bag branches
```

## Comparison with Related Methods

| Method | Strategy | Negative Handling | Fine-tuning |
|--------|----------|------------------|-------------|
| ABMIL | Attention-weighted pooling | All negatives equally | No |
| DSMIL | Dual-stream (max + attention) | All negatives equally | No |
| ItS2CLR | Self-training with contrastive loss | All negatives equally | Yes (instance contrast) |
| **HNM-WSI (ours)** | Hard negative mining + MI-Rank | Top-5% hard negatives only | Yes (pseudo-label + rank loss) |

**Key difference from ItS2CLR**: ItS2CLR uses contrastive learning over all instances from positive slides; HNM-WSI explicitly identifies the hardest negatives from negative slides and uses a ranking loss instead of NT-Xent, which is more directly aligned with the MIL classification objective.

> [!IMPORTANT]
> The efficiency gain (70–80% training time reduction) comes entirely from using $r_n = 5\%$ of negative patches. This is possible because hard negatives concentrate the learning signal: the remaining 95% of negative patches are "easy negatives" that contribute negligible gradient.

## Experiments

- **Datasets**:
  - **Camelyon16**: 399 slides total (270 normal + 129 tumor), lymph node sections, 0.25M patches at 5× magnification. Binary classification (tumor/normal).
  - **TCGA-LUAD Mutation**: 607 lung adenocarcinoma slides for 4 gene mutation prediction tasks (EGFR, KRAS, STK11, TP53), 0.52M patches at 10× magnification. Binary per-gene classification.
- **Hardware**: Not explicitly specified in the paper.
- **Optimizer**: Adam, learning rate $10^{-4}$.
- **Training**: 350 epochs MIL warm-up + 25 epochs fine-tuning.
- **Hyperparameters**: $K=10$, $r_p=0.2$, $r_n=0.05$, $w_b=0.5$, $w_r=0.1$.
- **Results**:
  - Camelyon16: Accuracy 93.02%, AUC 0.9604, outperforming DSMIL and ItS2CLR.
  - TCGA-LUAD: AUC 0.7235 (EGFR), 0.6473 (KRAS), 0.7396 (STK11), 0.7071 (TP53).
  - Training time at $r_n=5\%$: ~39 min/epoch vs. ~240 min/epoch at $r_n=100\%$.

> [!TIP]
> The code is available at: https://github.com/winston52/HNM-WSI
