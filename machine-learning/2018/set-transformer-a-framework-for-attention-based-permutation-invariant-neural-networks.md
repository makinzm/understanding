# Meta Information

- URL: [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *ICML 2019*, Volume 97, pages 3744–3753.

---

# Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks

## 1. Problem Setting

Many machine learning tasks operate on **set-structured inputs**: multiple instance learning, point cloud classification, few-shot image classification, and clustering all take sets of instances as input. A key requirement is **permutation invariance** — the output must not depend on the ordering of the input elements.

A function $f: 2^{\mathcal{X}} \to \mathcal{Y}$ is **permutation invariant** if for any permutation $\pi$:
$$f(\{x_1, \ldots, x_n\}) = f(\{x_{\pi(1)}, \ldots, x_{\pi(n)}\})$$

Naive application of a standard Transformer (self-attention over all input pairs) is permutation invariant but has **$O(n^2)$ complexity** in both memory and computation, which is prohibitive for large sets.

**DeepSets** (Zaheer et al., 2017) addressed this by decomposing set functions as $f(X) = \rho(\sum_{x \in X} \phi(x))$, which is $O(n)$ but cannot model pairwise interactions between elements.

Set Transformer fills the gap: it models interactions between elements efficiently using attention with **inducing points**, reducing complexity to $O(nm)$ where $m \ll n$.

---

## 2. Attention-Based Building Blocks

All blocks below receive sets as input and produce sets as output, preserving permutation equivariance or invariance as required.

### 2.1 Multihead Attention Block (MAB)

MAB is the core building block. It takes two inputs:
- $X \in \mathbb{R}^{n \times d}$ (queries source)
- $Y \in \mathbb{R}^{m \times d}$ (keys/values source)

and computes:
$$\text{MAB}(X, Y) = \text{LayerNorm}(H + \text{rFF}(H))$$
$$H = \text{LayerNorm}(X + \text{Multihead}(X, Y, Y))$$

where $\text{Multihead}(Q, K, V)$ is multi-head dot-product attention. The residual feed-forward network $\text{rFF}$ applies a two-layer MLP element-wise (each row independently):
$$\text{rFF}(x) = \text{LayerNorm}(x + \text{FC}(\text{ReLU}(\text{FC}(x))))$$

**Input/Output**: $\text{MAB}: \mathbb{R}^{n \times d} \times \mathbb{R}^{m \times d} \to \mathbb{R}^{n \times d}$

MAB is **permutation equivariant** in $X$ when $Y$ is fixed, and **permutation equivariant** in $Y$ only through the attention mechanism.

### 2.2 Set Attention Block (SAB)

SAB applies self-attention over the full input set:
$$\text{SAB}(X) = \text{MAB}(X, X)$$

**Input/Output**: $\text{SAB}: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$

SAB is **permutation equivariant** and models all pairwise interactions, but has **$O(n^2)$ complexity**.

### 2.3 Inducing-point Set Attention Block (ISAB)

ISAB introduces $m$ learnable **inducing points** $I \in \mathbb{R}^{m \times d}$ ($m \ll n$) to reduce complexity:

$$\text{ISAB}_m(X) = \text{MAB}(X, H) \quad \text{where} \quad H = \text{MAB}(I, X)$$

**Calculation order**:
1. $H = \text{MAB}(I, X) \in \mathbb{R}^{m \times d}$ — each inducing point attends over all $n$ input elements: $O(nm)$
2. $\text{ISAB}_m(X) = \text{MAB}(X, H) \in \mathbb{R}^{n \times d}$ — each input element attends over $m$ inducing points: $O(nm)$

**Input/Output**: $\text{ISAB}_m: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$

**Total complexity**: $O(nm)$ instead of $O(n^2)$. The inducing points act as a compressed "summary" of the set, analogous to inducing points in sparse Gaussian processes.

> [!NOTE]
> ISAB is permutation equivariant in $X$ because MAB is permutation equivariant in its first argument. The inducing points $I$ are shared (learned) parameters, not input-dependent.

### 2.4 Pooling by Multihead Attention (PMA)

PMA reduces a variable-size set to a fixed number of outputs for use in the decoder:
$$\text{PMA}_k(Z) = \text{MAB}(S, \text{rFF}(Z))$$

where $S \in \mathbb{R}^{k \times d}$ is a set of $k$ learnable **seed vectors**.

**Input/Output**: $\text{PMA}_k: \mathbb{R}^{n \times d} \to \mathbb{R}^{k \times d}$

When $k=1$, PMA produces a single vector summarizing the entire set. Multiple seeds ($k > 1$) allow the model to produce multiple pooled representations.

> [!NOTE]
> Lemma 1 in the paper proves that the standard mean pooling $\frac{1}{n}\sum_i x_i$ is a special case of dot-product attention with softmax, meaning PMA strictly generalizes simple pooling.

---

## 3. Set Transformer Architecture

The Set Transformer follows an **encoder-decoder** design.

### 3.1 Encoder

The encoder transforms input $X \in \mathbb{R}^{n \times d_{\text{in}}}$ into a representation $Z \in \mathbb{R}^{n \times d}$ using a stack of SAB or ISAB blocks:

$$Z = \text{Encoder}(X) = \text{SAB}(\text{SAB}(X)) \quad \text{(SAB variant)}$$
$$Z = \text{Encoder}(X) = \text{ISAB}_m(\text{ISAB}_m(X)) \quad \text{(ISAB variant, linear complexity)}$$

The encoder is **permutation equivariant**: if the input is permuted, the output is permuted identically.

### 3.2 Decoder

The decoder applies PMA to pool the encoder output, then applies SABs for inter-summary interaction, and finally a feed-forward layer:

$$\hat{y} = \text{Decoder}(Z) = \text{rFF}(\text{SAB}(\text{PMA}_k(Z)))$$

The full model:
$$\hat{y} = \text{rFF}(\text{SAB}(\text{PMA}_k(\text{Encoder}(X))))$$

The decoder is **permutation invariant** in $Z$ because PMA collapses the $n$-dimensional set into $k$ fixed outputs.

### 3.3 Universality

> [!IMPORTANT]
> Theorem 1 (cited in supplementary): Models of the form $\rho_{\text{FF}}(\text{sum}(\rho_{\text{FF}}(\cdot)))$ are universal function approximators for permutation-invariant functions. Set Transformer extends this: Proposition 1 states that the Set Transformer is a universal function approximator in the space of permutation-invariant functions.

### 3.4 Comparison with DeepSets

| Property | DeepSets | Set Transformer (SAB) | Set Transformer (ISAB) |
|---|---|---|---|
| Permutation invariant | Yes | Yes | Yes |
| Models pairwise interactions | No | Yes | Yes (approximate) |
| Complexity per layer | $O(n)$ | $O(n^2)$ | $O(nm)$ |
| Learnable pooling | No (sum/mean) | Yes (PMA) | Yes (PMA) |

DeepSets applies a shared MLP independently to each element and sums; it cannot capture any relationships between elements. SAB explicitly models all pairwise interactions at quadratic cost. ISAB approximates this through $m$ inducing points.

---

## 4. Pseudocode

**Forward pass of ISAB:**
```
Input: X ∈ R^{n×d}, inducing points I ∈ R^{m×d}
# Step 1: compress set through inducing points
H = MAB(I, X)        # H ∈ R^{m×d}
# Step 2: propagate back to input elements
out = MAB(X, H)      # out ∈ R^{n×d}
return out
```

**Forward pass of PMA:**
```
Input: Z ∈ R^{n×d}, seed vectors S ∈ R^{k×d}
Z' = rFF(Z)          # apply feed-forward first
out = MAB(S, Z')     # out ∈ R^{k×d}
return out
```

**Full Set Transformer (ISAB encoder, k=1 PMA):**
```
Input: X ∈ R^{n×d_in}
# Encoder
Z = ISAB_m(ISAB_m(X))          # Z ∈ R^{n×d}
# Decoder
pool = PMA_1(Z)                 # pool ∈ R^{1×d}
out = rFF(SAB(pool))            # out ∈ R^{1×d_out}
return out
```

---

## 5. Experiments

### 5.1 Max Regression

- **Task**: predict the maximum value in a set
- **Dataset**: sets of 1–10 real numbers uniformly sampled from $[0, 100]$
- **Architecture**: SAB + PMA(1), trained with Adam (lr = $10^{-3}$, batch = 128, 20,000 steps)
- **Baseline**: DeepSets and mean pooling
- **Result**: Set Transformer outperforms mean-pooling baselines by learning to attend to the maximum element

### 5.2 Counting Unique Characters

- **Task**: count distinct characters in a set (tests the model's ability to identify uniqueness)
- **Dataset**: sets of 6–10 characters drawn from an alphabet; target is the number of unique characters
- **Loss**: Poisson log-likelihood
- **Architecture**: SAB + PMA(1), Adam (lr = $10^{-4}$, 200,000 steps, batch = 32)
- **Best result**: SAB + PMA(1) achieves 0.6037 accuracy, outperforming DeepSets which cannot reason about element duplicates

### 5.3 Gaussian Mixture Model Clustering (GMMC)

- **Task**: given a set of points from a Gaussian mixture, predict cluster assignments (amortized clustering)
- **Datasets**:
  - 2D synthetic: $n \in [100, 500]$ points from 4 Gaussians
  - Large-scale synthetic: $n \in [1000, 5000]$ points, $k = 6$ clusters
  - CIFAR-100: 512-dim VGG features (pretrained at 68.54% top-1), treated as set inputs
- **Architecture**: ISAB encoder (required for scalability at $n \geq 1000$)
- **Training**: Adam with initial lr $10^{-3}$–$10^{-4}$, 50k steps with decay
- **Result**: Achieves competitive clustering vs. EM and k-means baselines while being amortized (no per-set optimization at test time)

### 5.4 Set Anomaly Detection

- **Task**: detect which image in a set is "anomalous" (different from all others)
- **Dataset**: CelebA face images; meta-learning formulation with 1,000 subsampled datasets (800 train, 200 test)
- **Architecture**: SAB + PMA, Adam (lr = $10^{-4}$)
- **Result**: Set Transformer outperforms DeepSets and attention baselines by modeling within-set interactions

### 5.5 Point Cloud Classification (ModelNet40)

- **Task**: classify 3D shapes from unordered point sets
- **Dataset**: ModelNet40 — 9,843 training and 2,468 test 3D models across 40 categories
- **Point set sizes tested**: 100, 1,000, 5,000 points
- **Architecture**: ISAB encoder (for scalability) + PMA decoder + fully-connected classifier
- **Training**: Adam (lr = $10^{-3}$, decay 0.3 every 20,000 steps)
- **Result**: Competitive with PointNet++ and other state-of-the-art point cloud methods, with the advantage of a fully set-based architecture

---

## 6. Theoretical Properties

### Lemma 1 (Mean as Attention)
The mean operator $\text{mean}(\{x_1,\ldots,x_n\}) = \frac{1}{n}\sum_i x_i$ is a special case of dot-product attention with softmax when the query (seed) vector is the zero vector. This shows PMA strictly generalizes mean pooling.

### Lemma 3 (PMA expresses sum pooling)
A PMA block with sufficient seeds can express sum pooling $\sum_i z_i$, demonstrating its expressiveness for aggregation.

### Proposition 1 (Universal Approximation)
The Set Transformer is a universal function approximator in the space of permutation-invariant continuous functions from finite sets to $\mathbb{R}^d$.

---

## 7. Applicability

**Who**: Researchers and practitioners working with set-structured data (e.g., point clouds, multi-instance learning, meta-learning, clustering).

**When**: Use ISAB when the set size $n$ is large (thousands of elements) and $O(n^2)$ attention is impractical. Use SAB when $n$ is small and full pairwise interactions are affordable.

**Where**: Any task where the input is an unordered collection of feature vectors and the output is permutation invariant (classification, regression, clustering, detection).

> [!TIP]
> The official implementation is available at [https://github.com/juho-lee/set_transformer](https://github.com/juho-lee/set_transformer). The ISAB approach is also conceptually related to cross-attention in Perceiver (Jaegle et al., 2021), which independently uses a fixed set of latent vectors to compress large inputs.

---

## 8. Conclusion

Set Transformer proposes a modular, attention-based framework for learning permutation-invariant functions over sets. The key contributions are:

1. **MAB/SAB/ISAB**: Reusable building blocks for set-to-set transformations with explicit pairwise interaction modeling
2. **ISAB with inducing points**: Reduces attention complexity from $O(n^2)$ to $O(nm)$ while preserving expressiveness
3. **PMA**: A learnable, attention-based pooling mechanism that generalizes mean pooling
4. **Universality**: The full architecture is a universal approximator for permutation-invariant functions

Compared to DeepSets, Set Transformer adds the ability to model element interactions; compared to vanilla Transformers, ISAB makes this tractable for large sets.
