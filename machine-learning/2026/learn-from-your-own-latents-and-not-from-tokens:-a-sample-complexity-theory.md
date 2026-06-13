# Meta Information

- URL: [Learn from your own latents and not from tokens: A sample-complexity theory](https://arxiv.org/abs/2605.27734)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Korchinski, D. J., Favero, A., & Wyart, M. (2026). Learn from your own latents and not from tokens: A sample-complexity theory. arXiv:2605.27734.

# Learn from your own latents and not from tokens: A sample-complexity theory

## Overview

This paper provides the first sample-complexity theory explaining why self-supervised learning (SSL) methods that predict latent representations of related views or masked regions — rather than raw tokens — achieve dramatically better data efficiency on hierarchical structured data.

**Core finding:** For a probabilistic context-free grammar model (the Random Hierarchy Model), latent prediction achieves sample complexity $O(vm^3)$, which is *constant in tree depth L*, while supervised learning and token-level prediction require $O(vm^{L+1})$ samples — exponential in depth. This exponential separation justifies the practical success of methods like BYOL, DINO, and data2vec.

**Who benefits from this work:**
- Researchers studying the theoretical foundations of self-supervised learning
- Practitioners choosing between masked token prediction and latent representation prediction
- Anyone building hierarchical SSL systems who wants a theoretical justification for design choices

## 2. The Random Hierarchy Model (RHM)

The RHM is a probabilistic context-free grammar with a fixed balanced tree topology. It captures compositional structure analogous to that found in natural language and images.

**Parameters:**
| Symbol | Meaning |
|--------|---------|
| $L$ | Tree depth |
| $s$ | Branching factor (each node has $s$ children) |
| $v_\ell$ | Vocabulary size at level $\ell$ (number of distinct symbols) |
| $m$ | Number of production rules per symbol |

**Data generation (root to leaf):**

1. Sample root symbol $z_0^{(1)} \in [v_0]$ uniformly
2. At each level $\ell$, each symbol $z_\ell^{(i)}$ independently samples one of $m$ production rules and expands into $s$ children at level $\ell+1$
3. Leaf tokens $x \in [v_L]^{s^L}$ are the observable data

The rule assignments are random at initialization and fixed throughout — the grammar is heterogeneous and unambiguous. The total number of observable sequences is $v_0 \cdot m^{L+1}$, making brute-force enumeration infeasible for large $L$.

**Latent variables:** At level $\ell$, each node holds a symbol $z_\ell^{(i)} \in [v_\ell]$. Adjacent leaves $i, i'$ sharing a level-$\ell$ ancestor (but different level-$(\ell-1)$ ancestors) are called *cousins at distance $\ell$*. Leaves sharing a parent are called *siblings*.

**Key quantity — correlation strength:** Two cousins at distance $\ell$ have mutual information scaling as $m^{-\ell}$. Each unresolved latent level costs a factor of $m$ in sample complexity, which drives the exponential gap between token- and latent-level prediction.

## 3. Sample Complexity of Token-Level vs. Latent-Level Prediction

**Token-level prediction (supervised classification / masked token prediction):**

To classify the root from $s^L$ leaf tokens, a learner must disentangle all $L$ levels of latent structure. The required number of training sequences is:

```math
\begin{align}
  P_{\text{token}} = \Omega\!\left(v_m^{(L+1)}\right)
\end{align}
```

where $v = v_0 = v_\ell$ (uniform vocabulary, $m$ rules per symbol). This is exponential in the tree depth $L$.

**Latent-level prediction (ILC / SLC / data2vec):**

By predicting latent clusters of siblings or masked-region encodings instead of raw tokens, the learner only needs to resolve correlations at distance 2 (siblings share a parent), costing $m^2$ per level. Combined with the number of clusters $v$:

```math
\begin{align}
  P_{\text{latent}} = O\!\left(vm^3\right)
\end{align}
```

This is *independent of $L$* (up to logarithmic factors), providing an exponential improvement over token-level approaches.

## 4. Iterative Latent Clustering (ILC) Algorithm

ILC is the paper's explicit hierarchical algorithm that provably achieves $O(vm^3)$ sample complexity.

**Input:** $P$ observed sequences $x^{(1)}, \ldots, x^{(P)} \in [v_L]^{s^L}$

**Output:** Cluster assignments $\hat{z}_\ell^{(i)}$ for each node at each level $\ell = 1, \ldots, L-1$ (non-root hierarchy)

**Algorithm (Iterative Latent Clustering):**

```
for ℓ = 1 to L−1:
  # Build "cousin context vectors" at level ℓ
  for each leaf group g of size s^ℓ (level-ℓ subtree):
    c(g) = empirical distribution over cousin groups at distance ℓ
           (co-occurrence counts of level-(ℓ−1) cluster pairs)
  
  # Cluster by cousin context
  Apply k-means or spectral clustering on {c(g)} with k = v·m clusters
  
  # Assign level-ℓ latents
  ẑ_ℓ^(i) = cluster_id(g_i)
```

**Theorem 1 (informal):** Under balanced and separated grammar assumptions, if:

```math
\begin{align}
  P \geq C \left[ vm \log\!\frac{Lvm}{\delta} + \frac{vm^3}{1 - f} \log\!\frac{Lvm}{\delta} \right]
\end{align}
```

then ILC recovers the non-root hierarchy with probability at least $1 - \delta$. Here $f$ is the maximum fraction of variance explained by confounding directions, and $C$ is a universal constant.

The dominant term $vm^3$ arises because cousin contexts at distance 2 (siblings with shared parent) require $m^2$ sample distinguishability, and there are $vm$ distinct cluster types.

**Comparison to token prediction:** A naive Bayes classifier trained on tokens needs $P = \Omega(vm^{L+1})$ samples to identify root categories — exponentially worse than ILC's $O(vm^3)$ for large $L$.

## 5. Stacked Latent-Clustering (SLC) Network

The SLC network is a neural end-to-end model that matches ILC's theoretical sample complexity through gradient descent.

**Architecture (stack of $L-1$ identical modules):**

```
Input: token sequence x ∈ [v_L]^{s^L}

Module ℓ (for ℓ = 1 to L−1):
  Predictor:
    - Input: level-ℓ cluster codes ẑ_ℓ^(i) ∈ R^d
    - Predict sibling cluster codes of teacher network
    - Loss: cross-entropy on categorical distributions
  
  Clusterer:
    - Input: predictor embeddings
    - Contrastive objective: pull siblings together, push cousins apart
    - Output: discrete cluster codes ẑ_{ℓ+1}^(i) ∈ R^d

Teacher network: exponential moving average (EMA) of student weights
```

**Key design choices:**
- Stop-gradients between modules enable *quasi-local* (biologically plausible) learning rules
- EMA teacher provides stable targets without collapse
- Discrete cluster codes prevent information leakage across levels

**Empirical result:** SLC achieves root classification accuracy that collapses to a single curve when sample count is rescaled by $vm^3$, for all tested depths $L \in \{3, 4, 5, 6, 7\}$. This confirms the $O(vm^3)$ sample complexity holds independent of $L$.

## 6. data2vec: Implicit Hierarchical Latent Prediction

The paper provides the first formal sample-complexity analysis of [data2vec](https://arxiv.org/abs/2202.04803), showing it implicitly performs hierarchical latent prediction.

**data2vec training:** A student encoder predicts teacher-network targets for masked positions. The teacher is a slow EMA of the student.

**Theoretical model (two phases):**

*Phase 0 (early training, token-level):*

The teacher target decomposes as:

```math
\begin{align}
  Y_i(x) = F_i(S) + \sum_a B_a \, e_{z_i^{(a)}} + \text{residual}
\end{align}
```

where $S$ is the visible (unmasked) context, $z_i^{(a)}$ are level-$a$ latents, and $B_a$ are learned projection matrices. In Phase 0, predicting $Y_i$ from context $S$ is essentially token-level prediction — it requires resolving all latent levels.

*Phase $\ell \geq 1$ (after level-$\ell$ latents are learned):*

As training progresses, the teacher's targets begin to linearly encode the level-$\ell$ latents already learned by the student. The masked prediction problem "lifts" to level-$\ell$ tuple clustering — structurally identical to one step of ILC. Each phase costs $O(vm^3)$ samples.

**Result:** data2vec's total sample complexity is $O(L \cdot vm^3)$ — polynomial in $L$ (from the $L$ phases), versus the $\Omega(vm^{L+1})$ requirement of token-level methods. Empirically, root classification curves collapse under $vm^3$ rescaling, not $vm^{L+1}$.

> [!NOTE]
> The paper treats the slow EMA as discretized sequential phases, which is an approximation. The actual data2vec training is continuous and the phase boundaries are not sharp.

## 7. Theoretical Assumptions

| Assumption | Formal Condition | Intuition |
|-----------|-----------------|-----------|
| Balanced grammar | Every grammatical tuple occurs with probability $\approx 1/(vm)$ | Ensures no symbol dominates the data |
| Separated grammar | Parent context vectors maintain minimum distance $\geq c_{\text{sep}} \sqrt{1-f}/m$ | Ensures distinct parents are distinguishable from cousins |
| Stable clustering | Algorithm correctly partitions points within $\varepsilon \leq \Delta/8$ of true centers | Ensures per-level clustering errors don't compound |

The separated grammar assumption is the most restrictive — it requires the grammar to have sufficient diversity in production rules. For random grammars (rules drawn uniformly), this holds with high probability for large enough $v$ and $m$.

## 8. Experiments

- **Dataset:** Synthetic data generated from the Random Hierarchy Model (RHM) with varying parameters $(v, m, L, s)$. No real-world datasets are used.
- **Hardware:** Not specified (primarily theoretical/synthetic experiments).
- **Optimizer:** Not specified for ILC (closed-form clustering); gradient descent for SLC network.
- **Key quantitative results:**
  - ILC clustering accuracy curves collapse to a single curve when sample axis is rescaled by $vm^3$, for all tested $(v, m, L)$ combinations (Figure 3)
  - SLC network root classification curves collapse under $vm^3$ rescaling for $L \in \{3, 4, 5, 6, 7\}$, confirming depth-independence (Figure 3)
  - data2vec synonym clustering scores rise with $vm^3$ scaling (Figure 5), confirming hierarchical latent discovery
  - All curves are incompatible with the token-level baseline's $vm^{L+1}$ scaling

## 9. Implications and Comparison with Related Methods

| Method | Prediction Target | Sample Complexity | Depth Dependence |
|--------|-----------------|------------------|-----------------|
| Supervised (tokens) | Root label | $\Omega(vm^{L+1})$ | Exponential in $L$ |
| Masked token prediction | Raw tokens | $\Omega(vm^{L+1})$ | Exponential in $L$ |
| BYOL / DINO (latent) | View encodings | $O(vm^3)$ (conjectured) | None |
| ILC (proposed) | Sibling latents | $O(vm^3)$ (proved) | None |
| SLC network (proposed) | Sibling latents | $O(vm^3)$ (empirical) | None |
| data2vec | EMA teacher targets | $O(L \cdot vm^3)$ (proved) | Linear in $L$ |
| H-JEPA (hierarchical) | Multi-level latents | $O(vm^3)$ (conjectured) | None |

> [!IMPORTANT]
> The paper argues that H-JEPA's explicit hierarchical stacking is largely redundant: a single latent-prediction module already implicitly learns multi-scale representations, as shown by the data2vec analysis. The exponential improvement comes from predicting *any* latent, not from stacking multiple levels.

**Biological plausibility:** The predictor-clusterer modules in SLC use stop-gradients between levels, making learning quasi-local. This connects to cortical predictive coding theories, where each cortical area predicts the activity of adjacent areas rather than distant sensory inputs.

## 10. Limitations

- The RHM uses a **fixed tree topology**, whereas real language and image structure has variable-length and recursive dependencies.
- **Non-recursive, unambiguous** production rules simplify analysis but exclude phenomena like anaphora in natural language.
- The **context-free** assumption ignores long-range dependencies across branches.
- The analysis of data2vec relies on discretizing the continuous EMA process into phases — a useful but idealized approximation.
- Extensions to these settings (variable topologies, recursive grammars, context-dependent rules) remain open problems.
