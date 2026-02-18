# Meta Information

- URL: [Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](https://arxiv.org/abs/2602.15659)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Luankang Zhang, Hao Wang, Zhongzhou Liu, Mingjia Yin, Yonghao Huang, Jiaqi Li, Wei Guo, Yong Liu, Huifeng Guo, Defu Lian, and Enhong Chen (2026). Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control. arXiv:2602.15659.

---

# Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control

## Overview

RSIR (Recursive Self-Improving Recommendation) is a framework that enables recommendation models to autonomously improve through iterative self-generated data augmentation, without relying on external teacher models or manually annotated data. The core insight is that a trained model can generate synthetic user interaction sequences, quality-filter them using a fidelity control mechanism, and retrain on the enriched dataset — repeating this cycle across multiple iterations for cumulative gains.

**Applicability**: RSIR is designed for practitioners building sequential recommendation systems where user interaction data is extremely sparse (e-commerce, content platforms). It is backbone-agnostic, demonstrated across Transformer-based, contrastive learning, and generative architectures.

---

## 1. Problem Formulation

**Input**: A dataset $D_0 = \{s_u\}_{u \in \mathcal{U}}$ where each user sequence $s_u = (i_1, i_2, \ldots, i_T)$ is an ordered list of items the user interacted with chronologically.

**Output**: Model parameters $\theta_K$ after $K$ self-improving iterations, producing improved next-item predictions.

**Training objective** (standard sequential recommendation):

$$\max_\theta \sum_{u \in \mathcal{U}} \sum_{t=2}^{|s_u|} \log P(i_t \mid s_{u,<t}; \theta)$$

**Core challenge**: User-item interaction matrices are extremely sparse — users interact with only a tiny fraction of available items $|\mathcal{V}|$. This creates rugged optimization landscapes that cause poor generalization.

> [!NOTE]
> "Extreme sparsity in user interactions leads to rugged optimization landscapes and poor generalization." — paper abstract

---

## 2. RSIR Framework

The framework operates as a four-phase closed loop over $K$ iterations:

1. **Model Training**: Train $\theta_k$ on current dataset $D_k$
2. **Synthetic Generation**: Generate synthetic sequences $D'_{k+1}$ using $\theta_k$
3. **Fidelity Filtering**: Accept only sequences passing quality control
4. **Dataset Expansion**: $D_{k+1} = D_k \cup D'_{k+1}$, then repeat

### 2.1 Bounded Exploration via Hybrid Candidate Pool

To generate the next item in a synthetic sequence, RSIR samples from a hybrid candidate pool $\mathcal{C}_t$ that mixes:

$$\mathcal{C}_t \sim p \cdot \text{Sample}(s_u) + (1-p) \cdot \text{Sample}(\mathcal{I})$$

where:
- $p \in [0,1]$: probability of sampling from the user's historical items (exploitation)
- $(1-p)$: probability of sampling from all items $\mathcal{I}$ (exploration)
- Top-$k$ scoring is applied over $\mathcal{C}_t$ to select the generated item

This prevents degenerate sequences (random noise) while allowing novel pattern discovery beyond the user's history.

### 2.2 Fidelity-Based Quality Control

A generated sequence $s'_{x,t}$ (prefix of length $t$ extended from user $x$) is accepted only if at least one held-out item from the original sequence ranks highly under the current model:

$$\exists i_j \in S_{\text{tmt}} \text{ such that } \text{Rank}_{f_{\theta_k}}(i_j \mid s'_{x,t}) \leq \tau$$

where:
- $S_{\text{tmt}}$: set of held-out (validation) items from the original user sequence
- $\tau$: rank threshold hyperparameter (smaller = stricter)
- $f_{\theta_k}(\cdot \mid s')$: scoring function of the current model

**Pseudocode for RSIR generation**:

```
Input: D_k, model θ_k, threshold τ, prob p, attempts m
Output: D'_{k+1}

for each user u in U:
    s_u ← sequence from D_k
    s' ← s_u  # start from original prefix
    for attempt in 1..m:
        C_t ← p·Sample(s_u) + (1-p)·Sample(I)
        i_gen ← top-k sample from f_{θ_k}(·|s', C_t)
        s' ← s' + [i_gen]
        if Rank_{f_{θ_k}}(i_held_out | s') ≤ τ:
            accept s', add to D'_{k+1}
            break
    # if no accepted sequence after m attempts, skip user
return D'_{k+1}
```

### 2.3 Computational Complexity

Per-iteration complexity:

$$\mathcal{T}^{(k)} = O\!\left(N_k \left[ E \cdot \mathcal{C}_{\text{model}} + m\!\left(L_e^2 d + L_e \cdot \mathcal{C}_{\text{score}}(|\mathcal{V}|)\right) \right]\right)$$

where:
- $N_k$: number of sequences in $D_k$
- $E$: training epochs
- $\mathcal{C}_{\text{model}}$: per-sequence model forward cost
- $m$: generation attempts per user
- $L_e$: effective sequence length (often shorter than max due to early termination)
- $d$: embedding dimension, $|\mathcal{V}|$: vocabulary size

Generation cost is bounded because fidelity control triggers early termination, keeping $L_e < L_{\max}$ in practice.

---

## 3. Theoretical Analysis

### 3.1 Implicit Regularization View

The loss on generated sequences is:

$$\mathcal{L}_{\text{gen}}(\theta) = \frac{1}{|D'_{k+1}|} \sum_{s' \in D'_{k+1}} \ell(f_\theta(s'))$$

Training on $D_{k+1} = D_k \cup D'_{k+1}$ is equivalent to:

$$\theta_{k+1} = \arg\min_\theta \left[ \mathcal{L}_k(\theta) + \lambda \cdot \mathcal{L}_{\text{gen}}(\theta) \right]$$

This defines an implicit regularizer $\Omega(\theta; \theta_k)$ that penalizes prediction sharpness. Specifically, RSIR applies a **Manifold Tangential Gradient Penalty**:

$$\Omega(\theta) \propto \|\nabla_\mathcal{M} f_\theta\|^2$$

Unlike isotropic smoothing (e.g., Dropout, weight decay), this penalty targets directions along the user preference manifold $\mathcal{M}$, preserving useful discriminative sharpness while smoothing spurious local optima caused by data sparsity.

### 3.2 Error Bound and Breakdown Point

The generalization error after iteration $k+1$ satisfies:

$$\mathcal{E}(\theta_{k+1}) \leq (1-\lambda)\mathcal{E}_0 + \lambda\left[(1-\tilde{p}_k)\rho \cdot \mathcal{E}(\theta_k) + \tilde{p}_k \cdot \mathcal{E}_{\max}\right]$$

where:
- $\tilde{p}_k$: fidelity leakage rate (fraction of accepted synthetic data that are still noisy)
- $\rho < 1$: contraction coefficient
- $\mathcal{E}_{\max}$: worst-case error from purely noisy data

A **breakdown point** exists at $\tilde{p}_k^* = \frac{(1-\rho)\mathcal{E}_0}{(\mathcal{E}_{\max} - \rho\mathcal{E}(\theta_k))}$: if fidelity leakage exceeds this threshold, recursive improvement diverges rather than converges. This explains why performance saturates or degrades in later iterations without strict fidelity control.

---

## 4. Comparison with Related Methods

| Aspect | RSIR | Traditional Data Augmentation | Distillation / Teacher-Student | Contrastive Augmentation (e.g., CL4SRec) |
|--------|------|-------------------------------|-------------------------------|------------------------------------------|
| Requires external model | No | No | Yes (teacher model) | No |
| Quality control | Fidelity filtering (rank-based) | None or heuristic | Teacher soft labels | Augmentation strategies (crop/reorder) |
| Iterative refinement | Yes ($K$ iterations) | No | No | No |
| Theoretical guarantee | Implicit regularization + error bound | None | None | None |
| Data density increase | High (+342% over 3 iterations) | Low | N/A | Moderate |
| Backbone agnostic | Yes | Yes | Partially | No (contrastive-specific) |

> [!IMPORTANT]
> RSIR differs from knowledge distillation in that the "teacher" and "student" are the same model at different iterations, eliminating the need for a separately trained, stronger teacher.

---

## 5. Experiments

- **Datasets**: Amazon Reviews (Beauty, Sports, Toys) and Yelp — all characterized by extreme interaction sparsity; evaluation uses leave-one-out strategy (last item for test, second-to-last for validation)
- **Hardware**: Single GPU (specific model not stated)
- **Optimizer**: Not explicitly specified; training uses RecStudio framework, max 1000 epochs, early stopping with patience 20
- **Backbone models**: SASRec (Transformer-based self-attention), CL4SRec (contrastive learning), HSTU (generative model)
- **Metrics**: NDCG@10, NDCG@20, Recall@10, Recall@20
- **Hyperparameters**: $\tau \in \{1,3,5,10,20,50,100\}$, $m \in \{5,10,20\}$, $p \in \{0.0, 0.2, \ldots, 1.0\}$

**Key results**:
- Single iteration: Recall@10 improves by **1.8%–10.97%** across backbone models vs. best baselines
- Multi-iteration (3 iterations, HSTU on Sports): NDCG@10 grows from 8.02% → 13.92%
- Weak-to-strong: a weaker teacher model (SASRec) training a stronger student (HSTU) yields +1.95% NDCG@10, confirming the regularization effect is not teacher-capacity-dependent
- Data density: +342.14% increase in interaction density across 3 iterations while information entropy also increases (vs. heuristic augmentation which adds low-entropy noise)

**Critical ablation**:
- Without fidelity control, performance collapses after iteration 1 (NDCG@10: 0.0293 → 0.0119), confirming the quality gate is essential
- Optimal $\tau \in [5,10]$: too strict ($\tau=1$) limits accepted data; too lenient ($\tau=100$) admits noise
- Optimal $p \approx 0.5$: pure exploitation ($p=1$) limits novelty; pure exploration ($p=0$) generates incoherent sequences
