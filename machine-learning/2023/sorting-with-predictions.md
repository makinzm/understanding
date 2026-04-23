# Meta Information

- URL: [Sorting with Predictions](https://arxiv.org/abs/2311.00749)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dinur, I., Hecht, G., Kur, G., & Zaoui, S. (2023). Sorting with Predictions. arXiv:2311.00749.

# Overview

This paper studies how **prediction information** can help sorting algorithms overcome the classical $\Omega(n \log n)$ comparison lower bound. The framework falls under *learning-augmented algorithms*, where a predictor provides approximate information that is trusted but may contain errors. When predictions are perfect, sorting cost is $O(n)$; as prediction quality degrades, the cost smoothly increases back to $O(n \log n)$, satisfying the three key properties: **consistency** (optimal with perfect predictions), **robustness** (bounded performance under adversarial predictions), and **smoothness** (graceful degradation).

> [!NOTE]
> The key insight is to measure prediction error *element-wise* rather than globally, enabling finer-grained analysis when errors are distributed non-uniformly.

# Settings and Error Measures

The paper addresses three distinct prediction models:

## Setting 1: Positional Predictions

Each element $a_i$ receives a predicted rank $\hat{p}(i) \in [n]$, approximating its true rank $p(i)$. Three error measures are defined:

| Error Measure | Formula | Interpretation |
|---|---|---|
| Displacement error | $\eta_i^\Delta = \|\hat{p}(i) - p(i)\|$ | Absolute difference between predicted and true rank |
| Left-error | $\eta_i^l = \|\{j : \hat{p}(j) \le \hat{p}(i) \wedge p(j) > p(i)\}\|$ | Items incorrectly predicted as smaller than $a_i$ |
| Right-error | $\eta_i^r = \|\{j : \hat{p}(j) \ge \hat{p}(i) \wedge p(j) < p(i)\}\|$ | Items incorrectly predicted as larger than $a_i$ |

The one-sided errors satisfy $\eta_i^l, \eta_i^r \le \eta_i^\Delta$ in general.

## Setting 2: Dirty Comparisons

Alongside the exact (clean) comparison $<$, a fast but potentially erroneous *dirty* comparison $\hat{<}$ is available. The error for element $a_i$ is:

$$\eta_i = \left|\{j \in [n] : (a_i < a_j) \neq (a_i \,\hat{<}\, a_j)\}\right|$$

i.e., the number of elements whose dirty comparison with $a_i$ gives the wrong answer.

## Setting 3: Multiple Predictors

Extends Setting 2 to $k$ dirty comparison oracles $\hat{<}^{(1)}, \ldots, \hat{<}^{(k)}$, where the best oracle is unknown in advance and must be selected online.

# Main Theorems

**Theorem 1.1** (Dirty Comparisons): There exists a randomized algorithm that sorts an array of $n$ elements using $O(n \log n)$ dirty comparisons and $O\!\left(\sum_{i=1}^n \log(\eta_i + 2)\right)$ clean comparisons in expectation, within $O(n \log n)$ total running time.

**Theorem 1.3** (Displacement Sort): There exists a deterministic algorithm that sorts using $O\!\left(\sum_{i=1}^n \log(\eta_i^\Delta + 2)\right)$ comparisons with positional predictions.

**Theorem 1.4** (Double-Hoover Sort): There exists a deterministic algorithm using $O\!\left(\sum_{i=1}^n \log(\min\{\eta_i^l, \eta_i^r\} + 2)\right)$ comparisons with positional predictions.

**Theorem 1.5** (Lower Bounds): No sorting algorithm can use $o\!\left(\sum_{i=1}^n \log(\eta_i + 2)\right)$ clean comparisons in the dirty comparison setting, nor $o\!\left(\sum_{i=1}^n \log(\eta_i^\Delta + 2)\right)$ comparisons in the positional prediction setting. These bounds are tight.

> [!IMPORTANT]
> When all $\eta_i = 0$ (perfect predictions), the total cost is $O(n)$. When all $\eta_i = n$, the cost returns to $O(n \log n)$, recovering standard sorting.

# Algorithms

## Algorithm 1: Dirty-Clean Sorting (Setting 2)

**Input**: Array $a_1, \ldots, a_n \in \mathbb{R}^n$; dirty comparison oracle $\hat{<}$; clean comparison $<$.
**Output**: Sorted permutation of input.

The algorithm inserts elements one by one into a binary search tree (BST), using three phases per insertion of $a_i$:

**Phase 1 — Dirty Search**: Traverse the BST using dirty comparisons, recording the traversal path as a sequence of nodes $(C_0, C_1, \ldots, C_T)$ with corresponding lower/upper bounds $(L_t, R_t)$ tightened at each step.

**Phase 2 — Verification**: Walk back from $C_T$ toward the root to find the last step $t^*$ at which the interval $(L_{t^*}, R_{t^*})$ still correctly contains $a_i$ (verified with one clean comparison per step). This identifies where the dirty path diverged from correctness.

**Phase 3 — Clean Binary Search**: From $C_{t^*}$, perform a clean binary search within the interval $(L_{t^*}, R_{t^*})$ to insert $a_i$ at its true position.

```
DirtyCleanSort(a[1..n]):
  T ← empty BST
  for i = 1 to n:
    # Phase 1: dirty traversal
    path ← DirtySearch(T, a[i])      # uses dirty comparisons
    t* ← Verify(path, a[i])          # clean rollback to find t*
    CleanInsert(T, a[i], from path[t*])  # clean binary search
  return InorderTraversal(T)
```

**Key Lemma**: Let $s_{t^*}$ be the size of the subtree at step $t^*$. Then $\mathbb{E}[\log(s_{t^*}+1)] = O(\log(\eta_i + 2))$, which bounds the expected number of clean comparisons per element.

## Algorithm 2: Displacement Sort (Setting 1)

**Input**: Array with positional predictions $\hat{p}(1), \ldots, \hat{p}(n)$; displacement errors $\eta_i^\Delta$.
**Output**: Sorted array.

Uses a **finger tree** (a balanced BST supporting $O(\log d)$ insertion near position $d$ from a finger):

1. Sort elements by predicted position $\hat{p}(i)$ (bucket sort in $O(n)$).
2. Insert elements into finger tree in predicted-position order; consecutive insertions in sorted predicted order are near each other.
3. Return inorder traversal.

The amortized cost per insertion is $O(\log(\eta_i^\Delta + 2))$, since element $a_i$ is inserted within distance $\eta_i^\Delta$ of its predecessor in predicted order.

```
DisplacementSort(a[1..n], p_hat[1..n]):
  order ← BucketSort(indices, key=p_hat)
  T ← empty FingerTree
  for i in order:
    T.FingerInsert(a[i])   # O(log(η_i^Δ + 2)) comparisons
  return InorderTraversal(T)
```

## Algorithm 3: Double-Hoover Sort (Setting 1)

**Input**: Array with positional predictions; one-sided errors $\eta_i^l, \eta_i^r$.
**Output**: Sorted array.

Runs $\lceil \log n \rceil + 1$ rounds with exponentially increasing *suction strength* $\delta = 1, 2, 4, \ldots, 2^{\lceil \log n \rceil}$. Maintains two sorted structures $L$ (built left-to-right) and $R$ (built right-to-left):

```
DoubleHooverSort(a[1..n], p_hat[1..n]):
  L ← [], R ← []
  for round = 0 to ⌈log n⌉:
    δ ← 2^round
    # Forward pass: insert into L if rank in L is within δ of predicted rank
    for i = 1 to n (predicted order):
      if a[i] > l_δ^i:       # l_δ^i = δ-th largest in L with predicted index < i
        L.Insert(a[i])
    # Backward pass: insert into R similarly
    for i = n downto 1 (predicted order):
      if a[i] < r_δ^i:
        R.Insert(a[i])
  return Merge(L, R)
```

The amortized cost per element is $O(\log(\min\{\eta_i^l, \eta_i^r\} + 2))$ comparisons.

## Algorithm 4: Multiple Predictors (Setting 3)

Uses a **multiplicative weights / expert advice** framework over $k$ dirty comparison oracles. At each step, the algorithm maintains a distribution over oracles and selects which oracle to follow based on past errors. After $O(\sqrt{nk \log k})$ rounds, the best oracle is identified and exploited.

# Differences from Related Work

| Work | Error Measure | Clean Comparisons | Notes |
|---|---|---|---|
| Lu et al. (2021) | Global error $w$ (inversions) | $O(nw)$ — becomes $O(n^2)$ with 1 error per item | Coarse-grained; degrades poorly |
| TimSort | Runs (presortedness) | $O(n + n \log r)$ where $r$ = # runs | No explicit predictions; adaptive to structure |
| Cook-Kim (1980) | Division into sorted sequences | $O(n \log s)$ — $s$ sequences | Adaptive but global measure |
| **This work** | Element-wise $\eta_i$ | $O\!\left(\sum_i \log(\eta_i+2)\right)$ | Fine-grained; $O(n)$ with perfect predictions |

> [!TIP]
> Finger trees are a functional data structure supporting $O(\log \min\{i, n-i\})$ split/merge and $O(\log d)$ finger-based insertions. See Hinze & Paterson (2006) for full details.

> [!NOTE]
> The paper's element-wise lower bound (Theorem 1.5) uses an information-theoretic argument: for each element $a_i$ with error $\eta_i$, there are at least $\eta_i + 1$ possible true positions, requiring $\Omega(\log(\eta_i + 1))$ comparisons to identify the correct one.

# Experiments

- **Dataset 1 — Synthetic Class Setting**: Arrays of size $n \in \{1\,000, 10\,000, 100\,000, 1\,000\,000\}$; items divided into $c$ classes, with predictions drawn uniformly from within each class's predicted range.
- **Dataset 2 — Synthetic Decay Setting**: Starting from the true ranking, one randomly chosen item shifts by $\pm 1$ per timestep; predictions from years $0$–$50$ of drift are evaluated.
- **Dataset 3 — Real-World Population Rankings**: Annual country/region population rankings from the World Bank (1960–2010); $n = 261$ entities; prediction for 2010 is taken from earlier decades.

Baselines compared: QuickSort, MergeSort, TimSort, Odd-Even Merge Sort, Cook-Kim division sort.

Key finding: All three proposed algorithms outperform baselines when prediction quality is high (small $\eta_i$), and match or gracefully approach baseline performance as $\eta_i \to n$.
