# Meta Information

- URL: [Fast EXP3 Algorithms](https://arxiv.org/abs/2512.11201)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sato, R., & Ito, S. (2025). Fast EXP3 Algorithms. arXiv:2512.11201.

---

# Fast EXP3 Algorithms

## Overview

EXP3 (Exponential-weight algorithm for Exploration and Exploitation) is a canonical algorithm for the **adversarial multi-armed bandit** problem, where an agent repeatedly selects one of $K$ arms and observes a loss (or reward) while an adversary may choose losses adaptively. The standard implementation runs in $O(K)$ time per round due to the need to sample from a weight distribution over all $K$ arms and to compute the total weight $W_t$.

This paper systematically reduces the per-round time complexity of EXP3 from $O(K)$ down to $O(1)$ expected time while preserving the optimal regret constant, making EXP3 practical at scales where $K$ is large.

**Who benefits:** Practitioners and researchers working on large-scale bandit problems (e.g., online recommendation with millions of items, combinatorial bandits) where $K \gg 1$ makes the naive $O(K)$ per-round cost prohibitive.

---

## Problem Setup

### Adversarial Multi-Armed Bandit

At each round $t = 1, \ldots, T$:
1. The agent selects arm $a_t \in [K] := \{1, \ldots, K\}$ according to distribution $\mathbf{p}_t \in \Delta^{K-1}$.
2. An adversary reveals loss $\ell_{t,a_t} \in [0,1]$ for the chosen arm only.
3. Regret is measured as:

$$\bar{R}_T := \mathbb{E}\left[\sum_{t=1}^T \ell_{t,a_t}\right] - \min_{i \in [K]} \sum_{t=1}^T \ell_{t,i}$$

### Standard EXP3

EXP3 maintains a weight vector $\mathbf{w}_t \in \mathbb{R}_{>0}^K$ initialized to $w_{1,i} = 1$ for all $i$.

**Sampling distribution:**

$$p_{t,i} := \frac{w_{t,i}}{W_t}, \quad W_t := \sum_{i=1}^K w_{t,i}$$

**Importance-weighted loss estimator for the selected arm $a_t$:**

$$\hat{\ell}_{t,i} := \frac{\ell_{t,i}}{p_{t,i}} \cdot \mathbb{I}[i = a_t]$$

**Weight update (only the selected arm changes):**

$$w_{t+1,i} :=
\begin{cases}
w_{t,i} \exp\!\left(-\eta \frac{\ell_{t,i}}{p_{t,i}}\right) & \text{if } i = a_t \\
w_{t,i} & \text{otherwise}
\end{cases}$$

**Learning rate:** $\eta := \sqrt{\dfrac{2 \ln K}{KT}}$

**Regret bound:**

$$\bar{R}_T \leq \sqrt{2} \cdot \sqrt{TK \ln K}$$

> [!NOTE]
> Because only $w_{t, a_t}$ changes each round, the total weight changes as $W_{t+1} = W_t - w_{t,a_t}(1 - \exp(-\eta \hat{\ell}_{t,a_t}))$.

**Computational bottleneck of naive implementation:**
- Sampling $a_t \sim \mathbf{p}_t$: requires computing cumulative sums over all $K$ weights → $O(K)$.
- Updating $W_t$: $O(1)$ since only one arm changes.
- Total per-round: $O(K)$.

---

## Computational Model

All analyses use the **RAM (Random Access Machine) model**, in which arithmetic operations (including $\exp$, $\log$), memory reads/writes, and random number generation each cost $O(1)$ time. This is the standard model for algorithm and data structure analysis and is justified in practice by constant-time hardware instructions.

---

## O(log K) Implementation via Segment Tree (Section 3)

### Data Structure

A **segment tree** (balanced binary tree) over $K$ leaves is maintained where:
- Each **leaf** stores the current weight $w_{t,i}$ for arm $i$.
- Each **internal node** stores the sum of all weights in its subtree.
- The root stores $W_t = \sum_{i=1}^K w_{t,i}$.

### Sampling in O(log K)

**Input:** Root value $W_t$.
**Output:** Arm index $a_t$ sampled proportionally to $\mathbf{w}_t$.

```
sample(node, r):
  if node is a leaf:
    return node.arm_index
  if r < node.left_child.sum:
    return sample(node.left_child, r)
  else:
    return sample(node.right_child, r - node.left_child.sum)

# Main call:
r ~ Uniform[0, W_t)
a_t = sample(root, r)
```

Traversal visits exactly one node per level → $O(\log K)$ worst-case.

### Weight Update in O(log K)

```
update(leaf_i, new_weight):
  leaf_i.sum = new_weight
  node = leaf_i.parent
  while node is not None:
    node.sum = node.left_child.sum + node.right_child.sum
    node = node.parent
```

**Per-round cost:** $O(\log K)$ for sampling + $O(\log K)$ for update = $O(\log K)$ total.

**Advantage over prior work (Chewi et al., 2022):** The segment tree achieves $O(\log K)$ with the **same regret constant** (coefficient 2 in $2\sqrt{KT \ln K}/\sqrt{2}$), whereas Chewi et al.'s approach doubled the regret constant to 4 when using sublinear-time sampling.

---

## O(1) Expected-Time Implementation via Alias Method (Section 5)

### Alias Method Background

The **alias method** (Walker, 1977) preprocesses a discrete distribution over $K$ elements in $O(K)$ time, then samples in $O(1)$ worst-case time by partitioning the distribution into $K$ bins each of equal probability $1/K$.

**Construction:** Let $\bar{W} = W / K$.
1. Separate arms into Small ($w_i < \bar{W}$) and Large ($w_i \geq \bar{W}$) groups.
2. Repeatedly pair one Small and one Large arm into a bin: fill the Small arm's bin to capacity $\bar{W}$ using part of the Large arm's weight.
3. Store for each bin: (primary arm index, alias arm index, threshold $q_i$).

**Sampling:** Draw bin index $j \sim \text{Uniform}[K]$, then return primary if $U \leq q_j$, else return alias. Both steps are $O(1)$.

### Challenge: Dynamic Weights

EXP3 updates $w_{t, a_t}$ every round. Rebuilding the alias table from scratch costs $O(K)$, eliminating the speedup.

### Solution: Periodic Reconstruction + Rejection Sampling

**Key insight:** Only one arm changes per round, and the change is multiplicative by a factor $\leq 1$. So the distribution drifts slowly.

**Algorithm (Section 5):**

Let $\tau$ denote the round of the last alias table reconstruction.

**Every $K$ rounds (at round $\tau$):**
- Rebuild alias table from $\mathbf{w}_\tau$ in $O(K)$ time (amortized $O(1)$ per round).

**Each round $t$ (between reconstructions), sampling:**
1. Draw candidate arm $k$ from the stale alias table (distribution $\propto \mathbf{w}_\tau$).
2. Accept $k$ with probability:

$$\alpha_{t,k} := \frac{w_{t,k}}{w_{\tau,k}}$$

3. If rejected, repeat from step 1.

**Acceptance probability analysis:**

The probability of acceptance in one attempt is:

$$\Pr[\text{accept}] = \sum_k \frac{w_{\tau,k}}{W_\tau} \cdot \frac{w_{t,k}}{w_{\tau,k}} = \frac{W_t}{W_\tau}$$

Since weights only decrease (each update multiplies $w_{t,a_t}$ by $\exp(-\eta \hat{\ell}) \leq 1$), and $W_t \geq W_\tau e^{-2}$ (shown below), the expected number of attempts is:

$$\mathbb{E}[\text{attempts}] = \frac{W_\tau}{W_t} \leq e^2 \leq 7.39$$

**Proof that $W_t \geq W_\tau e^{-2}$ (Proposition 5.1):**

In $K$ rounds, the total weight decreases by at most:

$$W_\tau \cdot \eta K \cdot 1 = W_\tau \cdot \sqrt{\frac{2 \ln K}{KT}} \cdot K \leq W_\tau$$

Under $\eta \leq 1/2$ and $\eta K \leq 1$, one can show $W_t / W_\tau \geq e^{-2}$.

> [!IMPORTANT]
> The bound $e^2 \leq 7.39$ is a worst-case expected number of rejection attempts. In practice, the acceptance rate is much higher early in training when weights are close to uniform.

**Per-round cost:** $O(1)$ expected for rejection sampling + $O(1)$ amortized for alias reconstruction = **$O(1)$ expected total**.

### Double Buffering for True O(1) (not amortized)

To avoid a single round with $O(K)$ reconstruction cost, double buffering is used:
- Maintain two alias tables: one **active** (used for sampling) and one **being rebuilt** in the background.
- Rebuild the next table incrementally, one arm per round ($O(1)$ per round), then atomically swap at the $K$-round boundary.

**Result:** $O(1)$ worst-case time per round (not merely amortized).

---

## Comparison with Related Methods

| Algorithm | Time/Round | Regret Coefficient $c$ in $c\sqrt{KT \ln K}$ | Notes |
|---|---|---|---|
| Naive EXP3 | $O(K)$ | $\sqrt{2} \approx 1.41$ | Baseline |
| Chewi et al. (2022) | $O(\log^2 K)$ amortized | 4 | Doubled regret constant |
| Segment Tree (this work) | $O(\log K)$ worst-case | $\sqrt{2}$ | Preserves regret constant |
| Advanced structure (this work) | $O(1)$ expected | $\sqrt{2}$ | Complex Matias-Monier DS |
| Alias + Rejection (this work) | $O(1)$ expected | $\sqrt{2}$ | Practical, simple |

> [!NOTE]
> Chewi et al. (2022) achieved sublinear time but at the cost of doubling the regret constant from $\sqrt{2}$ to $4$. This paper is the first to achieve $O(1)$ expected time **without** degrading the regret constant.

---

## Anytime EXP3 (Section 7)

When the horizon $T$ is unknown, three approaches are analyzed:

### Approach 1: Doubling Trick

Run EXP3 in epochs of doubling length $T_k = 2^k$. At each epoch boundary, reset weights.

- **Regret:** $\bar{R}_T \leq (2 + \sqrt{2}) \cdot \sqrt{KT \ln K} \approx 3.41 \cdot \sqrt{KT \ln K}$... in practice yields constant $\approx 4.83$.
- **Compatibility with $O(1)$ time:** Yes, reset is $O(K)$ every $2^k$ rounds (amortized $O(1)$).

### Approach 2: Time-Varying Learning Rate

Use $\eta_t = \sqrt{\frac{2 \ln K}{Kt}}$, updated each round.

- **Regret:** $\bar{R}_T \leq 2\sqrt{KT \ln K}$, optimal leading constant.
- **Problem:** $W_t / W_\tau$ can grow unboundedly when $\eta_t$ decreases, making rejection sampling inefficient.
- **Not easily accelerated** to $O(1)$ time.

### Approach 3: Delayed Parameter Updates (Proposed)

Use a **fixed learning rate within each $K$-round block**, updating $\eta$ only at block boundaries.

Let block $b$ span rounds $\tau_b + 1, \ldots, \tau_b + K$. Use learning rate:

$$\eta_b := \sqrt{\frac{2 \ln K}{K \tau_b}}$$

**Proposition 7.1:**

$$\bar{R}_T \leq 2\sqrt{KT \ln K} + K\sqrt{\ln K}$$

The leading term $2\sqrt{KT \ln K}$ matches the time-varying rate approach. The additive $K\sqrt{\ln K}$ term is lower-order for large $T$.

**Compatibility:** Within each block, $\eta$ is fixed, so $W_t / W_\tau \leq e^2$ still holds → alias+rejection sampling works in $O(1)$ expected time.

---

## Extension to EXP4 (Section 6)

**EXP4** (Exponential-weight algorithm for Exploration and Exploitation with Experts) handles the **bandit with expert advice** setting, where $N$ experts each recommend an arm per round.

**Setup:** $N$ experts, $K$ arms, expert weights $\tilde{w}_{t,j}$, expert-arm mapping $e_t(j) \in [K]$.

**Weight update:**

$$\tilde{w}_{t+1,j} := \tilde{w}_{t,j} \exp\!\left(-\eta \frac{\mathbb{I}[j \in E_t(j_t)] \ell_{t, e_t(j_t)}}{\sum_{j' \in E_t(j)} \tilde{p}_{t,j'}}\right)$$

where $E_t(j) := \{j' : e_t(j') = e_t(j)\}$ is the set of experts recommending the same arm as expert $j$.

**Per-round time with segment tree:** $O\!\left(\max_j |E_t(j)|\right)$, since only experts in $E_t(j_t)$ need weight updates.

**Speedup:** When experts cluster around few arms (small $|E_t(j)|$), this is much faster than $O(N)$.

---

# Experiments

- **Datasets:** No empirical datasets; the paper is theoretical.
- **Evaluation:** Theoretical regret bounds and worst-case time complexity analysis.
- **Key quantitative results:**
  - Fixed-horizon regret: $\bar{R}_T \leq \sqrt{2KT \ln K}$ with $O(1)$ expected time per round.
  - Anytime regret (delayed updates): $\bar{R}_T \leq 2\sqrt{KT \ln K} + K\sqrt{\ln K}$ with $O(1)$ expected time.
  - Expected sampling attempts: $\leq e^2 \leq 7.39$ regardless of $K$ or $T$.

---

## Applicability

| Context | Applicable? | Reason |
|---|---|---|
| Large action spaces ($K \gg 1$) | Yes | $O(1)$ vs $O(K)$ is significant |
| Online recommendation (millions of items) | Yes | Direct application |
| Combinatorial bandits with exponentially many arms | Possible | If actions factorize |
| Contextual bandits (linear EXP3) | Partial | Sampling speedup applies; feature updates are separate |
| Small $K$ (e.g., $K \leq 100$) | Marginal | Overhead of alias construction may dominate |
| Unknown horizon $T$ | Yes | Delayed-update anytime variant available |
