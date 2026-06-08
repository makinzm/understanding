# Meta Information

- URL: [From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence](https://arxiv.org/abs/2601.03220)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence. arXiv:2601.03220.

# From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence

## Overview and Motivation

Classical information theory (Shannon entropy, Kolmogorov complexity) treats information as observer-independent: a string has a fixed information content regardless of who or what is trying to extract it. This assumption breaks down for machine learning systems, which are computationally bounded and cannot solve NP-hard or cryptographically hard problems. The paper introduces **epiplexity** (epistemic complexity), a measure of the amount of structural, learnable information that a computationally-bounded observer can extract from data.

**Who benefits**: Machine learning researchers studying pre-training data quality, synthetic data generation, and out-of-distribution generalization. Theorists seeking rigorous foundations for empirical observations about data ordering, induction, and emergence.

**When to apply**: When selecting or transforming datasets to maximise downstream generalizability; when explaining why data ordering, factorisation direction, or synthetic augmentation affects learning outcomes.

**Key question**: When does data contain structure that bounded learners can exploit, independently of any specific downstream task?

## Three Apparent Paradoxes

### Paradox 1: Deterministic Transformations Can Create Learnable Information

Classical theory says deterministic functions preserve or reduce information. Yet AlphaZero's self-play (a deterministic process) produces highly informative training data, and synthetic data from trained models transfers well across tasks. Theorem 12 of the paper resolves this: a cryptographically-secure pseudo-random generator (CSPRNG) can increase time-bounded entropy, because inverting the generator requires exponential time. A computationally-bounded observer sees unpredictable outputs as genuinely random—and therefore as containing learnable structure relative to the bounded model class.

### Paradox 2: Data Ordering Changes Information Content

Shannon entropy is invariant to permutations of the dataset. But empirically, left-to-right English text trains causal language models better than reversed text. Theorem 13 shows that under one-way permutations, different factorisation orders of the same joint distribution $p(x_1, \dots, x_n)$ yield substantially different time-bounded entropies. The ordering determines which conditional distributions are easy vs. computationally hard to model.

### Paradox 3: Learned Models Can Encode More Structure Than Generating Processes

A model trained to predict text can generalise via induction to tasks the generating process never produced. Epiplexity captures this: the model's learned representation extracts structural patterns (circuits) absent from the generating distribution but discoverable by optimisation.

## Formal Framework

### Time-Bounded Probabilistic Models

A **time-bounded probabilistic model** $P$ is a program on a prefix-free universal Turing machine that:

- Samples from a distribution over binary strings in time $T(n)$ where $n = |x|$
- Evaluates $P(x) \in [0,1]$ exactly within time $T(n)$

$T(n)$ defines the computational budget (e.g., polynomial, exponential).

### Epiplexity and Time-Bounded Entropy

Given time budget $T$ and data $X$ (a random variable over binary strings), the optimal time-bounded model $P^*$ minimises the two-part description length:

$$P^* = \arg\min_{P \in \mathcal{P}_T} \left[ |P| + \mathbb{E}\left[\log \frac{1}{P(X)}\right] \right]$$

where $|P|$ is the program length in bits and $\mathcal{P}_T$ is the class of programs running within time $T$.

The two components are:

| Quantity | Symbol | Meaning |
|---|---|---|
| **Epiplexity** | $S_T(X) = \|P^*\|$ | Program length = structural, learnable information |
| **Time-bounded entropy** | $H_T(X) = \mathbb{E}[\log 1/P^*(X)]$ | Residual uncertainty that cannot be compressed further |
| **Time-bounded MDL** | $S_T(X) + H_T(X)$ | Generalisaton of minimum description length |

> [!NOTE]
> Epiplexity separates structure (what a bounded learner can learn) from noise (irreducible randomness under computational constraints). The same binary string may have high Shannon entropy but low epiplexity (e.g., CSPRNG output), or high epiplexity but low Shannon entropy (e.g., a chess database with deterministic optimal moves that require sophisticated look-up circuits to predict).

### Relationship to Classical Frameworks

- **Shannon entropy** $H(X)$: Observer-independent, permutation-invariant. Recovered in the limit $T \to \infty$.
- **Kolmogorov complexity** $K(X)$: Uncomputable, time-unlimited. Epiplexity is its computationally-bounded analogue.
- **Sophistication**: Prior work measuring structural content via two-part codes, limited by Chaitin's incompleteness. Epiplexity avoids this by restricting to time-bounded models.

## Measurement Procedures

Because the optimal $P^*$ is not directly accessible, the paper proposes two practical estimators.

### Prequential Coding

Estimates model description length as the area under the excess loss curve during training:

$$|P_{\text{preq}}| \approx \sum_{i=1}^{n} \left( \log \frac{1}{P_i(Z_i)} - \log \frac{1}{P_M(Z_i)} \right)$$

where $P_i$ is the model at step $i$, $P_M$ is the converged model, and $Z_i$ is the $i$-th token.

- **Input**: Training log of per-sample losses, converged model's test loss
- **Output**: Scalar estimate of epiplexity in bits
- **Advantage**: No additional computation beyond a standard training run
- **Limitation**: Heuristic; relies on symmetry of information assumption (order-independence of KL divergence under i.i.d. data)

### Requential Coding

Constructs an explicit program code using cumulative KL divergence between a student and a teacher model:

$$|P_{\text{req}}| \approx \sum_{t=1}^{T} \text{KL}\left(P^t_{\text{teacher}} \,\|\, P^t_{\text{student}}\right)$$

- **Input**: Teacher model checkpoints $\{P^t_{\text{teacher}}\}$, student model $P^t_{\text{student}}$ trained to match each checkpoint
- **Output**: Rigorous upper bound on epiplexity (explicit code with known runtime)
- **Advantage**: Theoretically rigorous, provides certified runtime
- **Limitation**: 2–10× computational overhead vs. prequential

## Key Theoretical Results

**Theorem 9** (CSPRNGs and pseudorandomness): Cryptographically-secure pseudo-random number generators have maximal time-bounded entropy ($n - O(\varepsilon)$) and minimal epiplexity ($c + O(n\varepsilon)$) for any polynomial-time observer. This formally distinguishes pseudorandomness from true randomness while explaining why CSPRNG outputs appear unstructured to bounded learners.

**Theorem 10** (Existence): Under one-way functions, random variables with epiplexity $\Omega(\log n)$ exist, guaranteeing the concept is non-trivial.

**Theorem 12** (Deterministic transformations increase $H_T$): Applying a one-way function $f$ to a random input $X$ yields $H_T(f(X)) > H_T(X)$ for bounded $T$, because computing $f^{-1}$ requires super-polynomial time.

**Theorem 13** (Factorisation dependence): Under one-way permutations, $H_T(x_1, \dots, x_n)$ and $H_T(x_n, \dots, x_1)$ differ by $\Omega(\text{poly}(n))$ for some distributions, explaining why left-to-right and right-to-left factorisation orders produce different learning outcomes.

## Algorithm: Prequential Epiplexity Estimation

```
Input:  dataset Z = {z_1, ..., z_n}, training algorithm A, model M_final
Output: epiplexity estimate S_T(Z)

1. Train model M using A on Z, recording loss at each step i:
     loss_i ← -log P_i(z_i)  // per-token NLL at step i

2. Compute final model's average loss:
     loss_final ← mean_{z in Z} [-log M_final(z)]

3. S_T(Z) ← Σ_{i=1}^{n} (loss_i - loss_final)

Return S_T(Z)
```

## Experiments

- **Datasets**:
  - Elementary Cellular Automata (ECA) rules: Rule 15 (periodic), Rule 30 (pseudorandom/chaotic), Rule 54 (complex)
  - Chess games (ordered forward and reversed)
  - Hard induction tasks: Rule 30 outputs with some input bits masked
  - Easy induction tasks: Markov chains with mixed hidden/visible transition probabilities

- **Hardware**: Not explicitly specified
- **Optimizer**: Not explicitly specified (standard neural network training implied)
- **Results**:
  - Rule 30 forward prediction converges to Shannon entropy; reversed prediction shows persistent gap, confirming asymmetric time-bounded complexity
  - Hard induction: epiplexity grows exponentially with number of hidden bits (models learn inversion strategies absent from generating process)
  - Easy induction: epiplexity grows as models capture both original and inductive strategies
  - Complex cellular automata (Rule 54) exhibits high both entropy and epiplexity, matching theoretical predictions

## Comparison with Similar Frameworks

| Framework | Observer-dependent | Computable | Ordering-sensitive | Emergence |
|---|---|---|---|---|
| Shannon entropy | No | Yes | No | No |
| Kolmogorov complexity | No | No | No | Partial |
| Sophistication | No | No | No | No |
| **Epiplexity** | **Yes** | **Yes (approximate)** | **Yes** | **Yes** |

> [!TIP]
> Sophistication (Vitányi, 2006) is the closest prior concept, but suffers from incomputability and Chaitin-incompleteness (no program provably minimises the two-part code). Epiplexity sidesteps this by fixing a computational time bound $T$, making the optimisation well-defined in principle.

## Applications

### Pre-training Data Quality

Epiplexity correlates with out-of-distribution generalisation: high-epiplexity data (e.g., natural text) induces learnable circuits reusable across tasks, explaining why text pre-training transfers more broadly than image pre-training.

### Data Selection and Generation

Unlike model selection (choosing architectures for a fixed dataset), epiplexity enables **data selection**—generating or transforming datasets to maximise structural content for downstream generalisation. A dataset's epiplexity measures its potential to teach reusable representations.

### Factorisation Choice

When a joint distribution admits multiple factorisations (e.g., left-to-right vs. right-to-left token prediction), epiplexity guides which ordering yields richer learned representations.

> [!CAUTION]
> The practical measurements (prequential and requential coding) are heuristics or upper bounds, not exact epiplexity values. Interpreting absolute epiplexity scores across different datasets or models requires careful normalisation and is an open research challenge.

## Limitations

- Theorem 10's logarithmic lower bound on epiplexity growth is far below empirically observed power-law scaling; tighter lower bounds remain open.
- Requential coding incurs 2–10× overhead; cheaper rigorous estimators are needed.
- Extension to continuous domains, non-binary strings, and memory-bounded (rather than time-bounded) observers requires further theoretical work.
- The interaction between different types of computational constraints (memory, architecture class, parallelism) and epiplexity is not yet characterised.
