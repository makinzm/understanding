# Meta Information

- URL: [PAC-Bayes Bounds for Gibbs Posteriors via Singular Learning Theory](https://arxiv.org/abs/2604.17219)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wang, C., & Yang, Y. (2026). PAC-Bayes Bounds for Gibbs Posteriors via Singular Learning Theory. arXiv:2604.17219.

# PAC-Bayes Bounds for Gibbs Posteriors via Singular Learning Theory

This paper derives finite-sample PAC-Bayes generalization bounds for Gibbs posteriors whose tightness is governed by the real log canonical threshold (RLCT) — a geometric quantity from singular learning theory that captures the intrinsic complexity of overparameterized models. The result replaces the ambient-dimension-dependent covering-number bounds of classical theory with RLCT-dependent rates that automatically adapt to true model complexity.

**Target audience**: Researchers in statistical learning theory, Bayesian deep learning, and overparameterized model analysis who need non-asymptotic generalization guarantees beyond regular models.

## Background

### PAC-Bayes Framework

PAC-Bayes theory provides non-asymptotic generalization guarantees for randomized (or averaged) hypotheses represented by data-dependent probability measures — Gibbs posteriors in this case. The classic PAC-Bayes bound for a posterior $\rho$ and prior $\pi$ states that with high probability:

```math
\begin{align}
\mathbb{E}_{\theta \sim \rho}[R(\theta)] \leq \mathbb{E}_{\theta \sim \rho}[R_n(\theta)] + \sqrt{\frac{\mathrm{KL}(\rho \| \pi) + \log(1/\delta)}{2n}}
\end{align}
```

where $R(\theta) = \mathbb{E}_Z[\ell(\theta; Z)]$ is the population risk, $R_n(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta; Z_i)$ is the empirical risk, and $\mathrm{KL}(\rho \| \pi)$ is the KL divergence penalizing complexity. Classical bounds require balancing these two terms, but do not handle unbounded losses or singularities gracefully.

### Gibbs Posterior

The Gibbs posterior with learning rate $\omega > 0$ and prior $\varphi$ is the data-dependent measure:

```math
\begin{align}
\Pi_n(d\theta) \propto \exp\{-\omega n R_n(\theta)\}\, \varphi(d\theta)
\end{align}
```

where $\theta \in \Theta$ (compact parameter space). The learning rate $\omega$ controls how sharply the posterior concentrates around empirical risk minimizers. When $\omega = 1$, this corresponds to the standard tempered Bayesian posterior for the exponential family likelihood with loss $\ell$.

### Real Log Canonical Threshold (RLCT)

The RLCT $\lambda$ characterizes the singularity structure of the population excess risk $R(\theta, \theta^*) = R(\theta) - R(\theta^*)$ near the true parameter $\theta^*$. It is the negative of the largest pole of the zeta function:

```math
\begin{align}
\zeta(s) = \int_\Theta R(\theta, \theta^*)^s\, \varphi(d\theta)
\end{align}
```

which converges for $\mathrm{Re}(s) > 0$ and extends meromorphically to $\mathbb{C}$.

> [!NOTE]
> In regular models (where the Fisher information is non-singular at $\theta^*$), $\lambda = d/2$ where $d = \dim\Theta$. In singular models — neural networks, mixture models, matrix factorizations — $\lambda < d/2$ and can be fractional, reflecting fewer effective degrees of freedom.

The dominant asymptotic of $\zeta(s)$ near $s = 0$ determines the behavior of the marginal likelihood. By Hironaka's resolution of singularities theorem, $\lambda$ is always a positive rational number and is computable algebraically.

## Main Result

### Assumption (Bernstein-type Moment Condition)

For constants $L, b > 0$, the loss $\ell(\theta, \theta^*; Z) = \ell(\theta; Z) - \ell(\theta^*; Z)$ satisfies: for all $|\omega| \leq 1/(2b)$,

```math
\begin{align}
\mathbb{E}_Z\!\left[\exp\bigl\{\omega\bigl(\ell(\theta, \theta^*; Z) - R(\theta, \theta^*)\bigr)\bigr\}\right] \leq \exp\!\left\{\frac{\omega^2 L \cdot R(\theta, \theta^*)}{2}\right\}
\end{align}
```

This is a sub-exponential condition that ties the moment generating function of the centered loss to the population excess risk, enabling two-sided concentration. It is satisfied by:

- **Squared loss** under $\sigma^2$-sub-Gaussian noise: $L = 32B_0^2 + 4\sigma^2$, $1/(2b) = \min\{3/(16B_0^2), 1/(2\sigma^2)\}$
- **Logistic loss** under margin condition $\tau \leq \eta(X) \leq 1-\tau$: $L = 8/[\tau(1-\tau)]$, $b = \log(1 + e^{B_3})$

### Theorem 5 (Main PAC-Bayes Bound)

Under the Bernstein-type condition, for any $\omega \in (0, 1/(2b))$ with $\omega L < 2$, with probability at least $1-\delta$ over the training data:

```math
\begin{align}
\int_\Theta R(\theta, \theta^*)\, \Pi_n(d\theta) \leq \frac{2}{(1 - \omega L/2)\,\omega n}\!\left[\lambda \log n - (m-1)\log\log n + \log\frac{2}{\delta} + C_0(\varphi, L, \omega)\right]
\end{align}
```

where $\lambda$ is the RLCT, $m \geq 1$ is the multiplicity of the dominant pole of $\zeta(s)$, and $C_0$ is a constant depending on the prior, $L$, and $\omega$.

> [!IMPORTANT]
> The leading term is $\lambda \log n / n$, not the classical $\sqrt{d/n}$ or $d/n$ rates. Since $\lambda \ll d/2$ in singular models, this is substantially tighter. The bound automatically adapts to the intrinsic complexity of the true model, not the ambient parameter count.

### Proof Strategy (Key Steps)

1. **Two-sided exponential concentration**: Apply the Bernstein MGF condition to obtain forward and reverse bounds between $R_n(\theta)$ and $R(\theta, \theta^*)$.
2. **Donsker-Varadhan variational identity**: Express KL divergence via $\mathrm{KL}(\rho \| \pi) = \sup_f \{\mathbb{E}_\rho[f] - \log \mathbb{E}_\pi[e^f]\}$, then substitute PAC-Bayes arguments.
3. **Posterior measure selection**: Choose a population-level Gibbs measure $\rho^* \propto \exp\{-((1+\omega L/2)\omega n/2) R(\theta,\theta^*)\}\,\varphi(d\theta)$ to combine empirical and KL terms into a single marginal integral.
4. **RLCT asymptotic expansion**: The marginal integral equals $\zeta(s)$ evaluated at a specific $s$, whose asymptotic expansion as $n \to \infty$ yields $\lambda \log n - (m-1)\log\log n + O(1)$ by singular learning theory.

The key departure from classical approaches: instead of bounding the empirical risk and KL terms separately and then optimizing the posterior, the authors jointly optimize by choosing $\rho^*$ so the two terms collapse into the zeta function integral.

## Applications

### Theorem 6: Matrix Completion

**Setup**: Observe noisy entries of $M^* \in \mathbb{R}^{d_1 \times d_2}$ with rank $r^*$. Model: $M = UV^\top$ with $U \in \mathbb{R}^{d_1 \times H}$, $V \in \mathbb{R}^{d_2 \times H}$ for overparameterized rank $H \geq r^*$.

**Input/Output**: Input is $n$ observed (index, value) pairs; output is $\hat{M} = \int UV^\top \Pi_n(dU\, dV)$.

**RLCT** depends on the regime of $(H, d_1, d_2, r^*)$:

```math
\begin{align}
\lambda = \begin{cases}
\dfrac{Hd_2 - Hr^* + d_1 r^*}{2} & \text{if } H \leq d_1 - d_2 + r^* \\
\text{(three other regimes based on relative sizes of } H, d_1, d_2, r^*\text{)}
\end{cases}
\end{align}
```

In all regimes, $\lambda \ll d_1 d_2 / 2$ (ambient half-dimension) and $\lambda < H(d_1+d_2)/2$ (half the parameter count), confirming the bound is tighter than dimension-based alternatives.

### Theorem 7: ReLU Neural Networks

**Setup**: Regression with a target function $f^*$ realizable by some minimal network of architecture $N^* = (H_1^*, \ldots, H_{K}^*)$. Train an overparameterized ReLU network of any larger architecture.

**Input**: $x \in \mathbb{R}^{H_0}$; **Output**: scalar prediction $\hat{y} \in \mathbb{R}$.

**RLCT upper bound**:

```math
\begin{align}
\bar{\lambda}_{\text{ReLU}} = \frac{1}{2}\sum_{k=2}^{N^*} H_k^*(H_{k-1}^* + 1)
\end{align}
```

This equals **half the number of parameters in the minimal network** $N^*$, regardless of the size of the actual (overparameterized) network used for training. The bound adapts to the true function's complexity, not the trained model's parameter count.

> [!NOTE]
> This formalizes the intuition behind implicit regularization: even though the trained network may have millions of parameters, the generalization bound scales with the minimal network needed to represent $f^*$.

## Comparison with Related Methods

| Method | Rate | Complexity measure | Requires bounded loss? |
|---|---|---|---|
| Classical covering-number bounds | $\sqrt{d/n}$ | Ambient dimension $d$ | Yes |
| Alquier et al. (2021) PAC-Bayes | $\sqrt{\log N(\varepsilon)/n}$ | Metric entropy | Yes (bounded $\ell \in [0,1]$) |
| Watanabe (2009) SLT | $\lambda \log n / n$ | RLCT $\lambda$ | No (asymptotic) |
| **This work** | $\lambda \log n / n$ | RLCT $\lambda$ | No (sub-exponential) |

The key advantage over Watanabe (2009) is **non-asymptotic guarantees**: this paper provides finite-$n$ high-probability bounds rather than asymptotic in-expectation convergence. The key advantage over Alquier et al. (2021) is handling **unbounded losses** and **singular models** where metric entropy blows up.

## Experiments

- **Dataset**: None — the paper is purely theoretical with no empirical experiments.
- **Hardware**: Not applicable.
- **Results**: Analytical bounds for matrix completion and ReLU networks, showing $\lambda \ll d/2$ in both cases.

> [!CAUTION]
> The paper does not empirically validate whether the RLCT-based bound is tight or computationally tractable to estimate in practice. Computing $\lambda$ via resolution of singularities is algebraically feasible but potentially expensive for large networks.
