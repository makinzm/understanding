# Meta Information

- URL: [Upper Bounds for Local Learning Coefficients of Three-Layer Neural Networks](https://arxiv.org/abs/2603.12785)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kurumadani, Y. (2026). Upper Bounds for Local Learning Coefficients of Three-Layer Neural Networks. arXiv:2603.12785.

# Overview

This paper derives upper-bound formulas for the **local learning coefficient** (real log canonical threshold, RLCT) of three-layer neural networks at singular points, extending previous work that applied only at non-singular points. The formulas cover general real analytic activation functions, including swish, tanh, and polynomial activations. When the input dimension equals one ($N=1$), the derived bounds match known exact values, making them tight in that setting.

**Target audience:** Researchers in singular learning theory (Watanabe's framework), Bayesian model selection, and algebraic geometry applications to neural networks.

## Background: Singular Learning Theory

Neural networks are *singular* statistical models: their Fisher information matrix $I(\theta)$ is not positive-definite at the true parameter $\theta^*$. Classical BIC/AIC theory breaks down in this setting. Watanabe's singular learning theory replaces the parameter dimension $d/2$ in BIC with the **learning coefficient** (RLCT) $\lambda$:

```math
\begin{align}
\text{WBIC} \approx nE_n[L_n] + 2\lambda \log n
\end{align}
```

where $n$ is sample size, $E_n[L_n]$ is the Bayes posterior average log-loss, and $\lambda$ governs generalization error via:

```math
\begin{align}
E[\text{generalization error}] = \frac{\lambda}{n} + O\!\left(\frac{1}{n^2}\right)
\end{align}
```

The **resolution of singularities** (Hironaka 1964) guarantees that there exists a proper analytic map $g: U \to W$ such that:

```math
\begin{align}
K(g(u)) &= u_1^{2k_1} u_2^{2k_2} \cdots u_d^{2k_d} \cdot a(u)^2 \\
|g'(u)| &= |b(u)| \cdot u_1^{h_1} u_2^{h_2} \cdots u_d^{h_d}
\end{align}
```

where $K(\theta)$ is the KL divergence from the true distribution $q(x)$ to the model $p(x|\theta)$, and $a(u)$, $b(u)$ are nonzero analytic functions locally (normal crossing form).

**Definition (Learning Coefficient):** The local RLCT at point $P \in \Theta^*$ (the set of true parameters) is:

```math
\begin{align}
\lambda_P := \inf_{Q \in U_P} \left\{ \min_{i} \frac{h_i^{(Q)} + 1}{2k_i^{(Q)}} \right\}
\end{align}
```

and the global learning coefficient is $\lambda = \inf_{P \in \Theta^*} \lambda_P$.

> [!NOTE]
> The multiplicity $\mathfrak{m}$ counts how many coordinate indices achieve the infimum in the expression above. It appears as the exponent in the subdominant term of the Bayes free energy: $F_n = n\lambda_0 + (\mathfrak{m}-1)\lambda \log\log n + O(1)$.

## Main Theorem

**Setup:** Separate the parameters into three groups:
- $\theta \in \mathbb{R}^r$: "regular" parameters (true value $\theta^* = 0$)
- $a \in \mathbb{R}^\alpha$: outer-layer parameters
- $b \in \mathbb{R}^\beta$: inner-layer parameters

The log-likelihood ratio at the realization point $P = (0,0,0)$ admits a Taylor expansion:

```math
\begin{align}
f(X|\theta=0, a, b) = \sum_{s=1}^{\gamma}\sum_{n=1}^{n_s} g_{s,n}(a,b)\, Z_{s,n} + \sum_{s=1}^{\infty} h_s(a,b)\, W_s
\end{align}
```

where $1 \leq m_1 < m_2 < \cdots < m_\gamma$ are the indices at which $\sigma^{(m_s)}(0) \neq 0$ (nonzero Taylor coefficients of the activation function), $g_{s,n}$ are analytic functions with lowest-degree $m_s$ in $b$, and $Z_{s,n}$, $W_s$ are random variables depending on the input $X$.

**Counting rule interpretation (Remark 2.1):** Imagine $\gamma$ shelves where shelf $s$ holds $n_s$ items each priced $m_s$. With total demand $\alpha$ and total budget $\beta$:

- Buy items shelf by shelf (cheapest first).
- $L$ = the shelf at which cumulative demand reaches $\alpha$.
- $n_s^*$ = items actually bought from shelf $s$ ($n_s$ for $s < L$; remainder for $s = L$).
- $K$ = last shelf fully purchased within budget $\beta$.

Formally:

```math
\begin{align}
L &:= \begin{cases}
\gamma & \text{if } \sum_{s=1}^{\gamma} n_s < \alpha \\
\min\!\left\{l \;\middle|\; \sum_{s=1}^{l} n_s \geq \alpha\right\} & \text{otherwise}
\end{cases} \\[6pt]
n_s^* &:= \begin{cases}
n_s & s < L \\
\min\!\left\{n_L,\, \alpha - \sum_{s=1}^{L-1} n_s\right\} & s = L
\end{cases} \\[6pt]
K &:= \max\!\left\{k=0,\ldots,L \;\middle|\; \sum_{s=1}^{k} m_s n_s^* \leq \beta\right\}
\end{align}
```

**Theorem (Main Result):** Under conditions (i)–(iv) (detailed below), the local RLCT satisfies:

```math
\begin{align}
\lambda_P \leq \frac{r}{2} + \begin{cases}
\dfrac{\beta}{2m_1} & K = 0 \\[8pt]
\dfrac{\beta + \sum_{s=1}^{K}(m_{K+1} - m_s)\,n_s^*}{2m_{K+1}} & 1 \leq K \leq L-1 \\[8pt]
\dfrac{\sum_{s=1}^{L} n_s^*}{2} & K = L
\end{cases}
\end{align}
```

**Multiplicity:** $\mathfrak{m} = 2$ if $\beta = \sum_{s=1}^{k} m_s n_s^*$ for some $k$; otherwise $\mathfrak{m} = 1$.

**Conditions (i)–(iv):** Let $\bar{g}_{s,n}(a,b)$ denote the degree-$m_s$ part of $g_{s,n}$ in $b$.

- **(i)** All $g_{s,n}$ vanish when $a=0$: captures that excess units contribute zero at the realization point.
- **(ii)** There exists $b \neq 0$ such that the Jacobian $\partial \bar{g}/\partial a|_{a=0}$ has rank $\sum_{s=1}^{L} n_s^*$: a transversality condition ensuring the coordinate change in Step 2 of the proof is valid.
- **(iii)** The functions $Z_{1,1},\ldots,Z_{L,n_L}$ are linearly independent of each other and of $\partial f/\partial \theta_i$: excludes collinearity between hidden units.
- **(iv)** For all $s$, $h_s(a,b) \in (g_{s,n})^2$: the remainder terms in the Taylor expansion are dominated by the leading terms.

> [!IMPORTANT]
> Condition (iii) can fail for $\sigma(x) = x/(1+e^{-x})$ (swish) when hidden units have symmetric weights $b_i = -b_j$, because $\sigma'(x) + \sigma'(-x) \equiv 1$ makes certain Vandermonde-type matrices singular. This is a known exception handled in Appendix A.

## Proof Strategy (Four-Step Blow-Up Procedure)

The proof constructs an explicit resolution of singularities via a sequence of coordinate changes (blow-ups):

1. **Step 1 ($m_1$ blow-ups at $\{(\theta, b)=0\}$):** Substitute $\theta_j \to t\theta_j'$, $b_k \to t b_k'$, introducing a new coordinate $t$. After $m_1$ such blow-ups, the KL divergence $K(\theta)$ picks up a factor $t^{2m_1}$, giving an intermediate bound:

```math
\begin{align}
\lambda_P \leq \frac{r}{2} + \frac{\beta}{2m_1}
\end{align}
```

2. **Step 2 (Coordinate change $a \mapsto a'$):** By condition (ii), a diffeomorphism $a \mapsto a'$ exists that "straightens" the leading parts of $g_{s,n}$ with respect to $a$, making subsequent blow-ups tractable.

3. **Step 3 ($L-1$ stages of blow-ups using $a'$):** At stage $k$, perform $m_{k+1} - m_k$ blow-ups. Each stage progressively refines the bound using the $k$-th shelf of the counting rule. The bound after stage $k$ is:

```math
\begin{align}
\frac{r}{2} + \min\!\left\{\frac{\beta + \sum_{s=1}^{k-1}(m_k - m_s)n_s^*}{2m_k},\; \frac{\beta + \sum_{s=1}^{k}(m_{K+1}-m_s)n_s^*}{2m_{K+1}}\right\}
\end{align}
```

4. **Step 4 (Final blow-up on $(\theta, a)$):** One additional blow-up gives the bound $r/2 + \sum_{s=1}^{L} n_s^*/2$. Combining all stages yields the main theorem.

## Application to Three-Layer Neural Networks

**Model architecture:**

- Input: $X \in \mathbb{R}^N$
- Hidden layer: $H$ units with inner weights $B \in \mathbb{R}^{H \times N}$ and activation $\sigma$
- Output: $Y \in \mathbb{R}^M$ with outer weights $A \in \mathbb{R}^{M \times H}$
- Noise: $\mathcal{N}(0, I_M)$

```math
\begin{align}
Y = A\,\sigma(BX) + \mathcal{N}
\end{align}
```

True distribution uses $H^* < H$ hidden units ($A^* \in \mathbb{R}^{M \times H^*}$, $B^* \in \mathbb{R}^{H^* \times N}$). **Assumption 1 (minimality):** Every column of $A^*$ is nonzero and $\{\sigma(b_i^{*T}X)\}_{i=1}^{H^*}$ are linearly independent.

### Point $P_1$: Redundant Hidden Units Set to Zero

Realization: matched units $i \leq H^*$ take their true values; excess units $i > H^*$ have $a_{\cdot,i} = 0$, $b_i = 0$.

Parameters: $r = (M+N)H^*$, $\alpha = M(H-H^*)$, $\beta = N(H-H^*)$.

The nonzero derivative indices of $\sigma$ at 0 give $n_s = M\binom{m_s + N - 1}{m_s}$ for each $s$.

**Upper bound at $P_1$:**

```math
\begin{align}
\lambda_{P_1} \leq \frac{(M+N)H^*}{2} + \begin{cases}
\dfrac{N(H-H^*)}{2m_1} & K=0 \\[8pt]
\dfrac{N(H-H^*) + M\sum_{s=1}^{K}\binom{m_s+N-1}{m_s}(m_{K+1}-m_s)}{2m_{K+1}} & 1 \leq K \leq L-1 \\[8pt]
\dfrac{M(H-H^*)}{2} & K=L
\end{cases}
\end{align}
```

### Point $P_2$: Weight-Sharing (Duplicated Hidden Units)

Realization: same as $P_1$ for matched units; excess units share the last matched unit's input weight $b_i = b_{H^*}^*$ for $i > H^*$.

Parameters: $r = (M+N)H^*$, $\alpha = M(H-H^*)$, $\beta = H - H^*$ (scalar, not $N$-dimensional), and $m_s = s$, $n_s = M\binom{s+N-1}{s}$.

**Upper bound at $P_2$:**

```math
\begin{align}
\lambda_{P_2} \leq \frac{(M+N)H^*}{2} + \begin{cases}
\dfrac{N(H-H^*)}{2} & K=0 \\[8pt]
\dfrac{N(H-H^*-K) + M\!\left(\binom{N+K+1}{K} - (K+1)\right)}{2(K+1)} & 1 \leq K \leq L-1 \\[8pt]
\dfrac{M(H-H^*)}{2} & K=L
\end{cases}
\end{align}
```

> [!NOTE]
> The key difference between $P_1$ and $P_2$ is in $\beta$: at $P_1$ each excess hidden unit contributes $N$ inner-weight degrees of freedom, while at $P_2$ they share a single set of inner weights (only their scalar multipliers vary), giving $\beta = H - H^*$ instead of $N(H-H^*)$.

### Activation-Specific Results ($N=1$)

The nonzero derivative indices of common activation functions at 0 determine the $m_s$ sequence:

| Activation function | Nonzero derivative indices $\{m_s\}$ |
|---|---|
| $e^x - 1$, $xe^x$, $x\tanh(\log(1+e^x))$ (gelu-like) | $1, 2, 3, 4, 5, \ldots$ |
| $x/(1+e^{-x})$ (swish), $x\Phi(x)$ | $1, 2, 4, 6, 8, \ldots$ |
| $\tanh(x)$, $\sin(x)$, $\arctan(x)$ | $1, 3, 5, 7, 9, \ldots$ |

For **swish-type** ($m_s = 1,2,4,6,8,\ldots$), $N=1$: $\lambda_{P_2} \leq \lambda_{P_1}$ when $H - H^* \leq 3M+8$ (for $M \geq 4$), meaning weight-sharing produces a *harder* singularity (smaller RLCT, worse generalization).

For **tanh-type** ($m_s = 1,3,5,7,9,\ldots$), $N=1$: $\lambda_{P_2} \leq \lambda_{P_1}$ when $H - H^* \leq M+3$ (for $M \geq 3$).

### Polynomial Activations

**Reduced-rank regression ($\sigma(x) = x$):** The three-layer linear network collapses to a matrix factorization model with $R = \text{rank}(A^*B^*) < \min\{M,N,H\}$.

Parameters: $r = R(M+N-R)$, $\alpha = (M-R)(H-R)$, $\beta = (N-R)(H-R)$, $\gamma=1$, $m_1=1$, $n_1 = (M-R)(N-R)$.

**Upper bound:**

```math
\begin{align}
\lambda_{P_1} \leq \begin{cases}
\dfrac{HN - HR + MR}{2} & N < H < M \text{ or } H \leq N < M \\[6pt]
\dfrac{HM - HR + NR}{2} & N \geq M \text{ and } N \geq H \\[6pt]
\dfrac{MN}{2} & H \geq M \text{ and } N < H
\end{cases}
\end{align}
```

This matches Aoyagi's (2005) exact formula in 3 of 4 parameter regimes; the remaining case may be a strict inequality.

## Detailed Example: tanh Network with $N=M=1$, $H=4$, $H^*=1$

**True model:** $Y = \tanh(X) + \mathcal{N}$. The fitted model has three excess hidden units:

```math
\begin{align}
Y = (\theta_1+1)\tanh((\theta_2+1)X) + a_1\tanh(b_1 X) + a_2\tanh(b_2 X) + a_3\tanh(b_3 X) + \mathcal{N}
\end{align}
```

At the realization point $P = (0,0,0)$: $r=2$, $\alpha=3$, $\beta=3$, and for tanh, $m_s = 2s-1$ ($s=1,2,3,\ldots$), giving $n_1 = n_2 = n_3 = 1$.

Counting rule: $L = \min\{l \mid l \geq \alpha=3\} = 3$; $K = \max\{k \mid \sum_{s=1}^k m_s \leq \beta=3\} = 1$ (since $m_1=1 \leq 3$ but $m_1+m_2=4>3$).

**Upper bound:**

```math
\begin{align}
\lambda_P \leq \frac{2}{2} + \frac{3 + (m_2 - m_1)\cdot 1}{2m_2} = 1 + \frac{3 + 2}{6} = 1 + \frac{5}{6} = \frac{11}{6}
\end{align}
```

**Proof trace (Steps 1–4):**

1. Blow-up centered at $\{(\theta,b)=0\}$ with $t = \theta_t$: initial bound $2/2 + 3/2 = 5/2$.
2. Coordinate change $a \mapsto a'$ using the Vandermonde structure of $\tanh$'s odd-order derivatives.
3. Two additional blow-ups on $a'_2$, $b_1$: bound refined from $5/2$ toward $11/6$.
4. Final blow-up on $(\theta', a')$: achieves normal crossing form, confirming $\lambda_P \leq 11/6$, $\mathfrak{m} = 1$.

## Comparison with Prior Work

| Aspect | Kurumadani (2025, semiregular) | This paper (2026) |
|---|---|---|
| Points covered | Non-singular $\Theta^*$ points only | All points, including singular |
| Activation functions | General real analytic | Same (swish, polynomial, tanh, etc.) |
| Result type | Exact formula | Upper bound (tight at $N=1$) |
| Reduced-rank regression | Not covered | 3 of 4 cases match Aoyagi (2005) |
| Input dimension | Any $N$ | Tight only at $N=1$ |

> [!TIP]
> Watanabe's *Algebraic Geometry and Statistical Learning Theory* (2009) is the foundational reference for singular learning theory. Lau et al. (2025, AISTATS) introduced the "local learning coefficient" framing used here.

> [!CAUTION]
> This paper provides upper bounds, not exact values. For $N > 1$, the bounds may not be tight. The author explicitly flags that the case $N < H < M$ or $H \leq N < M$ for reduced-rank regression may be a strict inequality.

# Experiments

- Dataset: None — this is a purely theoretical paper with no empirical evaluation.
- Hardware: N/A
- Optimizer: N/A
- Results: Mathematical proof that the local RLCT satisfies the stated upper bounds for three-layer neural networks with general analytic activations. The example ($N=M=1$, $H=4$, $H^*=1$, tanh) gives $\lambda_P \leq 11/6$ with $\mathfrak{m}=1$.
