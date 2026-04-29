# Meta Information

- URL: [Reinforcement Learning in Categorical Cybernetics](https://arxiv.org/abs/2404.02688)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hedges, J., & Rodríguez Sakamoto, R. (2024). Reinforcement Learning in Categorical Cybernetics. arXiv:2404.02688.

# Reinforcement Learning in Categorical Cybernetics

This paper shows that major families of reinforcement learning (RL) algorithms — dynamic programming, Monte Carlo, temporal difference learning, and deep RL — can all be embedded into a single categorical framework called **categorical cybernetics**, which formalises bidirectional processes using parametrised optics. The audience is researchers in category theory and ML theory who want a unified, compositional account of RL.

## Markov Decision Processes

A **Markov process (MP)** consists of a state set $S$ and a stochastic transition function $t : S \to \mathcal{D}S$, where $\mathcal{D}$ denotes the finite-support probability monad over **Set**.

A **Markov reward process (MRP)** augments an MP with a utility function $u : S \to \mathcal{D}\mathbb{R}$ that generates stochastic rewards.

A **Markov decision process (MDP)** further adds an action set $A$, making the joint transition-utility function $\langle t, u \rangle : S \times A \to \mathcal{D}(S) \times \mathcal{D}(\mathbb{R})$.

A **policy** $\pi : S \to TA$ maps states to actions (or distributions over actions), where $T$ is:
- the identity functor (deterministic policy),
- $\mathcal{D}$ (stochastic policy), or
- $\mathcal{P}$ (non-deterministic policy).

The agent's objective is to maximise the expected discounted return $\sum_i \gamma^i r_i$ with discount factor $\gamma \in (0, 1]$.

**Value functions:**
- **State value function** $V : S \to \mathbb{R}$: expected long-run reward from state $s$ following policy $\pi$.
- **State-action value function** $Q : S \times A \to \mathbb{R}$: expected long-run reward of taking action $a$ in state $s$ and then following $\pi$.

## Bellman Operators

The **Bellman value operator** $\mathfrak{B}_\text{val} : \mathbb{R}^S \times A^S \to \mathbb{R}^S$ performs value improvement:

```math
\begin{align}
  V(s) \leftarrow \sum_{a \in A} \pi(s, a) \sum_{s' \in S,\, r \in \mathbb{R}} t(s, a, s', r)\,(r + \gamma V(s'))
\end{align}
```

The **Bellman policy operator** $\mathfrak{B}_\text{pol} : \mathbb{R}^S \to A^S$ performs policy improvement (greedy update):

```math
\begin{align}
  \pi(s) \leftarrow \operatorname{argmax}_{a} \mathbb{E}[r + \gamma V(s')]
\end{align}
```

When $S$ is finite and $0 < \gamma < 1$, the policy-conditioned Bellman operator $\mathfrak{B}^\pi$ is a contraction mapping under the supremum metric, guaranteeing convergence to a fixed point.

Using lifted embeddings $\bar{\mathfrak{B}}_\text{val}(V, \pi) = (\mathfrak{B}_\text{val}(V, \pi), \pi)$ and $\bar{\mathfrak{B}}_\text{pol}(V, \pi) = (V, \mathfrak{B}_\text{pol}(V))$, the classic dynamic programming algorithms are expressed as approximate fixpoints $(-)^\dagger$:

| Algorithm | Expression |
|---|---|
| Policy Iteration (PIT) | $(\bar{\mathfrak{B}}_\text{pol} \circ \bar{\mathfrak{B}}_\text{val}^\dagger)^\dagger$ |
| Value Iteration (VIT) | $(\bar{\mathfrak{B}}_\text{pol} \circ \bar{\mathfrak{B}}_\text{val})^\dagger$ |
| Generalized Policy Iteration (GPI) | $(\bar{\mathfrak{B}}_\text{pol}^m \circ \bar{\mathfrak{B}}_\text{val}^n)^\dagger$, $m, n > 0$ |

## Categorical Cybernetics Background

### Actegories

An **actegory** is a functor $(\cdot) : \mathcal{M} \times \mathcal{C} \to \mathcal{C}$ where $\mathcal{M}$ is a monoidal category acting on $\mathcal{C}$, with coherent isomorphisms:

```math
\begin{align}
  I \cdot X &\cong X \\
  (M \otimes N) \cdot X &\cong M \cdot (N \cdot X)
\end{align}
```

A **symmetric actegory** is induced by a strong monoidal functor $F : \mathcal{M} \to \mathcal{C}$ via $M \cdot X = F(M) \otimes X$.

### Parametrised Morphisms

Given actegory $(\cdot) : \mathcal{M} \times \mathcal{C} \to \mathcal{C}$, a **parametrised morphism** $f : X \to Y$ is a pair $(M, f)$ where $M \in \mathcal{M}$ is the **parameter object** and $f : M \cdot X \to Y$ is a morphism in $\mathcal{C}$.

- **Identity:** $(I, \text{id}_X : I \cdot X \cong X \to X)$
- **Composition** of $(M, f : M \cdot X \to Y)$ followed by $(N, g : N \cdot Y \to Z)$:

```math
\begin{align}
  (N \otimes M) \cdot X \cong N \cdot (M \cdot X) \xrightarrow{N \cdot f} N \cdot Y \xrightarrow{g} Z
\end{align}
```

- **Reparametrisation** (2-cell): a morphism $h : M \to N$ such that $f = g \circ (h \cdot X)$.

These data form the **bicategory of parametrised morphisms** $\mathbf{Para}_\mathcal{M}(\mathcal{C})$. The dual construction yields **coparametrised morphisms** $(M, f : X \to M \cdot Y)$, forming $\mathbf{CoPara}_\cdot(\mathcal{C})$.

**Weighted parametrisation:** Given a symmetric lax monoidal functor $W : \mathcal{M} \to \mathbf{Set}$, the category of elements $\int W$ extends the actegory by $(M, w) \cdot X = M \cdot X$, yielding $\mathbf{Para}^W_\mathcal{M}(\mathcal{C}) \coloneqq \mathbf{Para}_{\int W}(\mathcal{C})$.

### Mixed Optics

Given $\mathcal{M}$ acting on both $\mathcal{C}$ and $\mathcal{D}$, a **mixed optic** $(X, X') \to (Y, Y')$ (with $X, X' \in \mathcal{C}$ and $Y, Y' \in \mathcal{D}$) is an equivalence class of triples $(M, f, f')$:
- $M \in \mathcal{M}$ (the **residual**),
- $f : X \to M \cdot Y$ (forward pass, a coparametrised morphism in $\mathcal{C}$),
- $f' : M \cdot Y' \to X'$ (backward pass, a parametrised morphism in $\mathcal{D}$).

Equivalence is given by the coend:

```math
\begin{align}
  \mathbf{Optic}_\mathcal{M}(\mathcal{C}, \mathcal{D})\big((X, X'), (Y, Y')\big)
  \;=\; \int^{M : \mathcal{M}} \mathcal{C}(X, M \cdot Y) \times \mathcal{D}(M \cdot Y', X')
\end{align}
```

**Self-optics** $\mathbf{Optic}(\mathcal{C})$ arise when $\mathcal{M} = \mathcal{C} = \mathcal{D}$ with the monoidal product action.

**Lenses** (cartesian case): via the ninja Yoneda lemma,
$\mathbf{Optic}((X, X'), (Y, Y')) \cong \mathcal{C}(X, Y) \times \mathcal{C}(X \times Y', X')$.

When the monoidal unit $I$ is terminal (e.g., in Markov categories), optics into $I$ become **continuations**:
$\mathbf{Optic}(\mathcal{C})((X, X'), I) \cong \mathcal{C}(X, X')$.

### Representable Continuation Functor

The **continuation functor** $\mathbb{K} = \mathbf{Optic}(\mathcal{C})(-, I) : \mathbf{Optic}(\mathcal{C})^\text{op} \to \mathbf{Set}$ maps optics to their continuation morphisms. This functor is central to representing Bellman operators.

### Iteration Functor

The **iteration functor** $\mathbb{I} : \mathbf{Optic}(\mathcal{C}) \to \mathbf{Set}$ (symmetric lax monoidal) assigns to each optic $(X, X')$ the set of **iterators**:

```math
\begin{align}
  \mathbb{I}(X, X') = \int^{M : \mathcal{C}} \mathcal{C}(I, M \otimes X) \times \mathcal{C}(M \otimes X', M \otimes X)
\end{align}
```

An element $(M, x_0, i) \in \mathbb{I}(X, X')$ consists of:
- **State space** $M$,
- **Initial state** $x_0 : I \to M \otimes X$,
- **Iterator** $i : M \otimes X' \to M \otimes X$.

For $\mathcal{C} = \mathbf{Set}$, the dinatural transformation $\langle{-}\mid{-}\rangle : \mathbb{K}(X, X') \times \mathbb{I}(X, X') \to X^\omega$ extracts the infinite sequence produced by iterating continuation $k : X \to X'$ with iterator $(M, m_0, x_0, i)$:

```math
\begin{align}
  \langle k \mid M, (m_0, x_0), i \rangle = x_0 : \langle k \mid M, i(m_0, k(x_0)), i \rangle
\end{align}
```

**Categorical cybernetics** is the study of $\mathbf{Para}(\mathbf{Optic})$, i.e., parametrised bidirectional processes. The **iterated optics category** $\mathbf{Optic}^\mathbb{I}(\mathcal{C}) \coloneqq \pi_0^*(\mathbf{Para}^\mathbb{I}(\mathbf{Optic}(\mathcal{C})))$ extends $\mathbf{Optic}(\mathcal{C})$ with iteration structure.

## Bellman Operators as Parametrised Optics

### Value Improvement as Optic Precomposition

The core prior result (on which this paper builds) shows the **Bellman value operator** factors through the continuation functor:

```math
\begin{align}
  \mathfrak{B}^\pi = \mathbb{K}(\ell^\pi)
\end{align}
```

where $\ell^\pi = (\ell^f, \ell^b) : (S, \mathbb{R}) \to (S, \mathbb{R})$ is a mixed optic in $\mathbf{Optic}_{\mathrm{Kl}(\mathcal{D})}(\mathrm{Kl}(\mathcal{D}), \mathrm{EM}(\mathcal{D}))$ with:
- **Forward pass** $\ell^f : S \to \mathcal{D}(S \times \mathbb{R})$, defined by $s \mapsto t(s, \pi(s))$ (sample next state and reward),
- **Backward pass** $\ell^b : \mathcal{D}(\mathbb{R}) \times \mathbb{R} \to \mathbb{R}$, defined by $(r, v) \mapsto \mathbb{E}[r] + \gamma v$ (compute Bellman target).

Here $\mathrm{Kl}(\mathcal{D})$ is the Kleisli category and $\mathrm{EM}(\mathcal{D})$ is the Eilenberg-Moore category (convex sets) of the probability monad $\mathcal{D}$.

A value function $V : S \to \mathbb{R}$ is represented as a costate (continuation) $V : (S, \mathbb{R}) \to (1, 1)$. Precomposing with $\ell^\pi$ yields the updated value function representing $\mathfrak{B}^\pi(V)$.

### Extension to Action-Value Functions

The paper's main contribution extends this to **parametrised optics** acting on $Q$-functions $Q : S \times A \to \mathbb{R}$. Rather than computing the full Bellman update, the Bellman operator is formulated as a **delta** (change) to the $Q$-matrix, which makes gradient-based methods (deep RL) a natural special case.

### Policy Improvement: Outside the Optic Image

> [!IMPORTANT]
> The **policy improvement operator** $\mathfrak{B}_\text{pol}$ does **not** arise as $\mathbb{K}$ applied to an optic. Policy improvement requires two coevaluation maps ($\lambda$), placing it outside the image of the continuation functor. This categorical distinction formalises why evaluation and improvement are structurally different steps in RL.

## RL Algorithms as Extremal Cases

All major RL algorithm families are embedded in the parametrised optics framework as different extremal cases along three axes: **model availability**, **sampling depth**, and **bootstrapping**.

### Dynamic Programming

- **Requirement:** Full model $\langle t, u \rangle$ is known.
- **Mechanism:** Bellman operator $\mathfrak{B}^\pi = \mathbb{K}(\ell^\pi)$ applied to $V : S \to \mathbb{R}$ (tabular, no sampling).
- **Algorithms:** Policy Iteration, Value Iteration, Generalized Policy Iteration.

### Monte Carlo Methods

- **Requirement:** No model; learns from complete sample trajectories.
- **Value target** (no bootstrapping):

```math
\begin{align}
  G = \sum_t \gamma^t r_t
\end{align}
```

- **Update** with learning rate $\alpha$ (averaging over returns):

```math
\begin{align}
  Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha G
\end{align}
```

- **Policy:** $\varepsilon$-greedy $\pi : S \to \mathcal{D}A$, uniform with probability $\varepsilon$, greedy otherwise.

### Temporal Difference Learning

Combines MC sampling with DP bootstrapping; the $n$-step return uses $n$ observed rewards plus a bootstrapped value estimate.

**SARSA** (1-step, on-policy): input $(s, a, r, s', a')$, target:

```math
\begin{align}
  G = r + \gamma Q(s', a')
\end{align}
```

**$n$-SARSA:** target $G = \sum_{t=1}^{n-1} \gamma^t r_t + \gamma^n Q(s_n, a_n)$.

**Q-Learning** (1-step, off-policy): separates the **behaviour policy** $\pi_\text{beh}$ (used for exploration) from the **target policy** $\pi_\text{tgt}(s') = \operatorname{argmax}_{a'} Q(s', a')$ (used only in updates):

```math
\begin{align}
  G = r + \gamma \max_{a' \in A} Q(s', a')
\end{align}
```

**Expected SARSA** (off-policy): uses the expectation under the target policy rather than a single sample:

```math
\begin{align}
  G = r + \gamma \mathbb{E}_{a \sim \pi_\text{tgt}(s')}[Q(s', a)]
\end{align}
```

### Deep Reinforcement Learning

When $S$ or $A$ is large or continuous, tabular $Q$-functions are intractable. Function approximation replaces the $Q$-table with a parametric model.

**Deep Q-Network (DQN):** parametrises $Q$ as a neural network with weights $\Theta$:

```math
\begin{align}
  \text{DQN} : \Theta \times S \to (A \to \mathbb{R})
\end{align}
```

The Bellman update becomes gradient descent on the squared Bellman error:

```math
\begin{align}
  \mathcal{L} &= (Q(s, a) - G)^2 \\
  Q &\leftarrow Q - \frac{\alpha}{2} \partial_{Q(s,a)} \mathcal{L}
\end{align}
```

**Policy Gradient (PG):** directly parametrises the policy distribution, bypassing the $Q$-function:

```math
\begin{align}
  \text{PG} : \Theta \times S \to \mathcal{D}A
\end{align}
```

DQN is a special case of PG via the Boltzmann (softmax) distribution over $Q$-values.

### Comparison Table

| Algorithm | Model? | Sampling | Bootstrap | Tabular/Deep | Categorical Representation |
|---|---|---|---|---|---|
| Value Iteration | Yes | None | Full | Tabular | $\mathbb{K}(\ell^\pi)$ on $V : S \to \mathbb{R}$ |
| Policy Iteration | Yes | None | Full | Tabular | Composition of $\mathfrak{B}_\text{pol}$, $\mathfrak{B}_\text{val}$ |
| Monte Carlo | No | Full trajectory | None | Tabular | Parametrised $\mathfrak{B}$ (no bootstrap) |
| SARSA | No | 1-step | 1-step | Tabular | Parametrised optic on $Q$ |
| Q-Learning | No | 1-step | Off-policy | Tabular | Parametrised optic on $Q$ |
| DQN | No | Batch | Gradient | Deep | Gradient delta on $Q$-network |
| Policy Gradient | No | Trajectory | Policy only | Deep | Direct action distribution |

## Unification Construction

The paper's central construction proceeds in three steps:

**Step 1 — Extend Bellman to parametrised optic on $Q$:** The standard Bellman operator $\mathfrak{B}$ is lifted to a parametrised optic that acts on action-value functions $Q : S \times A \to \mathbb{R}$ and depends on a sample parameter (the observed tuple $(s, a, r, s')$ or trajectory).

**Step 2 — Apply continuation functor:** Applying the representable contravariant functor $\mathbb{K}$ to the parametrised optic from Step 1 yields a parametrised function $\mathfrak{B}$ that implements Bellman iteration. The parameter now carries sampling and bootstrapping information.

**Step 3 — Compose as backward pass:** The parametrised function $\mathfrak{B}$ becomes the **backward pass** of a larger parametrised optic that models the full agent-environment interaction loop. The environment provides the forward pass (generating transitions); the agent's update rule is the backward pass.

> [!NOTE]
> "Parametrised optics appear in two different ways in our construction, with one becoming part of the other." This nesting — optics within optics — is what allows the framework to uniformly describe both tabular updates and gradient-based deep RL.

## Categorical Setting: Markov Categories

The probability monad interplay is handled in **representable Markov categories**, a class of symmetric monoidal categories suited for stochastic morphisms. The finite-support probability monad $\mathcal{D}$ is used throughout; convergence in more general settings (e.g., continuous state spaces) would require the **Kantorovich monad**, identified as future work.

**Dependent optics** (where the action set $A$ may depend on the current state) are deferred to future work, citing the framework of Vertechi.

## Relationship to Prior Work

| Framework | Connection |
|---|---|
| Categorical cybernetics (Hedges et al.) | This paper is a direct application of the optics + parametrisation machinery |
| Compositional game theory | RL as a special case of open games; policy improvement parallels Nash equilibrium |
| Deep learning in optics (Cruttwell et al.) | Same $\mathbf{Para}(\mathbf{Optic})$ structure for gradient-based learning |
| Compositional Bayesian inference | Variational EM also fits into the same framework |
| Myers/Spivak categorical systems theory | Complementary viewpoint using polynomial functors |

> [!TIP]
> The optics / $\mathbf{Para}(\mathbf{Optic})$ framework is also used to model deep learning (backpropagation as a parametrised lens), Bayesian inference, and game theory — making this RL embedding part of a broader unification programme.

# Experiments

- **Dataset:** None — this is a theoretical paper with no empirical experiments.
- **Hardware:** Not applicable.
- **Optimizer:** Not applicable.
- **Results:** The main results are mathematical theorems and constructions:
  - The Bellman value operator $\mathfrak{B}^\pi$ equals $\mathbb{K}(\ell^\pi)$ for a specific mixed optic $\ell^\pi$ in $\mathbf{Optic}_{\mathrm{Kl}(\mathcal{D})}(\mathrm{Kl}(\mathcal{D}), \mathrm{EM}(\mathcal{D}))$.
  - Dynamic programming, Monte Carlo, TD learning, and deep RL algorithms are all special cases of a single parametrised optic construction.
  - Policy improvement is shown to be categorically distinct from value improvement, lying outside the image of the continuation functor $\mathbb{K}$.
