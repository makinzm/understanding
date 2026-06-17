# Meta Information

- URL: [Universal Decision Learners](https://arxiv.org/abs/2605.30694)
- LICENSE: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Mahadevan, S. (2026). Universal Decision Learners. arXiv:2605.30694 [cs.LG].

# Universal Decision Learners

## Overview

Universal Decision Learners (UDL) is a categorical framework that unifies planning, reinforcement learning, causal inference, online learning, and game theory under one mathematical structure. The central thesis is that **learning to make decisions is the problem of canonically extending partial decision data to new contexts**. Rather than proposing a new algorithm, the paper identifies a shared semantic structure—Kan extensions—that underlies all these decision-making formalisms.

The framework is applicable to researchers and practitioners who want to understand *why* disparate decision algorithms share structural properties, and to theoreticians building principled bridges between RL, causality, and game theory.

> [!NOTE]
> The paper is purely theoretical and does not introduce new empirical benchmarks or neural architectures. Its contribution is a unifying semantic language for existing methods.

---

## Background: Category Theory Preliminaries

The paper relies on standard category theory. A **category** $\mathcal{C}$ consists of objects and morphisms (arrows) between them. A **functor** $F: \mathcal{C} \to \mathcal{D}$ maps objects and morphisms structure-preservingly between categories.

**Kan extensions** are the central tool. Given functors $F: \mathcal{C} \to \mathcal{D}$ and $J: \mathcal{C} \to \mathcal{E}$, the Kan extensions extend $F$ along $J$:

- **Left Kan extension** $\text{Lan}_J F: \mathcal{E} \to \mathcal{D}$: the *best* (initial) approximation of $F$ via $J$ using colimits (aggregation, rollout, interpolation).
- **Right Kan extension** $\text{Ran}_J F: \mathcal{E} \to \mathcal{D}$: the *best* (terminal) approximation of $F$ via $J$ using limits (constraint satisfaction, fixed-point semantics).

Formally, both are characterized by universal natural transformations:

```math
\begin{align}
  \eta: \text{Lan}_J F \circ J \Rightarrow F \quad \text{(left)} \\
  \epsilon: F \Rightarrow \text{Ran}_J F \circ J \quad \text{(right)}
\end{align}
```

The categories $\mathcal{E}$ are required to have sufficient limits and colimits, and may be enriched over semirings:
- **max-plus semiring**: standard RL with discounted returns
- **min-plus semiring**: shortest-path planning
- **Boolean semiring**: reachability queries

---

## Universal Decision Learner (UDL) Construction

**Input:** A local behavioral model $F: \mathcal{C} \to \mathcal{D}$ defined on a small category $\mathcal{C}$ (local states/actions), and an embedding functor $J: \mathcal{C} \to \mathcal{E}$ into a larger category $\mathcal{E}$ (global state/context space).

**Output:** A global decision model $\text{UDL}_J(F): \mathcal{E} \to \mathcal{D}$ that canonically extends $F$ to the full context space.

The UDL is defined as the composite:

```math
\begin{align}
  \text{UDL}_J(F) = \text{Ran}_J(\text{Lan}_J F)
\end{align}
```

**Computation steps:**
1. Compute $\text{Lan}_J F$: aggregate local data into candidate global models via colimits (rollout/interpolation step).
2. Compute $\text{Ran}_J(\text{Lan}_J F)$: enforce global consistency constraints via limits (fixed-point/Bellman step).

The left Kan extension produces *all plausible completions* of local data; the right Kan extension then selects the *most consistent* global model among them.

---

## Universal Comparison Theorems

**Theorem 7 (Initiality of Left Kan):** $\text{Lan}_J F$ is initial in the category of all global models $G: \mathcal{E} \to \mathcal{D}$ that factor through the local data. Any such $G$ admits a unique natural transformation $\text{Lan}_J F \Rightarrow G$.

**Theorem 8 (Terminality of Right Kan):** $\text{Ran}_J F$ is terminal in the category of all global models compatible with local constraints. Any such $G$ admits a unique natural transformation $G \Rightarrow \text{Ran}_J F$.

**Corollary 9 (Factorization):** Any decision procedure that computes identical semantics to the UDL factors uniquely through $\text{UDL}_J(F)$. This means the UDL is the *canonical* decision structure—any correct algorithm is a specialization of it.

> [!IMPORTANT]
> These theorems imply that algorithms like value iteration, Q-learning, and TD-learning are all *approximations* of the same universal construction (right Kan computation), not fundamentally different approaches.

---

## Behavioral Equivalence via Kan Bisimulation

**Definition (Kan Bisimulation):** Two decision models $F, G: \mathcal{C} \to \mathcal{D}$ are **Kan bisimilar** if their induced Kan extensions $\text{Lan}_J F \cong \text{Lan}_J G$ (or $\text{Ran}_J F \cong \text{Ran}_J G$) are naturally isomorphic, regardless of their syntactic presentation.

This generalizes classical MDP bisimulation to a universal categorical principle. Two agents with different state spaces but isomorphic Kan extensions behave identically on all reachable inputs.

**Theorem 15 (Minimal Abstraction):** The minimal state abstraction preserving Kan-extended semantics is the coarsest quotient of the state space under Kan bisimulation. That is, abstraction is *semantic* rather than syntactic—the finest lossless compression is the one that preserves Kan structure.

---

## Applications to Specific Decision Domains

### Reinforcement Learning

MDPs are formalized as coalgebras encoding transition dynamics $p(s' | s, a)$ and reward $r(s, a)$. 

- **Value functions** $V^\pi: \mathcal{S} \to \mathbb{R}$ are right Kan extensions of one-step reward/transition data along the embedding $J: \mathcal{S}_\text{local} \to \mathcal{S}$.
- **Bellman equations** express the right Kan consistency condition:

```math
\begin{align}
  V^\pi(s) = r(s, \pi(s)) + \gamma \sum_{s'} p(s'|s,\pi(s)) V^\pi(s')
\end{align}
```

- **Value iteration** is an iterative approximation of the right Kan fixed point.
- **Q-learning** and **TD-learning** are stochastic approximate Kan computations.

### Planning

Left Kan extensions aggregate path fragments (local trajectories) into globally rollout candidates. A planning algorithm that composes local transition models into a full trajectory plan is computing a left Kan extension.

### Game Theory

**Nash equilibrium** is a right Kan fixed point: each player's strategy $\sigma_i$ is the terminal extension of their local best-response data to the full strategy profile space. The fixed-point condition $\sigma_i = \text{BR}_i(\sigma_{-i})$ is a right Kan consistency equation.

### Online Learning

Regret minimization is approximation of Kan consistency over time. An online learning algorithm that converges to a best-response policy is computing an approximate right Kan extension over the sequence of observed data points.

### Causal Inference

Identifiability of causal effects is characterized as Kan invariance of interventional semantics: the do-calculus expression $P(Y | do(X))$ is identifiable if and only if the Kan-extended interventional model is invariant to syntactic presentations of the causal graph.

---

## Homotopy Kan Equivalence

**Definition 16 (Homotopy Kan Equivalence):** Two models $F$ and $G$ are **homotopy Kan equivalent** if there exists a sequence of intermediate models connected by natural transformations (a homotopy), making them equivalent *up to deformation* rather than strict isomorphism.

This relaxation is important for:
- **Representation learning**: learned embeddings may not be strictly isomorphic but are equivalent up to continuous deformation.
- **Causal models**: structurally different DAGs may encode the same interventional distributions up to homotopy.
- **Approximate RL**: neural value functions are exact Kan extensions only in the limit; homotopy equivalence captures the finite-sample regime.

---

## Function Approximation and Neural RL (Appendix B)

Neural RL is interpreted within the UDL framework:

- **Approximate right Kan computation**: projected Bellman equations (e.g., in DQN, actor-critic) compute approximate right Kan extensions onto the function approximation class $\Theta$.
- **Learned representations**: embedding networks $\phi: \mathcal{S} \to \mathbb{R}^d$ are approximate coalgebra morphisms; good representations are those where the Kan extension on the embedded space $\mathbb{R}^d$ approximates the true value Kan extension.
- **Structured losses**: losses like bisimulation metric losses or successor representation losses enforce *diagrammatic coherence*—they penalize deviation from the Kan commutativity conditions alongside value consistency.

The projected Bellman equation for function approximation with parameters $\theta \in \mathbb{R}^p$:

```math
\begin{align}
  \theta^* = \arg\min_\theta \| V_\theta - \Pi \mathcal{T}^\pi V_\theta \|^2_\mu
\end{align}
```

where $\Pi$ is projection onto the approximation class and $\mathcal{T}^\pi$ is the Bellman operator, is reinterpreted as approximate right Kan computation under the projection.

---

## Comparison with Related Work

| Approach | How It Relates to UDL |
|---|---|
| MDP bisimulation (Ferns et al.) | Special case of Kan bisimulation for finite MDPs |
| Successor representations (Dayan) | Approximate left Kan extension of one-step transitions |
| Deep Q-Networks (DQN) | Approximate right Kan computation with neural function class |
| Causal do-calculus (Pearl) | Identifiability as Kan invariance of interventional functors |
| Regret bounds (online learning) | Deviations from right Kan consistency measured as regret |
| Nash equilibrium (game theory) | Terminal object of right Kan fixed-point category |

The key difference from prior unification attempts (e.g., reward machines, options framework) is that UDL operates at the level of *semantic structure* rather than algorithmic structure—it unifies the *problem* each algorithm solves, not the algorithms themselves.

---

# Experiments

- **Dataset:** None. The paper is purely theoretical with no empirical evaluation.
- **Hardware:** Not applicable.
- **Optimizer:** Not applicable.
- **Results:** The paper establishes formal theorems (Theorems 7, 8, 15, Definition 16, Corollary 9) and shows analytical connections between existing algorithms and the UDL framework. No quantitative benchmarks are reported.

> [!NOTE]
> The authors explicitly acknowledge the lack of empirical results as a limitation and suggest three future research directions: (1) comparing algorithms by their target Kan extensions, (2) studying representation learning via Kan invariance, and (3) extending the framework to richer decision processes (POMDPs, multi-agent settings, continuous time).
