# Meta Information

- URL: [Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://arxiv.org/abs/2508.17536)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Choi, H. K., Zhu, X., & Li, Y. (2025). Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?. University of Wisconsin-Madison.

# Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?

## Abstract

Multi-Agent Debate (MAD) has emerged as a promising paradigm for improving LLM performance through collaborative reasoning. Despite recent advances, the key factors driving MAD's effectiveness remain unclear. This work disentangles MAD into two components — majority voting and inter-agent debate — and measures each contribution independently. Through experiments across seven NLP benchmarks, the authors find that majority voting alone accounts for most of the performance gains typically attributed to MAD. A theoretical framework modeling debate as a stochastic process proves that MAD induces a **martingale** over agents' belief trajectories, implying debate alone does not improve expected correctness. Guided by these insights, targeted interventions that bias belief updates toward correct signals can meaningfully enhance debate effectiveness.

## 1. Introduction

Multi-Agent Debate (MAD) frames multiple LLM instances as independent agents who iteratively exchange and refine answers. The key open question is: do agents benefit from *meaningful communication* with each other, or is the gain simply from aggregating multiple independently-sampled outputs? This paper cleanly separates the two effects to answer that question empirically and theoretically.

The core contributions are:

1. An empirical demonstration that majority voting matches or exceeds standard MAD on 7 benchmarks using open-source 7B–8B models.
2. A probabilistic framework (Dirichlet-Compound-Multinomial) that formally proves debate yields a martingale belief process.
3. Practical interventions (MAD-Conformist, MAD-Follower) that outperform vanilla MAD by biasing updates toward the majority.

## 2. Preliminaries

### 2.1 Notation and Setup

- Input space: $\mathcal{X}$ (questions), output space: $\mathcal{Y}$ (candidate answers, $K$ classes).
- $N$ agents with stochastic generation functions $f_i : \mathcal{X} \to \mathcal{Y}$.
- Initial responses at round $t=0$: $y_{i,0} \sim f_i(x)$.
- Communication topology: undirected graph $G = (V, E)$ where neighbors of agent $i$ are $\mathcal{N}(i)$.

### 2.2 Majority Voting (MV)

All agents generate responses independently in a single round $t=0$. The final answer is the mode of all responses:

```math
\begin{align}
  y_{\text{mv}} = \arg\max_{k \in \mathcal{Y}} \left| \{ i : y_{i,0} = k \} \right|
\end{align}
```

### 2.3 Multi-Agent Debate (MAD)

Agents iterate for $T$ rounds. At each round $t$, agent $i$ observes responses from its neighbors and updates its own response:

```math
\begin{align}
  y_{i,t} = D\!\left(x;\, \mathcal{R}_i^{(t)}\right), \quad \mathcal{R}_i^{(t)} = \{ y_{j,t-1} \mid j \in \mathcal{N}(i) \}
\end{align}
```

where $D$ is the debate update function (the LLM conditioned on neighbors' prior responses). Three topology variants are tested:

| Variant | Communication Structure |
|---|---|
| Decentralized MAD | Every agent observes all other agents |
| Sparse MAD | Each agent observes only a subset of peers |
| Centralized MAD | One aggregator agent synthesizes all peers' responses |

## 3. Is Debate Really Necessary?

### 3.1 Experimental Setup

- **Models**: Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct.
- **Agents**: $N = 5$ in main experiments; ablations vary $N \in \{1, 2, 3, 4, 5\}$.
- **Debate rounds**: $T \in \{2, 3, 5\}$.
- **Benchmarks**: Arithmetics (100q), GSM8K (300q), MMLU-Pro.Med. (272q), MMLU-Formal Logic (126q), HellaSwag (300q), CommonsenseQA (300q), HH-RLHF (300q).
- **Extraction format**: agents output `{final answer: y}` to reduce parsing errors.

### 3.2 Key Observations

Majority voting (averaged across 7 benchmarks: **0.7691**) exceeds all MAD configurations for Qwen2.5-7B-Instruct. Decentralized MAD peaks at T=2 (0.7377) and degrades as $T$ increases, while voting is stable and requires no inter-agent communication cost.

> [!IMPORTANT]
> MAD performance degrades across rounds (T=2 → T=5) because agents increasingly converge to the initial majority regardless of correctness — a phenomenon consistent with the martingale theory derived later.

The ablation on agent count $N$ shows that both voting and MAD improve as $N$ increases, confirming the gains are primarily an ensemble effect.

## 4. Theoretical Analysis

### 4.1 Agent Response Generation via DCM

Each agent's response generation is modeled as a two-stage process using the **Dirichlet-Compound-Multinomial (DCM)** distribution. This captures both internal uncertainty (variable belief across runs) and stochastic sampling variability.

**Definition 1 (DCM generation at round $t$):**

1. Sample agent belief: $\theta_{i,t} \sim \text{Dirichlet}(\alpha_{i,t})$, where $\alpha_{i,t} \in \mathbb{R}^K_{>0}$ is the concentration vector over $K$ answer classes.
2. Sample response: $y_{i,t} \sim \text{Categorical}(\theta_{i,t})$.

The marginal probability of answer class $k$ is:

```math
\begin{align}
  P(y_{i,t} = k \mid \alpha_{i,t}) = \frac{\alpha_{i,t}^{(k)}}{\sum_{j=1}^K \alpha_{i,t}^{(j)}}
\end{align}
```

The mean belief in the correct answer (class 1) is $\bar{\theta}_{i,t}^{(1)} = \alpha_{i,t}^{(1)} / \|\alpha_{i,t}\|_1$.

### 4.2 Majority Voting Success Probability

**Theorem 1**: Let $\Delta := \bar{\theta}_1 - \bar{\theta}_2$ be the margin between the mean belief in the correct answer and the next-best answer. If $N > K/\Delta^2$, then:

```math
\begin{align}
  P(y_{\text{mv}} = 1) \geq 1 - \exp\!\left(-N \left(\frac{\Delta}{\sqrt{K}} - \frac{1}{\sqrt{N}}\right)^{\!2}\right)
\end{align}
```

This shows a **magnifying effect**: even a small positive margin $\Delta$ is exponentially amplified as $N$ grows. MV success probability approaches 1 as $N \to \infty$.

### 4.3 Bayesian Belief Update in Debate

When agent $i$ observes neighbor responses $\{y_{j,t-1}\}_{j \in \mathcal{N}(i)}$, it forms a count vector $c_{i,t} \in \mathbb{Z}^K_{\geq 0}$ of those responses.

**Definition 2 (Bayesian update):**

```math
\begin{align}
  \alpha_{i,t} = \alpha_{i,t-1} + c_{i,t}
\end{align}
```

**Lemma 1 (Bayesian conjugacy)**: The posterior after observing neighbor responses remains Dirichlet:

```math
\begin{align}
  \theta_{i,t} \mid \{y_{j,t-1}\}_{j \in \mathcal{N}(i)} \sim \text{Dirichlet}(\alpha_{i,t-1} + c_{i,t})
\end{align}
```

### 4.4 Martingale Behavior of MAD

**Theorem 2 (Martingale)**: Let $p_t := \bar{\theta}_{i,t}^{(1)}$ denote agent $i$'s mean belief in the correct answer at round $t$. Then:

```math
\begin{align}
  \mathbb{E}[p_t \mid p_{t-1}, \ldots, p_0] = p_{t-1}, \quad \forall t \geq 0
\end{align}
```

*Proof sketch*: $\mathbb{E}[c_{i,t}^{(k)}] = |\mathcal{N}(i)| \cdot \bar{\theta}_{t-1}^{(k)}$ (expected neighbor vote count for class $k$ is proportional to current mean belief). Since the Bayesian update adds $c_{i,t}$ to $\alpha_{i,t-1}$, the resulting mean shifts by an amount proportional to $\bar{\theta}_{t-1}^{(k)} - \bar{\theta}_{t-1}^{(k)} = 0$ in expectation.

> [!NOTE]
> The martingale property does **not** say debate is harmful — it says debate provides no *systematic* improvement in expectation. High-variance agents may individually move up or down, but on average debate neither helps nor hurts.

The Martingale Convergence Theorem further implies that $p_t$ converges almost surely to some random variable $p_\infty$, meaning debate rounds eventually stabilize without guaranteed improvement.

## 5. Theory-Informed Improved MAD Designs

### 5.1 MAD-Oracle (Upper Bound)

An oracle variant where agents who have produced the correct answer are "locked in" — their belief concentrations become fixed and immune to peer influence. This breaks the martingale by selectively reinforcing correct signals. It serves as an upper bound but requires ground truth access.

### 5.2 MAD-Conformist

A practical intervention without oracle access. If agent $i$'s response at round $t-1$ matches the majority vote at that round, it retains that response at round $t$ instead of generating a new one. This suppresses correct answers from being corrupted by incorrect peer influence.

**Algorithm (MAD-Conformist)**:
```
For round t = 1 to T:
  Compute majority vote m_{t-1} = mode({y_{j,t-1}})
  For each agent i:
    if y_{i,t-1} == m_{t-1}:
      y_{i,t} = y_{i,t-1}   # lock response
    else:
      y_{i,t} = D(x; R_i^(t))  # debate update
```

### 5.3 MAD-Follower

A softer intervention. At each round, with probability $p = 0.30$ an agent adopts the majority response directly; otherwise it generates a new response via debate:

**Algorithm (MAD-Follower)**:
```
For round t = 1 to T:
  Compute majority vote m_{t-1} = mode({y_{j,t-1}})
  For each agent i:
    u ~ Uniform(0,1)
    if u < 0.30:
      y_{i,t} = m_{t-1}   # follow majority
    else:
      y_{i,t} = D(x; R_i^(t))  # debate update
```

Both MAD-Conformist and MAD-Follower consistently outperform vanilla MAD (T=5) across all 7 benchmarks.

## 6. Extended Experiments

### 6.1 Larger Models

Findings replicate on Qwen2.5-32B-Instruct (Table 3). On GSM8K, majority voting (0.9433) outperforms Decentralized MAD T=2 (0.9400), which itself nearly matches. On HellaSwag, both are near-identical (0.8667 vs 0.8633).

### 6.2 Heterogeneous Agent Personas

Agents are assigned distinct personas (e.g., different professional backgrounds). Table 4 shows majority voting (0.9367 on GSM8K) still outperforms Decentralized MAD (0.8033). On MMLU-Pro.Med., persona-based MAD (0.8419) outperforms voting (0.8235), suggesting specialized domain personas may add marginal debate value in knowledge-intensive tasks.

### 6.3 Open-Ended Text Generation

On CNN/DailyMail summarization (30 samples), ROUGE-1 scores vary minimally: best single-agent (0.2760), MAD T=1 (0.2686), MAD T=3 (0.2825). Debate provides negligible gain for open-ended generation, consistent with the theory (no clear "correct" majority to reinforce).

### 6.4 Closed-Source Models (GPT-4)

GPT-4 experiments (Appendix G) confirm the findings. Majority voting outperforms or matches Decentralized MAD T=3 on Arithmetics (0.9967 vs. 0.9833), HellaSwag (0.9078 vs. 0.9044), and HH-RLHF (0.5612 vs. 0.5459). CommonsenseQA is the only case where MAD marginally leads (0.8780 vs. 0.8721).

## 7. Related Work

MAD has been studied extensively since Du et al. (2024), who showed that iterative exchange improved factuality and reasoning. Subsequent work extended MAD with sparse communication topologies (Li et al., 2024), group discussion structures (Liu et al., 2024), and role-playing personas (Wang et al., 2024). Critical analyses (Huang et al., 2024; Smit et al., 2024; Wang et al., 2024) find MAD often fails to outperform simpler baselines, attributing success to social conformity and position bias rather than genuine collaborative reasoning. This work provides the first **formal martingale proof** that debate does not improve expected correctness under the DCM model, unifying these empirical observations theoretically.

> [!TIP]
> For comparison: Kaesberg et al. (2025, arXiv) independently study voting vs. consensus in MAD; Zhang et al. (2025, arXiv) ask "If multi-agent debate is the answer, what is the question?" — both reach similar conclusions empirically. This paper adds the formal theoretical treatment.

## Experiments

- **Datasets**: Arithmetics (100q, synthetic arithmetic expressions $a + b \times c + d - e / f$), GSM8K (300q subsampled), MMLU-Professional Medicine (272q full test), MMLU-Formal Logic (126q full test), HellaSwag (300q subsampled), CommonsenseQA (300q validation split), HH-RLHF (300q random subset), CNN/DailyMail (30q version 3.0.0).
- **Hardware**: RTX A6000 or RTX A100 GPUs.
- **Models**: Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct (primary); Qwen2.5-32B-Instruct, GPT-4 (extended).
- **Hyperparameters**: temperature = 1.0, nucleus sampling $p = 0.9$, max tokens = 512, $N = 5$ agents, $T \in \{2, 3, 5\}$ rounds.
- **Key results**:
  - Majority voting averaged **0.7691** vs. best MAD configuration **0.7377** (Qwen2.5-7B-Instruct, 7 benchmarks).
  - MAD-Conformist (0.7524) and MAD-Follower (0.7577) outperform vanilla MAD T=5 (0.7084).
  - MAD-oracle achieves **0.8289**, demonstrating the headroom available if correct signals can be identified.
  - Performance degrades as $T$ increases in vanilla MAD (0.7377 → 0.7050), consistent with martingale convergence.

## References

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.
- Li, Y., Du, Y., Zhang, J., et al. (2024). Improving multi-agent debate with sparse communication topology. *EMNLP 2024 Findings*.
- Huang, J., Chen, X., Mishra, S., et al. (2024). Large language models cannot self-correct reasoning yet. *ICLR 2024*.
- Smit, A. P., Grinsztajn, N., Duckworth, P., et al. (2024). Should we be going mad? A look at multi-agent debate strategies for LLMs. *ICML 2024*.
- Wang, Q., Wang, Z., Su, Y., Tong, H., & Song, Y. (2024). Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? *ACL 2024*.
- Kaesberg, L. B., Becker, J., Wahle, J. P., Ruas, T., & Gipp, B. (2025). Voting or consensus? Decision-making in multi-agent debate. *arXiv*.
- Zhang, H., Cui, Z., Wang, X., et al. (2025). If multi-agent debate is the answer, what is the question? *arXiv*.
- Yang, A., Yang, B., Zhang, B., et al. (2024). Qwen2.5 technical report. *arXiv:2412.15115*.
- Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 herd of models. *arXiv:2407.21783*.
- Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training verifiers to solve math word problems. *arXiv:2110.14168*.
- Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring massive multitask language understanding. *ICLR 2021*.
- Zellers, R., Holtzman, A., Bisk, Y., et al. (2019). HellaSwag: Can a machine really finish your sentence? *ACL 2019*.
- Talmor, A., Herzig, J., Lourie, N., & Berant, J. (2019). CommonsenseQA. *NAACL 2019*.
- Bai, Y., Jones, A., Ndousse, K., et al. (2022). Training a helpful and harmless assistant with RLHF. *arXiv:2204.05862*.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get to the point: Summarization with pointer-generator networks. *ACL 2017*.
- Pemantle, R. (2007). A survey of random processes with reinforcement. *Probability Surveys*, 4, 1–79.
