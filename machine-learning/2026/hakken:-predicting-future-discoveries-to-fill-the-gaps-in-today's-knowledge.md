# Meta Information

- URL: [Hakken: Predicting future discoveries to fill the gaps in today's knowledge](https://arxiv.org/abs/2609.04494)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Besold, T. R. et al. (2026). Hakken: Predicting future discoveries to fill the gaps in today's knowledge. arXiv:2609.04494.

> [!NOTE]
> Authors: Tarek R. Besold (lead) and 22 co-authors, including Uchenna Akujuobi, Pablo Sanchez, and Alessandra Toniato. Primary affiliations are Sony AI (Barcelona, Tokyo, Zurich, London, New York), the Systems Biology Institute (Tokyo), and the Okinawa Institute of Science and Technology (OIST).

# Overview

This paper introduces **Hakken**, a system that predicts scientific relationships that have not yet been documented in the literature, rather than merely extracting or summarizing relationships that already exist in published text. This distinguishes it from Literature-Based Discovery (LBD) tools and generative-AI summarization pipelines, which the authors describe as staying within the "deductive closure of pre-existing knowledge." Hakken is instead positioned alongside data-generating scientific AI systems such as AlphaFold and GNoME, which produce genuinely new results rather than re-arranging known ones.

Hakken applies to biomedical researchers, drug-discovery teams, and knowledge-graph practitioners who need to (a) prioritize which untested hypotheses are most likely to be true before committing wet-lab resources, and (b) obtain a human-interpretable rationale for why a hypothesis was proposed. It operates on time-stamped knowledge graphs extracted from scientific publications, so it is only applicable in domains where a large historical text corpus with extractable entity relations exists (the paper demonstrates this for biomedicine using PubMed/PMC literature).

The system has two components:

- **THiGERLLM** (Temporal Hierarchical Graph-based Encoder Representation with LLM): a predictor that estimates, for a pair of entities, the probability that each of a fixed set of relation types will be reported as true in the future.
- **PHELInE** (Predicted Hypothesis Elucidation with Literature-Inferred Explanations): a model-agnostic explanation module that finds which existing facts in the knowledge graph best justify a given prediction.

# Problem Setting

The input to the system is a sequence of temporally-stamped knowledge graph snapshots $G^{(1)}, \dots, G^{(T)}$, where each snapshot $G^{(t)}$ contains triples $(s, r, o)$ extracted from publications up to time $t$. Each entity $i$ additionally has an ontology-derived feature vector and a domain label. Given a candidate entity pair $(s, o)$ observed at some point in the graph, the task is multi-label classification over $C$ relation types: for each relation class $c \in \{1, \dots, C\}$, predict a probability $p_{s,o,c} \in [0,1]$ that the pair $(s,o)$ will be reported as having relation $c$ at some future time, using only information available up to the current cutoff. This is a strictly harder task than the authors' earlier model THiGER, which only predicted **whether** a relationship would exist between two entities, not **which type** it would be.

# THiGERLLM Architecture

## Node Embedding Initialization

Each node $i$ starts with an initial feature vector built by concatenating a trainable embedding, an ontology-derived vector, and a domain-assignment embedding, then projecting with a learned weight matrix:

```math
\begin{align}
  \mathbf{h}_i^{(0)} = \mathbf{W}_{\text{comb}}[\mathbf{e}_i ; \mathbf{o}_i ; \mathbf{d}_{g(i)}] \in \mathbb{R}^{d_0}
\end{align}
```

Here $\mathbf{e}_i$ is a trainable per-node embedding, $\mathbf{o}_i$ is an ontology-derived vector obtained via RDF2vec (which performs random walks over the relational graph to produce node sequences, then trains word2vec on those sequences to obtain embeddings), and $\mathbf{d}_{g(i)}$ is an embedding indexed by the node's assigned macro-domain $g(i)$.

## Per-Snapshot Message Passing (GraphSAGE)

For each temporal snapshot $t$, node representations are refined over $K$ GraphSAGE layers by aggregating neighbor information:

```math
\begin{align}
  \mathbf{m}_i^{(k,t)} &= \text{AGG}^{(k)}\left(\{\mathbf{h}_j^{(k-1,t)} : j \in \mathcal{N}_k^{(t)}(i)\}\right) \in \mathbb{R}^{d_{k-1}} \\
  \mathbf{h}_i^{(k,t)} &= \phi\left(\mathbf{W}^{(k)}[\mathbf{h}_i^{(k-1,t)} ; \mathbf{m}_i^{(k,t)}] + \mathbf{b}^{(k)}\right) \in \mathbb{R}^{d_k}
\end{align}
```

where $\mathcal{N}_k^{(t)}(i)$ is the neighborhood of node $i$ within snapshot $t$, $\text{AGG}^{(k)}$ is a permutation-invariant aggregator (e.g. mean/pooling), and $\phi$ is a nonlinearity. After $K$ layers, per-snapshot pair embeddings are formed by normalizing and linearly projecting the final-layer node states of $s$ and $o$.

## Hierarchical Temporal Transformer

Because the number of snapshots $T$ can be large, the per-snapshot pair embeddings are compressed hierarchically over $\log_2 T$ stages rather than fed through a flat sequence Transformer. At stage $l$, round $r$:

```math
\begin{align}
  \mathbf{H}^{(l,r+1)} &= \text{Tx}^{(l,r)}\left(\mathbf{H}^{(l,r)} ; \mathbf{B}^{(l)}\right) \\
  \mathbf{H}_k^{(l+1,0)} &= \mathbf{H}_{2k-1}^{(l,R)} + \mathbf{H}_{2k}^{(l,R)}
\end{align}
```

i.e. adjacent pairs of temporal states are merged by summation after $R$ rounds of Transformer self-attention at level $l$, halving the sequence length at each stage. To prevent the model from taking a "shortcut" that only attends within an already-adjacent pair without mixing across pair boundaries, a pair-block attention bias masks out same-group non-diagonal attention:

```math
\begin{align}
  \mathbf{B}_{ij}^{(l)} = \begin{cases} -\beta & \text{if } \lfloor (i-1)/2 \rfloor = \lfloor (j-1)/2 \rfloor \text{ and } i \neq j \\ 0 & \text{otherwise} \end{cases}
\end{align}
```

with $\beta$ a large penalty constant, forcing genuine cross-group temporal information flow before each length-reduction step.

## Graph-to-Token Mapping and Time Encoding

Time is encoded with a bank of Fourier features, projected and combined with the reduced graph representation:

```math
\begin{align}
  \psi(t_m) &= \mathbf{W}_t\left[\sin(2\pi f_1 t_m), \dots, \sin(2\pi f_X t_m) ; \cos(2\pi f_1 t_m), \dots, \cos(2\pi f_X t_m)\right] \in \mathbb{R}^d \\
  \mathbf{g}_m &= \sigma(\gamma) \cdot \text{RMSNorm}\left(\phi(\mathbf{r}_m) + \psi(t_m)\right)
\end{align}
```

where $\mathbf{r}_m$ is the pair representation at reduction step $m$ and $\sigma(\gamma)$ is a learned gate. The final graph-branch pair representation is a further normalized projection:

```math
\begin{align}
  \mathbf{g}_{so}^{\text{pair}} = \text{RMSNorm}\left(\mathbf{W}_{gp} \mathbf{z}_{so}^{+}\right) \in \mathbb{R}^d
\end{align}
```

## Branch Fusion

THiGERLLM computes per-class logits from two branches and fuses them. The **text branch** uses an LLM-derived representation $\mathbf{h}_{so}^{\text{text}}$ (built from sampled literature sentences mentioning $s$ and $o$) plus a learned label-history term:

```math
\begin{align}
  \ell_{so,c}^{\text{text}} = \ell_{so,c}^{\text{base}} + \alpha_c \left\langle \mathbf{W}_{\text{emb}} \mathbf{e}_c + \mathbf{W}_{\text{ctx}} \mathbf{h}_{so,c}^{\text{hist}}, \; \mathbf{W}_{\text{text}} \mathbf{h}_{so}^{\text{text}} \right\rangle
\end{align}
```

The **graph branch** uses the temporal graph pair embedding $\mathbf{z}_{so}^{+}$ analogously:

```math
\begin{align}
  \ell_{so,c}^{\text{graph}} = \left\langle \mathbf{W}_z \mathbf{z}_{so}^{+} + \mathbf{W}_h \mathbf{h}_{so,c}^{\text{hist}} + \mathbf{b}, \; \mathbf{e}_c \right\rangle
\end{align}
```

At inference, the graph and text branch scores are combined via a label-wise ensemble whose weights are calibrated (Platt scaling and isotonic regression) on a held-out validation split, rather than fixed at training time.

## PU-Safe Dynamic-Weighted Focal Loss

Because a missing triple in the graph can mean "false" or simply "not yet observed" (positive-unlabeled setting), training targets are softened by an annealed pseudo-label term rather than treated as hard negatives:

```math
\begin{align}
  \tilde{y}_{so,c} &= y_{so,c} + (1 - y_{so,c}) \, \eta_c(i) \\
  \eta_c(i) &= \text{clip}\left(\hat{q}_c \, \eta_{\max} \, r(i), \; 0, \; \eta_c^{\text{cap}} \, r(i)\right)
\end{align}
```

where $y_{so,c} \in \{0,1\}$ is the observed (possibly incomplete) label, $\hat{q}_c$ is an estimated class prior, $r(i)$ is a training-progress ramp factor, and $\eta_{\max}, \eta_c^{\text{cap}}$ bound the amount of soft positive mass injected into unlabeled pairs. This soft target is then used inside a focal loss with per-class dynamic weighting, so that rare relation types and likely-but-unconfirmed positives are not penalized as hard errors.

> [!IMPORTANT]
> The PU-safe loss is the mechanism that lets the model be trained directly on an incomplete knowledge graph (most true relations are never written down) without collapsing all unobserved pairs to confident negatives, which would otherwise suppress exactly the "novel discovery" signal Hakken is built to surface.

# Differences from Prior/Related Methods

| Method | Predicts existence | Predicts relation type | Uses temporal graph evidence | Uses text/LLM evidence |
|---|---|---|---|---|
| ComplEx (KG embedding) | Yes | Yes (static) | No | No |
| tNodeEmbed | Yes | No | Yes | No |
| THiGER (authors' prior work) | Yes | No | Yes | No |
| **THiGERLLM (this paper)** | Yes | **Yes (multi-label)** | Yes | **Yes** |

THiGERLLM's central novelty relative to THiGER is moving from binary link existence prediction to multi-label relation-type prediction, and fusing the temporal graph branch with an LLM text branch rather than relying on graph structure alone. Compared to standard static KG-completion models like ComplEx, THiGERLLM additionally conditions on the temporal ordering of evidence so that predictions can be backtested against a strict knowledge cutoff.

# PHELInE Explanation Framework

PHELInE is model-agnostic: it treats THiGERLLM (or any other scorer) as a black box and explains a specific prediction $(s, r, o)$ after the fact.

1. **Candidate generation**: enumerate relational paths of length $k$ connecting $s$ and $o$ in the knowledge graph, and collect the triples along each path as candidate explanatory facts.
2. **Surrogate scoring**: rather than repeatedly querying the (expensive) original THiGERLLM model, a lightweight GraphSAGE-based surrogate is trained to approximate its scoring behavior, so that many hypothetical graph edits can be evaluated cheaply.
3. **Sufficiency score**: evaluates the surrogate on a graph containing *only* the candidate explanation's triples, measuring whether those triples alone can reproduce a high prediction score.
4. **Necessity score**: evaluates the surrogate on the full training graph with the candidate explanation's triples *removed*, measuring how much the prediction score drops.
5. **Reranking**: candidate explanations are ranked by their influence scores (combining sufficiency and necessity) and then re-ranked to favor diversity across distinct relational paths, so the final explanation set covers multiple plausible causal mechanisms instead of near-duplicate path variants.

> [!TIP]
> Sufficiency/necessity as a pair of counterfactual-style scores mirrors the standard framing used in feature-attribution explainability work (e.g. "sufficient" vs. "necessary" causes); here it is adapted to knowledge-graph triples instead of input features.

# Experiments

- **Dataset**: A biomedical knowledge graph built from PMC Open Access full text, MEDLINE titles/abstracts, and a commercially licensed corpus. Raw extraction yielded 249,329,650 triples; after cleaning, the graph used for experiments contains 254,806 entities and 7,127,960 triples spanning 23 relation types across 20 macro-domains, with each triple timestamped by its source publication year. A fully open-source variant is trained only on NCBI's public PubTator3 dataset and is released at `https://github.com/SonyResearch/HakkenOSS`.
- **Evaluation protocol**: temporally split to avoid leakage — models are trained on knowledge available up to a cutoff year (e.g. 2020) and evaluated on relations first observed strictly after that cutoff. For each entity pair, 20 sentences are randomly sampled from PMC publications predating the cutoff to build the text-branch input.
- **Baselines**: random scoring, ComplEx, k-nearest-neighbor over learned representations, a multi-label MLP classifier, a rule-based multi-label classifier, tNodeEmbed, and the authors' own prior model THiGER.
- **Results (2020 cutoff)**: THiGERLLM reaches macro-F1 51.16 / macro-recall 60.72, compared to THiGER's macro-F1 50.98 / macro-recall 46.13 — i.e. adding the LLM text branch mainly improves macro recall (catching more of the true positives, especially for less frequent relation types) while weighted precision/recall stay close to THiGER's. nDCG (ranking quality) is similar between the two (90.30 vs. 92.19).
- **Temporal backtesting**: training cutoffs were swept across 1990, 2000, and 2010, with evaluation windows binned into non-overlapping future horizons (0-2 years, 2-4 years, ..., up to 10 years ahead). Macro recall degrades moderately (about 8%) as the prediction horizon lengthens, while nDCG remains stable across cutoffs and horizons, and the ranking advantage over a random-reassignment null baseline (~0.24 recall) persists at every horizon — indicating the model's relative ranking ability does not degrade sharply even a decade out.
- **Wet-lab validation**: from 1,543,297 above-confidence-threshold hypotheses generated over an expert-curated list of aging-relevant genes, 3 candidate hypotheses were selected for experimental testing. Two were confirmed: TP53 induction consistently increased BAMBI expression, and RAF1 inhibition consistently increased TNF mRNA expression, across multiple assays — both are described as previously undocumented gene-gene/enzyme-gene interactions relevant to cancer and inflammation research. The third (SOAT1 stimulation affecting STAT3 activity) was not supported by the wet-lab assay.
- **Scale of hypothesis generation**: applying the trained model across the aging domain produced roughly 1.5 million predictions above the confidence threshold, of which the above 1,543,297-hypothesis pool for the aging use case was drawn.

> [!CAUTION]
> The wet-lab validation covers only 3 tested hypotheses out of over a million generated candidates, so the reported "2 out of 3 confirmed" result is a small-sample proof of concept rather than a calibrated estimate of the model's true positive rate in the lab; readers should not extrapolate a precision number from it directly.

# Limitations and Future Work

The authors note several directions for improvement: incorporating quantitative/numeric information from the literature (rather than only categorical relation types), modeling temporal occurrence frequency of relations more explicitly, and extending the system toward generating experimental designs for testing its own hypotheses (moving beyond prediction and explanation into experiment planning).
