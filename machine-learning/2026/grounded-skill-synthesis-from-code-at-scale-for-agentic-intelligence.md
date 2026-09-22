# Meta Information

- URL: [Grounded Skill Synthesis from Code at Scale for Agentic Intelligence](https://arxiv.org/abs/2609.05571)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Tong, Y., Wang, P., Wang, H., Li, J., Zhang, X., Yang, J.-M., & Wu, W. (2026). Grounded Skill Synthesis from Code at Scale for Agentic Intelligence. arXiv:2609.05571 (Ant International).

---

# Grounded Skill Synthesis from Code at Scale for Agentic Intelligence

## 1. Motivation and Problem Setting

A "skill" in this paper is reusable procedural knowledge — a description of when a procedure applies, what it does, and how it fails — that an agent can retrieve at inference time instead of re-deriving the same steps from scratch. The paper's central claim is that **source code**, rather than agent trajectories or human documentation, is an underused but superior substrate for automatically manufacturing such skills at scale.

**Who / when / where this applies:** the method targets developers of general-purpose or domain-specific coding/tool-using agents (e.g., SWE-agents, terminal agents, math/reasoning agents) who want to inject reusable procedural knowledge into an agent harness without first collecting agent interaction trajectories. It is most useful before an agent has accumulated its own experience, or when trajectory collection is expensive or unavailable.

The paper contrasts two prior synthesis substrates and their weaknesses:

| Substrate | Method examples | Weakness |
|---|---|---|
| Agent trajectories | Trace2Skill, ExpeL, SkillRL-Bank | Skill quality is bounded by the generating agent's own competence; coupled to the specific model, tools, and task distribution that produced the trajectories; breaks when any of those change |
| Human documentation | Liang et al. 2026, Zhou et al. 2026 | No executable evidence exists to verify or ground the synthesized skill |
| **Source code (this paper)** | Code2Skill | Requires no prior agent experience; code is executable, so claims about it can be checked by attempting to regenerate the code from the claim |

> [!NOTE]
> "Source code offers a different pathway — it requires no previous experience while retaining executable evidence for grounding later abstractions." (paper, Abstract)

## 2. Problem Formulation

Construction is defined as a mapping from a source unit (a function, method, CLI entry point, or file-level component) plus its repository context to a candidate **skill record**. An accepted skill record must specify:
- when the procedure applies (applicability condition),
- what behavior to reproduce (the procedure itself),
- which preconditions and invariants govern execution,
- how failures should be handled, and
- which source spans support these claims (provenance).

This yields three requirements a record must satisfy:

1. **Grounded** — its procedural claims must be recoverable from, and checked against, actual implementation spans (verified through a reconstruction test, described in §4 below).
2. **Transferable** — project-specific identifiers and integration glue must be abstracted away, while the reusable preconditions, steps, invariants, and failure-handling logic are preserved.
3. **Maintainable** — provenance, supporting source spans, record type, and construction status must be retained so a record can be inspected, invalidated, or regenerated as the underlying code changes.

## 3. Code2Skill Pipeline

Code2Skill is a four-stage, fully automated pipeline: (1) selecting procedural source units, (2) generating typed skill records, (3) validating records against implementation evidence, and (4) building feature-tagged / purpose-indexed retrieval views.

### 3.1 Selecting Candidate Procedural Evidence

**Input:** public GitHub repositories available as of April 14, 2026, filtered to those with more than 500 stars, yielding a pool of 19,769 repositories.
**Output:** a filtered set of "tagged" source units (functions, methods, CLI entry points, file-level components) considered candidates for extraction.

Within each repository, a parser first builds candidate units, excluding unsafe files, test files, binary payloads, and units too short to expose reusable behavior. An LLM tagger then scores each remaining unit against six rubric signals, each in $[0,1]$:

1. `reusable_intent` — does the unit implement a meaningful operation beyond project-local glue?
2. `procedural_steps` — are there ordered steps, branching, or state/resource transitions?
3. `boundary_conditions` — are preconditions, invariants, validations, or error paths visible?
4. `transferability` — can the behavior be described without project-specific names/hidden context?
5. `interface_sufficiency` — are inputs, outputs, side effects, and required context recoverable?
6. `non_triviality` — is the unit more than a getter/setter/thin wrapper/boilerplate adapter?

Units whose `keep_for_extraction` flag is true and whose rubric profile clears a configured gate (subject to a per-repository cap) proceed to extraction; this stage only filters candidates and does not itself accept a skill record.

### 3.2 Skill Record Generation (Extraction)

**Input:** one tagged source unit, its source-unit tag JSON, and repository/file/language/symbol/interface metadata.
**Output:** a single JSON object with a record `level` and the fields `name`, `summary`, `worth_extracting`, `skill_value_score`, `generality_score`, `abstraction_score`, `inputs`, `outputs`, `workflow` (ordered steps), `invariants`, `error_cases`, `implementation_notes`, `evidence`, `anti_goals`, and `confidence`.

Code2Skill defines **three granularities**, because procedural knowledge appears at different source scopes and a single record type would either fragment multi-step procedures or overgeneralize local behavior:

- **Atomic skills** — a single, well-defined operation within one function or method.
- **Composite skills** — an ordered workflow coordinating multiple operations.
- **Recurring-pattern skills** — higher-level implementation patterns that recur beyond one localized operation or workflow.

A candidate is dropped from further processing if `worth_extracting=false` or `skill_value_score < 0.45`; everything else proceeds to reconstruction. Figure 2 of the paper shows a worked example: a `recoverDoQAfterCachedQUICFailure` composite skill extracted from AdGuard's DNS-over-QUIC resolver, containing a problem statement, a "when to use" trigger, a control rule, an ordered workflow, invariants/failure list, anti-goals, and links back to the exact source commit and functions.

### 3.3 Source-Body-Blind Reconstruction and Consistency Checking (Verification)

This is the grounding mechanism that distinguishes Code2Skill from purely descriptive code-summarization methods (e.g., CodeBERT/GraphCodeBERT-style summarizers): instead of trusting the extracted description, the pipeline tests whether the description alone is sufficient to regenerate equivalent code.

**Algorithm (Algorithm 1 in the paper, "source-body-blind reconstruction and source-aware LLM consistency checking"):**

```
Require: skill description s, metadata I_rec (visible to reconstructor),
         I_eval (visible to judge/adjudicator), original source body b_u

1. Reconstructor sees:      {s, I_rec}          # b_u is HIDDEN
2. b_hat <- M_rec(s, I_rec)                       # regenerate code from the skill alone
3. Equivalence judge sees:  {I_eval, summary(s), b_u, b_hat}
4. j <- M_judge(I_eval, summary(s), b_u, b_hat)
5. if j.equivalent:
6.     d <- DirectAccept
7. else:
8.     Adjudicator sees:    {j.reason, I_eval, s, b_u, b_hat}
9.     a <- M_adj(j.reason, I_eval, s, b_u, b_hat)
10.    if a.keep:
11.        d <- AdjudicatedAccept
12.    else:
13.        d <- Reject

Ensure: d in {DirectAccept, AdjudicatedAccept, Reject}
```

Three model roles are chained, each with a deliberately restricted information window (an information-boundary matrix in Appendix C.2 formalizes exactly what each stage can and cannot see):

1. **Regenerator** (`M_rec`) — receives only the skill description `s` and interface metadata `I_rec` (symbol, unit type, interface, optional chunk context). The original source body, repository, and file path are explicitly withheld. It must "generate code only from the provided skill specification," preserving observable behavior without adding new features.
2. **Equivalence judge** (`M_judge`) — receives the original code, the regenerated code, and interface/summary metadata, but not the unit type, tests, execution traces, type-checker output, or maintenance metadata. It returns `{equivalent, confidence, reason, differences, risk_level}` and focuses on behavior/side-effects/error-handling rather than style.
3. **Adjudicator** (`M_adj`) — invoked only when the judge found the pair non-equivalent. It receives the judge's reasoning plus the full skill JSON and decides whether the extracted *skill* is still faithful even though the *reconstruction* failed, returning `{keep, skill_is_correct, generation_failed, reason, corrective_notes}`.

This decouples two distinct failure modes: a bad skill description vs. a reconstruction model that simply failed to regenerate code well — only the former should cause rejection.

> [!IMPORTANT]
> The verification loop never uses unit tests or execution; it is a purely LLM-based comparison. The paper is explicit that this is a scalable repository-level filter, not a substitute for test-based verification (§2, Related Work): "reconstruction-based consistency checking serves as a scalable repository-level filter for assessing whether synthesized skills are supported by their source implementations, rather than as a substitute for test-based verification."

Accepted records retain their acceptance stage (`DirectAccept` / `AdjudicatedAccept`), rationale, and full repository/file/symbol/source-span provenance, enabling later inspection, refresh, or deprecation as the source repository evolves.

### 3.4 Retrieval-Oriented Feature Tagging and Purpose Indexing

**Input:** an accepted skill record plus five controlled vocabularies (task families, capability tags, constraint tags, success tags, interaction patterns — enumerated in Table 5 of the paper).
**Output:** one canonical, retrieval-facing "skill card" per accepted record, in one-to-one correspondence with the source record — this stage relabels/rewrites but does not merge or drop records. Fields include `transfer_title`, `transfer_summary`, `when_to_use`, `task_family`, `capability_tags`, `constraint_tags`, `success_signal`, `failure_modes`, `interaction_pattern`, `retrieval_phrases`, and `notes`. Invalid/out-of-vocabulary values fall back to defaults (`workflow` for task family, `single_step` for interaction pattern).

Separately, **purpose indexing** (Algorithm 2, Appendix C.3) is a *deterministic*, non-LLM step that deduplicates near-identical records: a fixed key over task family, inferred action, and inferred target groups records into clusters, and a fixed scoring function over workflow/invariants/error-cases/evidence/summary-length/record-type selects one existing record as the representative for each cluster (no new record is synthesized at this stage). This reduces retrieval-time redundancy while leaving the full evidence archive intact.

## 4. CodeSkillBank: Resulting Resource

Applying Code2Skill to the 19,769-repository pool yields **CodeSkillBank**, containing **1,006,822 accepted skill records**.

**Source pool characteristics:** the median repository has 3,133 stars and 82 merged pull requests; 78.3% have ≥1,000 stars, 46.9% have ≥100 merged PRs, and 66.0% were pushed within the previous year — i.e., the pool skews toward actively maintained, broadly adopted code.

**Human-annotation quality check:** annotators evaluated four separately-sampled pipeline outcomes (filtered-out, directly accepted, accepted-after-adjudication, rejected) for description accuracy, reconstruction correctness, and retention value. In the final bank, 92% of skill descriptions were judged accurate and 80% of records judged worth retaining; among directly accepted records, 84% also supported correct reconstruction. The rejected-sample pool, by contrast, reached only 32% description accuracy, 28% retention value, and no correct reconstructions — evidence that the verification stage is doing real filtering work rather than passing everything through.

**Purpose-indexed granularity mix:** of the purpose-level cards, 65.35% (490,457) are atomic-only, 31.12% (233,549) composite-only, and 3.53% (26,530) mixed — meaning over a third of purpose clusters retain workflow-level, not just single-operation, knowledge.

**Reasoning depth:** 57.3% of records encode multi-step procedures and 33.5% encode constraint reasoning; only 8.6% are surface API calls. 79.6% of records are labeled correctness-critical.

### 4.1 Utilization Interface

Given a task $x$ and agent/model state $h_t$ before decision step $t$, if the downstream interface issues a retrieval query $q_t$ against the bank $\mathcal{B}$:

```math
\begin{align}
  z_t &=
  \begin{cases}
    \mathrm{Render}_r\big(\mathrm{TopK}_k(q_t; \mathcal{B})\big), & \text{if a query is issued} \\
    \varnothing, & \text{otherwise}
  \end{cases} \\
  a_t &\sim \pi_{\theta(s)}(\cdot \mid x, h_t, z_t)
\end{align}
```

Here $\mathrm{TopK}_k$ retrieves the $k$ most relevant records for query $q_t$, $\mathrm{Render}_r$ renders them under rendering scheme $r$ (e.g., full record vs. compact summary) into a context block $z_t$, and $a_t$ is the output at that decision point (a generation, a plan, a review verdict, a reward, or an environment action) produced by policy $\pi$ with parameters $\theta(s)$ at checkpoint/stage $s$. The **no-skill control** is the identical protocol with $z_t = \varnothing$, which the paper uses as the baseline in every comparison. This formalization is deliberately interface-agnostic: it covers prompt-level retrieval, planning-time retrieval, post-generation review, verifier-side reward use, and RL training-time conditioning, all evaluated separately in the experiments below.

## 5. Experiments

The paper investigates six research questions (RQ1–RQ6). All comparisons are **protocol-matched**: only skill access changes (`Yes`/`No` retrieval of CodeSkillBank), while the model, reasoning mode, benchmark task set, evaluator, and agent loop are held fixed. Unless stated otherwise, the default agent harness is a draft → review → revise loop (Self-Refine / Reflexion-style), and for computational efficiency a random 10% subsample of CodeSkillBank forms the default retrieval pool.

- **Models:** DS4-Flash, Qwen3.5, Qwen3.6, Gemini 2.5 Pro, GPT-5.2 (nine model/reasoning-mode settings total).
- **Benchmarks (main evaluation, Table 1):** BigCodeBench and SWE-bench Verified (coding/software engineering); TerminalBench, LongCLI-Bench, AgentBench-OS (terminal/OS control); AIME 2026, HMMT 2025, GPQA (math/scientific reasoning).
- **Provenance experiment (§5.7) dataset:** a 400-task subset of LiveCodeBench.

### RQ1 — Does a code-derived skill bank improve agent performance?

CodeSkillBank retrieval improves **57 of 72** protocol-matched evaluations (9 model settings × 8 benchmarks). The macro-average score rises from **42.90 (no skills) to 47.90 (with skills)**, an **11.7% relative gain**; per-setting eight-benchmark averages rise 2.26–7.39 points (6.0%–20.7% relative). All nine SWE-bench Verified pairs improve. Gains are largest on SWE-bench Verified and TerminalBench (multi-step repository/system interaction gives more opportunities to apply stored workflows and invariants); BigCodeBench shows mixed results under some reasoning modes, suggesting less benefit when the base model can already solve short problems directly.

### RQ2 — How does CodeSkillBank compare with trajectory-derived skill banks under a shared interface?

CodeSkillBank is compared against re-implementations of three trajectory-derived baselines — **Trace2Skill** (error-analysis-patch consolidation), **ExpeL** (natural-language experience memory), and **SkillRL-Bank** (retrievable policy checks) — all evaluated through the identical downstream interface and agent loop with DS4-Flash, mean ± std over five runs on seven shared benchmarks.

| Method | Avg. score (7 benchmarks) |
|---|---|
| Trace2Skill | 31.0 |
| ExpeL | 27.9 |
| SkillRL-Bank | 32.8 |
| Oracle (best baseline per-benchmark) | 40.1 |
| **Code2Skill (CodeSkillBank)** | **49.5** |

Code2Skill ranks first on all seven benchmarks and beats even an oracle that hand-picks the best baseline separately for each benchmark by 9.5 points, and beats the single strongest baseline on any given benchmark by 6.6–13.3 points. This isolates the *source* of the gain: it is not merely "having a skill bank in this loop," since trajectory-derived banks in the same loop underperform code-derived ones.

### RQ3 — Where should skills enter the agent workflow?

Three insertion points are compared: **generation-time prompting** (skills appended to the first-pass prompt, review/revision stages removed), **planning-time guidance** (skills supplied while the agent builds its plan), and **post-generation critique** (skills revealed only during review of an existing draft — the paper's default).

- Planning-time guidance improves all 8 evaluated model×benchmark combinations (DS4-Flash, Qwen3.5 × AIME/HMMT/Terminal/SWE), e.g., DS4-Flash gains up to +22.0 points on AIME.
- Generation-time prompting helps DS4-Flash consistently (up to +22.0 on AIME, +10.4 on Terminal) but is inconsistent for Qwen3.5 (one benchmark even drops −5.6 points).
- Post-generation critique (the broad Table 1 evaluation) is the most uniformly positive: the eight-benchmark average rises for every model and reasoning mode.

**Insight:** review-time use lets a retrieved skill constrain/check a concrete candidate answer; generation-time use requires the model to translate guidance directly into a full solution before task-specific context exists, which is harder and less consistent.

### RQ4 — Can skills be represented compactly without losing utility? (BigCodeBench Instruct-Hard, DS4-Flash and Qwen3.5)

Three retrieval-design axes are varied: depth $k \in \{1, 3, 10\}$, full-record vs. summary rendering, and retrieval from all accepted records vs. purpose-indexed representatives.

- **Depth:** increasing $k$ from 1 to 10 grows average rendered context from 2.1K to 17.8K characters with little additional benefit (Qwen gains only 1.40 points; DS4-Flash stays at or below its no-skill baseline).
- **Rendering:** at $k=3$, summary rendering cuts average context by **88.9%** (6,352 → 707 characters) while Qwen keeps its full-record score and DS4-Flash *improves* (28.40 → 31.80).
- **Purpose indexing:** reduces full-record context from ~6.4K to ~5.0K characters but has mixed pass-rate effects — it helps DS4-Flash's full-record condition but hurts three of the other four model×renderer pairs, since selecting one canonical representative can displace a locally relevant candidate.

**Insight:** compact summary rendering is a more effective lever than deeper retrieval — most of the utility survives after removing ~90% of the retrieved text, implying the *procedure* matters more than surrounding implementation detail.

### RQ5 — Does insertion point still matter under reinforcement learning?

Using a fixed Qwen3-32B SWE-World checkpoint and coding-RL protocol (Sun et al., 2026b) at training step 150, four CodeSkillBank interfaces are compared against a no-skill control:

| Interface | Skill placement | Resolve rate |
|---|---|---|
| No CodeSkill | — | 24% |
| Full prompt | Policy, full records | 32% |
| Summary prompt | Policy, summaries | 31% |
| Reward reference | Verifier | 31% |
| **Post-generation review** | **Reviewer** | **38%** |

All four skill-enabled conditions beat the no-skill control; post-generation review gives the largest gain (+14 points vs. +7–8 for policy/reward-side access). The paper notes this is a single checkpoint without repeated seeds or full learning curves, so it establishes compatibility with multiple RL interfaces rather than differences in learning speed or final convergence.

### RQ6 — Can AI-generated code supply a continuously expanding skill source?

25 tested source implementations are sampled from CodeSkillBank; for each, GPT-5.1 (Codex, extra-high reasoning effort) generates an interface-matched AI implementation that must pass the same public/hidden tests as the human version. Origin labels are withheld from the extractor. Each of the 25 human and 25 AI implementations yields 2 skills (implement + verify), producing two 50-skill banks, evaluated with identical top-$k=4$ retrieval (DS4-Flash, high reasoning effort, temperature 0) on a 400-task LiveCodeBench subset.

- Human-code bank: **93.00%** pass rate (372/400).
- AI-code bank: **93.50%** pass rate (374/400).

Although aggregate scores differ by only 0.5 points, the two banks disagree on 16 individual tasks (human-only solves 7, AI-only solves 9) — i.e., they are not redundant even though similarly effective, suggesting AI-generated code can be a legitimate, continuously growing input to the same pipeline as AI coding becomes more prevalent.

## 6. Comparison with Related / Similar Approaches

| Approach family | Substrate | Coupling | Verification against execution | Notes |
|---|---|---|---|---|
| Trace2Skill, ExpeL, SkillRL-Bank | Agent execution trajectories | Coupled to the model/tools/task distribution that generated the trajectories | Indirect (comes from trajectory outcomes) | Quality bounded by generating agent's own competence; skills can go stale when the harness changes |
| Document-based synthesis (Liang et al. 2026, Zhou et al. 2026) | Human-readable text/docs | Decoupled from any specific agent | None — no executable evidence | Avoids trajectory dependence but cannot ground or verify claims |
| Bi et al. 2026 (`SKILL.md` mining) | Open-source *agent* repositories | Scoped to agent-framework code | Not reconstruction-based | Closest prior work; retrieves/serializes skills from agentic repos but does not rank arbitrary procedural source units, use typed multi-granularity records, or apply reconstruction-based grounding |
| CodeBERT / GraphCodeBERT-style code summarization | Any source code | N/A (representation learning, not skill synthesis) | None | Produces textual/embedding representations of code, not verified, task-facing procedural skill records |
| **Code2Skill (this paper)** | Arbitrary maintained source code (19,769 repos) | Independent of any agent's trajectories | Explicit — source-body-blind reconstruction + source-aware judge/adjudicator | Multi-granularity (atomic/composite/pattern) typed records with provenance; scores 49.5 vs. 27.9–32.8 for trajectory baselines under an identical downstream interface (RQ2) |

## 7. Conclusions

Code2Skill demonstrates that reusable agent skills can be manufactured automatically and at large scale from maintained source code, without requiring any agent trajectories, by using a regenerate-and-compare (source-body-blind reconstruction) loop as the grounding/verification mechanism. The resulting CodeSkillBank (1,006,822 records from 19,769 repositories) improves 57 of 72 protocol-matched agent evaluations (macro-average 42.90 → 47.90, +11.7% relative) and outperforms three trajectory-derived baselines on all seven shared benchmarks under an identical downstream interface. Practically, the experiments suggest that (a) skills help most when used to guide planning or critique an existing draft rather than direct first-pass generation, (b) compact summaries preserve most of the benefit while sharply cutting context cost, and (c) AI-generated (not just human-written) code can feed the same pipeline going forward.

> [!CAUTION]
> This is my own reading and may contain errors. One open question the paper does not fully resolve: because reconstruction/judgment/adjudication are all LLM-based rather than execution/test-based, the verification loop's reliability itself depends on the judge/adjudicator LLM's own competence at recognizing "equivalent" behavior — the human-annotation check (92% accurate, 80% retention-worthy) is reassuring but is a sample-based estimate, not a guarantee, and the paper explicitly flags this loop as a heuristic filter rather than a soundness proof.

# Experiments

- Dataset:
  - Skill-bank construction corpus: 19,769 GitHub repositories with >500 stars, scanned as of April 14, 2026 (median 3,133 stars, 82 merged PRs; 66.0% pushed within the previous year).
  - Downstream evaluation benchmarks: BigCodeBench, SWE-bench Verified, TerminalBench, LongCLI-Bench, AgentBench-OS, AIME 2026, HMMT 2025, GPQA.
  - RQ6 provenance study: 400-task subset of LiveCodeBench, plus 25 human + 25 AI-generated (GPT-5.1/Codex) implementations sampled from CodeSkillBank.
- Hardware: not explicitly specified in the paper (an industry research report from Ant International; no GPU/TPU counts or cluster details are given for either bank construction or evaluation).
- Optimizer: not applicable to the main results (skill construction and retrieval use frozen LLMs at inference time only); the RQ5 reinforcement-learning experiment reuses the coding-RL training procedure of Sun et al. (2026b) starting from a fixed Qwen3-32B SWE-World checkpoint, but the paper does not restate optimizer hyperparameters in the main text.
- Results:
  - RQ1: macro-average score 42.90 → 47.90 with CodeSkillBank retrieval (+11.7% relative), improving 57/72 protocol-matched evaluations across 9 model settings × 8 benchmarks.
  - RQ2: Code2Skill averages 49.5 across 7 shared benchmarks vs. 31.0 (Trace2Skill), 27.9 (ExpeL), 32.8 (SkillRL-Bank), and 40.1 for a per-benchmark oracle over the three baselines.
  - RQ3: post-generation critique is the most consistently positive insertion point; planning-time guidance improves all 8 tested cases; generation-time prompting is strong for DS4-Flash but inconsistent for Qwen3.5.
  - RQ4: summary rendering at $k=3$ cuts context by 88.9% (6,352 → 707 chars) while preserving or improving pass rate.
  - RQ5: post-generation review yields the largest RL resolve-rate gain (24% → 38%) versus policy/reward-side integration (→31–32%).
  - RQ6: human-code and AI-code skill banks reach 93.00% and 93.50% pass rate respectively on a 400-task LiveCodeBench subset, disagreeing on 16 of 400 tasks.
