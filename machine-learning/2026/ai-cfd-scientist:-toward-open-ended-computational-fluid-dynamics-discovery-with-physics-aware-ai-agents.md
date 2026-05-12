# Meta Information

- URL: [AI CFD Scientist: Toward Open-Ended Computational Fluid Dynamics Discovery with Physics-Aware AI Agents](https://arxiv.org/abs/2605.06607)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Somasekharan, N., Pathak, R., Dhanakoti, M., Zhang, T., Yue, L., Zhu, A., & Pan, S. (2026). AI CFD Scientist: Toward Open-Ended Computational Fluid Dynamics Discovery with Physics-Aware AI Agents. arXiv:2605.06607.

# AI CFD Scientist

## Overview

AI CFD Scientist is an open-source multi-agent framework that automates the full cycle of computational fluid dynamics (CFD) research: literature-guided ideation, validated solver execution, physics-level result verification via vision-language models (VLMs), source-code modification for novel physics models, and figure-grounded manuscript generation.

The system targets CFD practitioners (graduate students, researchers, simulation engineers) who want to systematically explore turbulence model hyperparameters, implement custom physics closures, and discover novel model corrections — without manually managing solver configurations, mesh independence checks, or post-processing pipelines. The backend solver is **OpenFOAM** (via the Foam-Agent interface), but the framework's architecture is solver-agnostic.

The core insight is that **solver convergence is not the same as physical validity**. Many failure modes — incorrect field magnitudes, missing post-processor outputs, degenerate flow patterns — are only visible in rendered flow-field imagery, not in solver exit codes or log files. This motivates the VLM physics-verification gate at the center of the system.

## Design Principles (P1–P5)

| Principle | Statement |
|---|---|
| P1 | Physical validity requires image-level inspection; log files alone are insufficient |
| P2 | Source-code modifications (C++ library injection) are first-class research variables, not side effects |
| P3 | Mesh independence is mandatory before accepting any quantitative result |
| P4 | Agents may not hallucinate experiments or lower validity thresholds to pass a case |
| P5 | All manuscript claims must trace to figures or numerical values from validated runs |

P1 is the most operationally consequential: the VLM gate prevents the system from declaring success on cases where OpenFOAM exits cleanly but the resulting flow field is physically nonsensical. P4 ensures that the verification logic cannot be bypassed or diluted — a necessary safeguard for autonomous systems making scientific claims.

## System Architecture

### Input and Output

- **Input**: A natural-language research objective (e.g., "Investigate the sensitivity of backward-facing step reattachment length to RANS closure choice"), an optional reference dataset (experimental or DNS data), and an OpenFOAM case directory.
- **Output**: A structured experimental matrix (parameter settings → convergence status → physics verdict → extracted metrics), optionally a LaTeX/Markdown manuscript, and for open-ended discovery tasks, C++ source-code modifications with measured improvement over the baseline.

### Three Coupled Pathways

**Pathway 1 – Regular Experimentation**

Handles parametric sweeps within a fixed OpenFOAM solver configuration:

```
Input: research objective, OpenFOAM case
1. Retrieve related literature (Semantic Scholar API)
2. Filter proposed configurations for novelty (string-similarity deduplication)
3. Validate requirements (boundary conditions, physical ranges)
4. Execute mesh-independence study (baseline + refined mesh)
5. Run each configuration via Foam-Agent
6. Pass rendered flow field to VLM physics gate
7. Record verdict: PASS / FAIL / UNRESOLVED
```

**Pathway 2 – Code Modification**

Extends Pathway 1 by autonomously compiling case-local C++ libraries for custom physics models:

```
Input: natural-language description of new physics model
1. Generate C++ source for OpenFOAM function object or turbulence model
2. Compile locally; capture compiler errors
3. Feed compiler errors back to LLM for iterative fix (up to N attempts)
4. Smoke-test against a simple verification case
5. Deploy into the target case and execute Pathway 1
```

The key constraint is that the library is compiled **locally** (not injected into the system OpenFOAM installation), making each modification isolated and reproducible.

**Pathway 3 – Open-Ended Discovery**

Drives an autonomous hypothesis loop:

```
Input: reference dataset (e.g., DNS periodic hill data), baseline solver
Repeat until budget exhausted:
  1. Propose source-term modification (structured hypothesis)
  2. Compile C++ library for proposed modification
  3. Execute and verify (Pathways 1+2)
  4. Compute RMSE against reference data
  5. Compare with best-known result; store improvement
Output: C++ source, learned coefficients, RMSE improvement
```

### Agent Inventory

| Agent | Role |
|---|---|
| Literature Retrieval | Queries Semantic Scholar; extracts relevant method names and parameters |
| Novelty Filter | Removes near-duplicate configurations via string similarity |
| Specification Agent | Translates research objective into concrete OpenFOAM parameter sets |
| Validation Agent | Checks physical feasibility of proposed parameters |
| Mesh Independence Agent | Runs two mesh resolutions; invokes Richardson/GCI analysis at 5% threshold |
| Code Modification Agent | Generates, compiles, and debugs C++ function objects |
| VLM Quality Filter | Checks rendered images for missing deliverables, magnitude errors, convergence issues |
| VLM Physics Checker | Domain-specific check for flow-field plausibility (e.g., reversed velocity where not expected) |
| Rerun Controller | Decides whether to retry failed cases or mark as UNRESOLVED |
| Manuscript Writer | Assembles figure-grounded write-up referencing only validated results |

## VLM Physics-Verification Gate

The gate is a two-stage visual check applied to every completed simulation:

1. **Quality filter** (domain-agnostic): detects missing output files, empty plots, crashed post-processors, and scale anomalies in rendered images.
2. **Physics checker** (domain-specific): applies CFD knowledge to flag physically implausible results — e.g., velocity profiles inconsistent with expected turbulent boundary layer shape, pressure fields with spurious oscillations, or reattachment lengths outside physically credible bounds.

**Input**: A rendered flow-field image (PNG/SVG), the case configuration, and optionally reference data overlaid on the same plot.
**Output**: A verdict (`PASS`, `FAIL`, `UNRESOLVED`) with a text explanation traceable to specific visual features.

In the planted-failure ablation (16 injected failures across 4 categories), the gate detected 14/16 failures:

| Failure Category | Injected | Detected |
|---|---|---|
| missing_deliverable | 4 | 4 |
| wrong_magnitude_metric | 4 | 4 |
| broken_postprocessing | 4 | 4 |
| convergence_not_settled | 4 | 2 |

The two missed cases were partially-converged simulations where visual indicators were ambiguous. Solver-level checks alone detected 0/16 because OpenFOAM returned exit code 0 in all cases.

## Mesh Independence Workflow

Mesh independence is a mandatory gate (P3) applied before accepting any quantitative result:

```
1. Run baseline mesh (call it M1)
2. Generate refined mesh M2:
   - ~10% refinement in near-wall regions
   - ~5% refinement in bulk flow
3. Execute both cases
4. Compare: local field values + global integral metrics
5. If max relative difference > 5%:
   - Run Richardson extrapolation
   - Compute Grid Convergence Index (GCI)
   - Flag result with uncertainty band
6. If difference ≤ 5%: accept M1 result
```

This prevents the system from reporting numerically under-resolved results as if they were converged solutions — a common failure mode in automated CFD workflows.

## Experimental Tasks (T1–T5)

| Task | Type | Configuration | Key Result |
|---|---|---|---|
| T1 | Parametric (RANS) | 4 turbulence closures for backward-facing step | VLM flagged post-processor error in 1 case; 3 valid results tabulated |
| T2 | Parametric (transient) | 7 Reynolds numbers for jet/plume | Correct velocity scaling recovered across all cases |
| T3 | Code modification | Custom viscosity model (power-law) | Autonomously compiled and validated against Newtonian limit |
| T4 | Code modification (SA) | Manual SA modifier baseline | Control case confirmed code path validity for T5 |
| T5 | Open-ended discovery | SA source-term correction via Gaussian patches | RMSE reduced from 0.004297 to 0.003958 (7.89%) on periodic hill |

## T5: Discovered Spalart-Allmaras Correction

The open-ended discovery task targeted the Spalart-Allmaras (SA) one-equation turbulence model. The SA transport equation for the modified kinematic viscosity $\tilde{\nu}$ is:

```math
\begin{align}
  \frac{D\tilde{\nu}}{Dt} = c_{b1}\tilde{S}\tilde{\nu} + \frac{1}{\sigma}\left[\nabla\cdot((\nu+\tilde{\nu})\nabla\tilde{\nu}) + c_{b2}(\nabla\tilde{\nu})^2\right] - c_{w1}f_w\left(\frac{\tilde{\nu}}{d}\right)^2
\end{align}
```

The system proposed adding a spatially-localized source term to the production component:

```math
\begin{align}
  S_{\text{extra}} = \left[C_{\text{rec}}G_{\text{rec}} - C_{\text{sink}}G_{\text{sink}} + C_{\text{src}}G_{\text{src}} - C_{\text{tail}}G_{\text{tail}}\right] \cdot |\nabla\mathbf{U}| \cdot \tilde{\nu}
\end{align}
```

where each $G$ term is a **wall-normalized Gaussian spatial patch**:

```math
\begin{align}
  G_i(y^+) = \exp\!\left(-\frac{(y^+ - \mu_i)^2}{2\sigma_i^2}\right)
\end{align}
```

with $y^+ = y/d_w$ the wall-normalized height. The system executed 44 iterations to learn the four coefficients $\{C_{\text{rec}}, C_{\text{sink}}, C_{\text{src}}, C_{\text{tail}}\}$ and patch locations $\{\mu_i, \sigma_i\}$ by minimizing RMSE against DNS data for the periodic hill benchmark.

**Algorithm: Open-Ended SA Discovery**

```
Input: DNS periodic hill reference data, baseline SA case
Initialize: best_rmse = RMSE(baseline SA)

For iteration = 1 to 44:
  1. Propose Gaussian patch configuration (locations, widths, coefficients)
  2. Generate C++ OpenFOAM function object implementing S_extra
  3. Compile locally (retry up to 3 times on compiler error)
  4. Run SA+S_extra on periodic hill case
  5. Verify with VLM gate (P1)
  6. Check mesh independence (P3)
  7. Compute RMSE vs DNS reference
  8. If RMSE < best_rmse: update best_rmse, store coefficients
  9. Log iteration result

Output: C++ source with learned coefficients
        best_rmse = 0.003958 (vs baseline 0.004297)
```

> [!NOTE]
> The periodic hill benchmark (Re = 10,595, based on hill height) is a standard test case for separating-reattaching turbulent flows from the ERCOFTAC database. DNS reference data comes from Breuer et al. (2009).

## Comparison with Baseline AI-Scientist Systems

| Capability | ARIS | DeepScientist | AI CFD Scientist |
|---|---|---|---|
| Literature retrieval | ✓ | ✓ | ✓ |
| Code generation | ✓ | ✓ | ✓ |
| CFD execution | Partial | Partial | ✓ (Foam-Agent) |
| Mesh independence gate | ✗ | ✗ | ✓ (mandatory) |
| VLM physics verification | ✗ | ✗ | ✓ |
| Reference data alignment | ✗ | ✗ | ✓ |
| Conservative verdict handling | ✗ | ✗ | ✓ (UNRESOLVED) |
| Manuscript generation | ✓ | ✓ | ✓ (figure-grounded only) |

The critical differentiator is conservatism: ARIS and DeepScientist convert executed simulations into manuscript claims even when the physical validity is uncertain. AI CFD Scientist records `UNRESOLVED` and explicitly refuses to generate manuscript text for unverified results (P4, P5).

> [!IMPORTANT]
> The framework was evaluated using a single LLM backbone (GPT-4o / GPT-5.5). No LLM ablation sweep was performed, so the contribution of the agent architecture vs. the backbone capability cannot be fully disentangled from these results.

## Experiments

- **Datasets**: ERCOFTAC periodic hill DNS (Re=10,595, Breuer et al. 2009) for T5 quantitative validation; backward-facing step experimental data for T1; jet/plume DNS data for T2
- **Hardware**: Not specified in the paper
- **Solver**: OpenFOAM via Foam-Agent interface
- **LLM Backbone**: GPT-4o (specification/validation agents), GPT-5.5 (code modification and manuscript agents) — single backbone, no sweep
- **T5 Result**: Wall friction coefficient RMSE reduced from 0.004297 to 0.003958 over 44 discovery iterations
- **VLM Gate**: Detected 14/16 planted failures (87.5%) vs. 0/16 for solver-level checks alone

## Limitations

- Framework designed as **supervised scientific assistance**, not autonomous publication; a domain expert must review all outputs before use
- Single LLM backbone evaluation: no systematic comparison across different LLMs
- No automated CFD rubric exists for cross-framework comparison; T1–T4 evaluations used manual inspection
- T5 discovery was tested on one benchmark (periodic hill); transfer of the discovered correction to other geometries was not evaluated
- Convergence failure detection (T=convergence_not_settled category) had the lowest VLM gate accuracy (2/4), indicating that distinguishing partial convergence from full convergence remains difficult visually

> [!TIP]
> Code, prompts, run artifacts, and experimental matrices: https://github.com/csml-rpi/cfd-scientist
