# Meta Information

- URL: [AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents](https://arxiv.org/abs/2512.22387)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Vangala, B. P., Adibifar, A., Malik, T., & Gehani, A. (2025). AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents. arXiv:2512.22387.

# AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents

## Overview

This empirical study evaluates three LLM-based coding agents (Claude, OpenAI Codex, Gemini) across 300 generated projects in Python, JavaScript, and Java, measuring whether generated code can execute in a clean environment using only the dependencies the LLM specifies. Only 68.3% of generated projects execute without manual intervention, and runtime dependencies are on average 13.5× more than what LLMs declare.

> [!IMPORTANT]
> The paper introduces **executable reliability** as a distinct evaluation dimension from functional correctness. A project may pass unit tests (functional) yet still fail to run in a clean environment due to missing or incorrect dependency specifications (reproducible).

## Problem Formulation

**Input**: A natural language prompt $p_i$ and a programming language $L_j$.

**Output**: A generated project tuple $G(L_j, p_i) = \langle C_i^j, D_i^j, I_i^j \rangle$, where:
- $C_i^j$: the generated source code
- $D_i^j$: the declared dependencies (e.g., `requirements.txt`, `package.json`, `pom.xml`)
- $I_i^j$: setup/run instructions

**Executable Reliability Metric**: Given a clean environment $E$ containing only OS defaults and a standardized base of 91 packages, a project succeeds if it executes without any manual intervention after only installing $D_i^j$.

## Three-Layer Dependency Framework

The paper distinguishes three dependency types to quantify gaps:

| Layer | Symbol | Definition |
|-------|--------|------------|
| Claimed Dependencies | $D^c$ | Packages explicitly listed by the LLM in manifest files |
| Working Dependencies | $D^w$ | Actual packages required to run the project (after iterative debugging) |
| Runtime Dependencies | $D^r$ | Transitive closure of all packages loaded during execution |

**Completeness Gap**: $|D^w \setminus D^c|$ — the number of missing packages not declared by the LLM.

**Runtime Multiplier**: $|D^r| / |D^c|$ — ratio of actual runtime packages to declared packages. Average across all languages: **13.5×** (Python: 12.3×, JavaScript: 9.7×, Java: 18.4×).

## Dataset and Experimental Setup

**Prompts**: 100 standardized prompts designed to request realistic, multi-file projects. Prompts cover web APIs, data processing pipelines, CLI tools, and utility libraries.

**Distribution**: Each of the three agents received each prompt per language:
- Python: 40 prompts/agent × 3 agents = 120 projects
- JavaScript: 35 prompts/agent × 3 agents = 105 projects
- Java: 25 prompts/agent × 3 agents = 75 projects
- **Total: 300 projects**

**Environments**: AWS EC2 instances with fresh state reset between evaluations. Each environment starts with exactly 91 baseline packages (OS defaults + common build tools).

**Dependency Capture Tools**:
- Python: SciUnit (provenance capture via `ptrace`)
- JavaScript: `npm list --all` after execution
- Java: `mvn dependency:tree` output parsing

## Algorithm: Iterative Dependency Resolution

The paper defines a structured iterative resolution protocol to distinguish between dependency errors and code errors:

```
Algorithm: IterativeDepResolution(project P, env E)
Input:  Project P = (C, D, I), clean environment E
Output: ExecutionResult ∈ {Success, Partial, Failure}

1. Install D^c into E
2. Attempt to execute C following I
3. If execution succeeds → return Success
4. Collect error messages
5. For each ImportError / ClassNotFoundException / ModuleNotFoundError:
   a. Identify missing package name m
   b. Install m into E
   c. Retry execution
6. Apply minimal code fixes for syntax errors or incorrect paths (≤ 3 attempts)
7. If execution still fails → return Failure
8. If execution partially completes → return Partial
```

> [!NOTE]
> Steps 5–6 apply only to **post-failure analysis** to characterize error types. The primary reproducibility metric counts only projects succeeding at Step 3, with no manual intervention.

## Results

### Overall Reproducibility

| Outcome | Count | Percentage |
|---------|-------|------------|
| Success (no intervention) | 205 | 68.3% |
| Partial (some components run) | 14 | 4.7% |
| Failure | 81 | 27.0% |

### Language-Specific Reproducibility

| Language | Success Rate |
|----------|-------------|
| Python | 89.2% |
| JavaScript | 61.9% |
| Java | 44.0% |

Python's flat `requirements.txt` model has minimal nesting and straightforward pip resolution. Java's Maven POM XML with nested scopes (`<scope>test</scope>`, BOM imports) and multi-module projects create ambiguity LLMs consistently mishandle.

### Agent × Language Performance

| Agent | Python | JavaScript | Java |
|-------|--------|-----------|------|
| Gemini | 100% | ~64% | 28% |
| Claude | ~87% | ~65% | 80% |
| OpenAI Codex | 87.5% | ~57% | 24% |

> [!IMPORTANT]
> These specializations reflect training data distributions, not documented capabilities. Gemini's perfect Python reproducibility and poor Java performance and Codex's low Java score (24%) suggest significant variation in exposure to Java project structures during training.

### Completeness Gap Distribution

| Missing Packages ($|D^w \setminus D^c|$) | % of Projects |
|------------------------------------------|---------------|
| 0 (complete) | 87% |
| 1 | 8% |
| 2 | 3% |
| ≥3 | 2% |

Most dependency failures involve a small number of missing packages, yet that 13% gap causes substantial reproducibility failures.

**Commonly missed packages**:
- Python: `lxml`, `bcrypt`, `cryptography`
- JavaScript: `body-parser`, `dotenv`, `cors`
- Java: `junit`, `slf4j-simple`, `jackson-databind`

### Failure Mode Classification

Among 95 failed/partial projects:

| Error Type | Count | Percentage |
|------------|-------|------------|
| Code bugs (syntax, logic, path errors) | 50 | 52.6% |
| Not processed (timeout/infra) | 16 | 16.8% |
| Other | 15 | 15.8% |
| Dependency errors | 10 | 10.5% |
| Environment issues | 4 | 4.2% |

> [!NOTE]
> Code generation errors (52.6%) dominate over dependency specification errors (10.5%). The reproducibility problem is broader than just missing package lists — LLMs generate structurally incorrect code (wrong file paths, undefined variables, malformed syntax) that prevents execution regardless of dependencies.

## Comparison with Prior Work

| Study | Focus | Metric |
|-------|-------|--------|
| HumanEval (Chen et al., 2021) | Single-function correctness | Pass@k |
| SWE-bench (Jimenez et al., 2023) | Repository-level bug fixing | Patch success |
| **This paper** | Multi-file project reproducibility | Executable reliability |

Prior benchmarks test **functional correctness** assuming a working environment with pre-installed dependencies. This paper instead evaluates **from-scratch executability** in a clean environment, which is the realistic scenario for developers adopting AI-generated starter projects.

## Applicability

- **Who**: Engineering teams using LLMs (GitHub Copilot, Claude, ChatGPT) to bootstrap new projects or generate complete code repositories.
- **When**: When accepting AI-generated code that will be run in CI/CD pipelines or shared across team members.
- **Where**: Most critical in polyglot organizations where Java or JavaScript projects are generated — Python projects are substantially more reliable (89.2%).

**Practical implication**: At organizational scale (100 generated projects/month), the 31.7% failure rate translates to ~32 projects requiring manual debugging averaging 15 minutes each — roughly **8 developer-hours per 100 AI-assisted project initializations**.

## Limitations

1. **Scope**: Only 300 projects across 3 agents and 3 languages. Prompt diversity (100 prompts) may not represent the full distribution of real-world project generation requests.
2. **Static environment**: The baseline of 91 packages is fixed; actual developer machines vary widely.
3. **No LLM refinement**: The study tests single-shot generation, not iterative prompting with error feedback loops.
4. **Language coverage**: Go, Rust, TypeScript, and other popular languages are not evaluated.
