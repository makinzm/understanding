# CLAUDE.md

## Project Overview

A personal knowledge repository documenting understanding of topics in computer science, machine learning, software development, and Japanese food culture. All content is written in Markdown.

## Directory Structure

- `machine-learning/` - ML papers and concepts, organized by year (e.g., `2017/`, `2023/`)
- `software-engineering/` - Software engineering topics, organized by category then year (e.g., `architecture/2025/`, `databases/2025/`)
- `foods/` - Food culture documentation (currently Japanese seasonal foods)
- `dance/` - Street dance fundamentals, organized by genre (`genre/<Name>/basic-steps.md`)
- `chess/` - Chess strategy fundamentals (no piece-movement rules), organized by game phase (`opening/`, `ending/`)
- `dod/` - Definition of Done guidelines (English and Japanese); `deliverables-dod.json` is the source of truth, see "Document Factory" below
- `scripts/` - Utility shell scripts, including the Document Factory generator (`sync-dod-skills.py`) and its Stop hook (`dod-review-check.sh`)
- `.github/` - PR template

## Document Factory

This repo is a "Document Factory": `dod/deliverables-dod.json` is the single source of truth for DoD items, each tagged with the directory `scopes` it applies to (`common` = every directory; otherwise a specific topic like `machine-learning`, `dance`, `chess`). `scripts/sync-dod-skills.py` materializes each item into one Skill (`.claude/skills/dod-understanding-<id>/`) and one Task-tool review Agent (`.claude/agents/dod-understanding-<id>.md`) — ids are prefixed `dod-understanding-` so they never collide with any other machine's global `dod-*` ids. After editing the JSON, re-run that script (tests: `scripts/test-dod-sync.sh`, picked up by `make test`), then manually update `dod/en.md`/`dod/ja.md` to match.

`scripts/dod-review-check.sh` (registered as a `Stop` hook in `.claude/settings.json`) enforces this scope-aware: when a `.md` file is written/edited under a tracked topic directory, it maps that directory to its required scopes and blocks completion until the Agents relevant to *those* scopes (never all of them) have actually been invoked via the Task tool. A dance document never triggers an ML-specific check, and vice versa.

## Commands

- `make test` - Run all test scripts (`scripts/test-*.sh`)
- `make clean` - No-op

## Conventions

### File Naming

- Use kebab-case for all filenames (e.g., `attention-is-all-you-need.md`)
- Use `scripts/title-converter.sh` to convert titles to filenames

### Content Structure

- Markdown with LaTeX/MathJax for math notation
- Include source links, references, and license/copyright info at top
- Use hierarchical headings with clear sections

### Math Notation

- **Inline math**: Use `$...$` for math within a sentence (e.g., `$x \in \mathbb{R}^{d}$`)
- **Block math**: Use a fenced code block with the `math` language tag for any display equation, even single-line. Do NOT use `$$...$$` as it causes rendering issues.
  - Single-line: use `\begin{align}...\end{align}` with one equation
  - Multi-line: use `\begin{align}...\end{align}` with `\\` line breaks

  ````markdown
  ```math
  \begin{align}
    y = Wx + b
  \end{align}
  ```
  ````

  ````markdown
  ```math
  \begin{align}
    y &= Wx + b \\
    \hat{y} &= \sigma(y)
  \end{align}
  ```
  ````

### Definition of Done (Quality Checklist)

See `dod/en.md` / `dod/ja.md` (mirrors of `dod/deliverables-dod.json`, the source of truth) for the full, scope-tagged list. Summary:

All documents must:
- Contain concrete, detailed sentences (not vague "I understand" statements)
- Describe applicability conditions (who, when, where)
- Include license and copyright information

ML/CS documents (`machine-learning/`) must additionally have:
- Clear Input/Output specifications
- Algorithms described with pseudocode
- Datasets explicitly explained
- Clear calculation order
- Differentiation between similar algorithms

Dance documents (`dance/`) must additionally:
- State each move's provenance, flagging thin/conflicting sources explicitly
- Describe concrete body mechanics (joint/body part, beat/count)
- State which dance style/family each move belongs to
- Cover each specifically-requested named move by name, not just implicitly

Chess documents (`chess/`) must additionally:
- Never explain how a piece moves (assumed prior knowledge)
- Explain why each principle/technique works, not only its name
- State which broader category each named concept belongs to
- Stay within the directory's declared game phase (no middlegame content in `opening/`/`ending/`)

### PR Workflow

- Branch from `main`, use descriptive branch names
- PR template requires: Objective, Effect, Test, Note, and DoD checklist

## Output Formats

- In English.
- Please git add and commit changes with meaningful messages if you edit files.
