# CLAUDE.md

## Project Overview

A personal knowledge repository documenting understanding of topics in computer science, machine learning, software development, and Japanese food culture. All content is written in Markdown.

## Directory Structure

- `machine-learning/` - ML papers and concepts, organized by year (e.g., `2017/`, `2023/`)
- `software-engineering/` - Software engineering topics, organized by category then year (e.g., `architecture/2025/`, `databases/2025/`)
- `foods/` - Food culture documentation (currently Japanese seasonal foods)
- `dod/` - Definition of Done guidelines (English and Japanese)
- `scripts/` - Utility shell scripts
- `.github/` - PR template

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

All documents must:
- Contain concrete, detailed sentences (not vague "I understand" statements)
- Describe applicability conditions (who, when, where)
- Include license and copyright information

ML/CS documents must additionally have:
- Clear Input/Output specifications
- Algorithms described with pseudocode
- Datasets explicitly explained
- Clear calculation order
- Differentiation between similar algorithms

### PR Workflow

- Branch from `main`, use descriptive branch names
- PR template requires: Objective, Effect, Test, Note, and DoD checklist

## Output Formats

- In English.
- Please git add and commit changes with meaningful messages if you edit files.
