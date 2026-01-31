# CLAUDE.md

## Project Overview

A personal knowledge repository documenting understanding of topics in computer science, machine learning, software development, and Japanese food culture. All content is written in Markdown.

## Directory Structure

- `machine-learning/` - ML papers and concepts, organized by year (e.g., `2017/`, `2023/`)
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
