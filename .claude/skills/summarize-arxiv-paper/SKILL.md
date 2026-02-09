---
name: summarize-arxiv-paper
description: Fetch an arXiv paper via ar5iv and create a structured summary following the project's ML document conventions and DoD checklist
user-invocable: true
---

Summarize an arXiv paper and save it as a Markdown document in the `machine-learning/` directory.

The user will provide an arXiv URL (e.g., `https://arxiv.org/abs/2312.00752`) as the argument.

## Steps

1. **Convert URL to ar5iv**: Convert the arXiv URL from `https://arxiv.org/abs/<ID>` to `https://ar5iv.labs.arxiv.org/html/<ID>` to get an HTML-rendered version of the paper.
2. **Fetch the paper**: Use WebFetch to retrieve the paper content from the ar5iv URL. Use the prompt: "Extract the full paper content including: title, authors, abstract, all section headings and their content, mathematical formulas, algorithms, experimental results, datasets, and references."
3. **Extract the publication year**: Get the year from the arXiv ID (e.g., `2312.00752` → `2023`, `1706.03762` → `2017`).
4. **Generate the filename**: Run `echo "<paper title>" | bash scripts/title-converter.sh` to convert the paper title to a kebab-case filename.
5. **Create the year directory** if it does not exist: `mkdir -p machine-learning/<year>/`
6. **Write the summary** to `machine-learning/<year>/<filename>.md` following the Document Template below.

## Document Template

The summary MUST follow this structure, modeled after existing documents in this repository:

```markdown
# Meta Information

- URL: [<Paper Title>](<original arxiv URL>)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: <authors> (<year>). <title>. <venue/journal>.

# <Follow the paper's own section structure>

## <Section headings matching the paper>

<Content summarized in concrete, specific sentences.>
<Mathematical notation using LaTeX/MathJax ($...$).>
<Specify tensor/matrix dimensions explicitly.>
<Use > [!NOTE] blocks for direct quotes or clarifications.>
<Use > [!TIP] blocks for helpful external references.>
<Use > [!IMPORTANT] blocks for critical information.>

# Experiments

- Dataset: <list all datasets used>
- Hardware: <if mentioned>
- Optimizer: <if mentioned>
- Results: <key quantitative results>

```

## Summarization Rules

Follow these rules strictly when writing the summary. These come from the Definition of Done (DoD) checklist:

### Common Requirements
- Write concrete, detailed sentences that demonstrate understanding (NEVER write vague statements like "I understand" or "this is important")
- Describe applicability conditions: who would use this, when, and where
- Include license and copyright information in the Meta Information section

### ML/CS Requirements
- **Clear Input/Output**: For each major component or layer, specify the input and output dimensions (e.g., $x \in \mathbb{R}^{n \times d}$)
- **Algorithms with pseudocode**: Describe key algorithms using pseudocode or step-by-step mathematical formulation
- **Datasets**: Explicitly list all datasets used, including splits (train/dev/test sizes) when available
- **Clear calculation order**: Present mathematical operations in the order they are computed
- **Differences from similar algorithms**: Compare with related or predecessor methods, highlighting what is new

### Style Conventions (from existing documents)
- Use `> [!NOTE]` for direct quotes from the paper
- Use `> [!TIP]` for links to external references or tutorials
- Use `> [!IMPORTANT]` for critical details not obvious from the paper
- Use `> [!CAUTION]` for personal interpretations that may contain errors
- Define all mathematical variables before using them, with explicit dimensions
- Use tables for terminologies, comparisons, and experimental results
- Content can be written in English or Japanese, matching the paper's language or user preference

## Don'ts

- Do NOT copy-paste large blocks of text from the paper. Always paraphrase and summarize in your own words.
- Do NOT create table of experimental results because they are already present in the paper. Only include key results with context.
