---
name: summarize-article
description: Fetch a general article (ACM DL, IEEE, blog posts, etc.) and create a structured summary following the project's document conventions and DoD checklist
user-invocable: true
---

Summarize a general article (non-ML-specific) and save it as a Markdown document in the appropriate topic directory.

The user will provide a URL to an article (e.g., `https://dl.acm.org/...`, `https://ieeexplore.ieee.org/document/...`, or any web article).

## Steps

1. **Fetch the article**: Use WebFetch to retrieve the article content from the provided URL. Use the prompt: "Extract the full article content including: title, authors, abstract or introduction, all section headings and their content, key concepts, experimental results or evaluations if any, and references."
   - If the URL returns a paywall or restricted content, inform the user and ask them to paste the content or provide an accessible URL.
2. **Determine the topic directory**: Ask the user which top-level directory to save the document in. List existing directories and offer to create a new one using the `create-topic` skill if none fit.
3. **Determine the publication year**: Extract the year from the article metadata (publication date, copyright year, etc.). If unclear, ask the user.
4. **Determine the organization**: Check the target directory's structure:
   - If organized by year (like `machine-learning/`), save to `<topic>/<year>/`.
   - If organized by category, ask the user which subdirectory to use.
   - If flat, save directly in the topic directory.
   - Create subdirectories as needed with `mkdir -p`.
5. **Generate the filename**: Run `echo "<article title>" | bash scripts/title-converter.sh` to convert the article title to a kebab-case filename.
6. **Write the summary** to the determined path following the Document Template below.

## Document Template

The summary MUST follow this structure:

```markdown
# Meta Information

- URL: [<Article Title>](<original URL>)
- LICENSE: <license information — check the publisher's copyright notice>
- Reference: <authors> (<year>). <title>. <venue/publisher>.

# <Follow the article's own section structure>

## <Section headings matching the article>

<Content summarized in concrete, specific sentences.>
<Use > [!NOTE] blocks for direct quotes or clarifications.>
<Use > [!TIP] blocks for helpful external references.>
<Use > [!IMPORTANT] blocks for critical information.>
<Use > [!CAUTION] blocks for personal interpretations that may contain errors.>

## Key Contributions

<Summarize the main contributions or takeaways of the article.>

## Comparison with Related Work

<How this work differs from or builds upon related approaches.>
<Applicability: who would use this, when, and where.>
```

## Summarization Rules

Follow these rules strictly when writing the summary. These come from the Definition of Done (DoD) checklist:

### Common Requirements
- Write concrete, detailed sentences that demonstrate understanding (NEVER write vague statements like "I understand" or "this is important")
- Describe applicability conditions: who would use this, when, and where
- Include license and copyright information in the Meta Information section

### Content-Specific Rules
- If the article contains algorithms or technical mechanisms, describe them with pseudocode or step-by-step explanation
- If the article contains evaluations or experiments, summarize key quantitative results
- If the article contains mathematical notation, use LaTeX/MathJax ($...$) and define variables before use
- Compare with related or alternative approaches, highlighting what is new or different
- Use tables for terminology definitions, comparisons, and result summaries

### Style Conventions (from existing documents)
- Use `> [!NOTE]` for direct quotes from the article
- Use `> [!TIP]` for links to external references or tutorials
- Use `> [!IMPORTANT]` for critical details not obvious from the article
- Use `> [!CAUTION]` for personal interpretations that may contain errors
- Content MUST be written in English

## Don'ts

- Do NOT copy-paste large blocks of text from the article. Always paraphrase and summarize in your own words.
- Do NOT create tables of experimental results that merely duplicate the article. Only include key results with context.
- Do NOT assume the article is about machine learning. Adapt the summary structure to the article's actual content.
