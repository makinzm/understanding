---
name: create-topic
description: Scaffold a new topic directory with README and document template following the project's conventions and DoD checklist
user-invocable: true
---

Create a new top-level topic directory with a README and suggested structure.

The user will provide a topic name as the argument (e.g., `software-engineering`, `distributed-systems`).

## Steps

1. **Normalize the topic name**: Run `echo "<topic name>" | bash scripts/title-converter.sh` to get a kebab-case directory name. Strip the `.md` extension from the output to use as the directory name.
2. **Create the directory**: `mkdir -p <topic-name>/`
3. **Ask the user for organization style**: Present these options:
   - **By category** (default): Subdirectories grouping related topics (e.g., `architecture/`, `practices/`, `databases/`). Best for concept-based topics.
   - **By year**: Year-based subdirectories (e.g., `2025/`). Best for dated content like papers or conference talks.
   - **Flat**: All documents directly in the topic directory. Best when there are few documents.
4. **Ask the user for subtopic categories**: Based on the organization style, ask what subcategories they want. For example, if the topic is `software-engineering`, suggest categories like `architecture/`, `practices/`, `databases/`, `ai-engineering/` but let the user decide.
5. **Create subdirectories** if the user chose category-based organization.
6. **Write the README.md** to `<topic-name>/README.md` following the README Template below.
7. **Update the project CLAUDE.md**: Add the new directory to the `## Directory Structure` section, following the existing format.

## README Template

```markdown
# <Topic Title>

<One-line description of what this directory covers.>

## Directory Structure

<Describe the chosen organization style and list subdirectories.>

### Document Template

Each document in this directory should follow this structure:

- **Meta Information**: Source URL, LICENSE, and Reference at the top
- **Sections**: Organized by the topic's natural structure
- **Concrete content**: Specific sentences demonstrating understanding (not "I understand X")
- **Applicability**: Who would use this, when, and where
- **Comparisons**: How this differs from similar concepts or alternatives

### File Naming

- Use kebab-case for all filenames (e.g., `clean-architecture.md`)
- Generate filenames with: `echo "Title" | bash scripts/title-converter.sh`
```

## Document Template

When creating the first document or when the user asks to add a document to this topic, use this structure:

```markdown
# Meta Information

- URL: [<Title>](<source URL>)
- LICENSE: <license information>
- Reference: <author(s)> (<year>). <title>. <publisher/source>.

# <Topic Title>

## Overview

<Concise summary of what this topic is and why it matters.>
<Applicability: who uses this, when, and where.>

## Key Concepts

<Core concepts explained with concrete details.>
<Use > [!NOTE] blocks for direct quotes.>
<Use > [!TIP] blocks for helpful external references.>
<Use > [!IMPORTANT] blocks for critical information.>
<Use > [!CAUTION] blocks for personal interpretations.>

## How It Works

<Detailed explanation with examples, diagrams (Mermaid), or pseudocode as appropriate.>

## Comparison with Alternatives

<How this differs from similar approaches, tools, or methodologies.>

## References

<Additional resources and links.>
```

## Rules

Follow these rules strictly. These come from the Definition of Done (DoD) checklist:

### Common Requirements
- Write concrete, detailed sentences that demonstrate understanding (NEVER write vague statements like "I understand" or "this is important")
- Describe applicability conditions: who would use this, when, and where
- Include license and copyright information in the Meta Information section

### Content-Specific Rules
- Compare with alternatives or similar concepts
- Include practical examples where applicable
- Use tables for comparisons and terminology definitions
- Content can be written in English or Japanese, matching user preference
