---
name: collect-daily-trends
description: Collect daily AI & Tech trends from Hacker News, Reddit, and Hatena with anti-hallucination validation
user-invocable: true
---

Collect daily trends about AI Agents, software tools, and news from Japan and worldwide.

## Instructions

**Follow the complete instructions in:** `.github/prompts/daily-trends-prompt.md`

Read that file and follow all steps exactly. The prompt contains:
- NO HALLUCINATION policy
- Data fetching requirements
- Validation steps
- Output format
- PR creation workflow

## Key Points

1. **Read the prompt first**: `Read .github/prompts/daily-trends-prompt.md`
2. **Follow every step** in order (compliance check, fetch data, parse, generate, validate, PR)
3. **No shortcuts**: Must run `python3 scripts/fetch_trends.py` for real data
4. **Validate everything**: Run all grep checks before committing

## Difference from Automated Workflow

- **Manual trigger**: You invoke this skill when needed
- **Same workflow**: Uses identical instructions from `.github/prompts/daily-trends-prompt.md`
- **Same validation**: Same anti-hallucination checks apply

## Usage

```
/collect-daily-trends
```

This will execute the full daily trends collection workflow with real data fetching and validation.
