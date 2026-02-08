# Daily Trends

This directory contains automated daily trend reports collecting AI, tech, and general news.

## Directory Structure

Reports are organized by year and month:
```
trends/
├── 2026/
│   ├── 02/
│   │   ├── 20260208-daily.md
│   │   ├── 20260209-daily.md
│   │   └── ...
│   └── 03/
│       └── ...
└── README.md
```

## Format

Files are named `YYYYMMDD-daily.md` and contain:
- **60% AI content**: AI agents, frameworks, LLMs, autonomous systems
- **30% Tech content**: Software development, tools, engineering
- **10% Other content**: World news, Japan news, general interest

**Data Source**: All content is fetched from real sources via `scripts/fetch_trends.py` - no hallucinated or generated content.

## Sources

- **Hacker News**: Top stories via official API
- **Reddit**: AI and tech subreddits via JSON API
- **Hatena Bookmark**: Japanese tech news

## Automation

Reports are generated daily at 6:00 AM JST via GitHub Actions (`.github/workflows/daily-trends.yml`).

Each report triggers a pull request for review before merging.

**Manual Triggering**: You can also run the workflow ad-hoc anytime. See [MANUAL-TRIGGER.md](MANUAL-TRIGGER.md) for instructions.

## Compliance

All data collection respects terms of service, robots.txt, and rate limits.
See `.github/prompts/compliance-notes.md` for details.
