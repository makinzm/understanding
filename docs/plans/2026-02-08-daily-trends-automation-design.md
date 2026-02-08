# Daily Trends Automation - Design Document

**Date**: 2026-02-08
**Status**: Approved
**Author**: Claude (via brainstorming session)

## Overview

Automated system for collecting daily trends about AI Agents, software tools, and news from Japan and worldwide. Runs at 6:00 AM JST via GitHub Actions and creates pull requests with curated content.

## Requirements

### Functional Requirements

1. **Content Weighting**:
   - 60% AI (agents, frameworks, LLMs, autonomous systems)
   - 30% Tech (software development, tools, engineering)
   - 10% Others (world news, Japan news)

2. **Sources**:
   - Hacker News (via official API)
   - Reddit (AI and tech subreddits via JSON API)
   - Hatena Bookmark (Japanese tech news)

3. **Compliance**:
   - Respect robots.txt for all sources
   - Implement rate limiting (1-2 sec delays)
   - Use proper User-Agent headers
   - Only access public endpoints

4. **Output**:
   - Daily markdown file in `trends/YYYYMMDD-daily.md`
   - Structured format with ratings (★★★, ★★, ★)
   - Pull request creation for review

5. **Schedule**:
   - Daily at 6:00 AM JST (21:00 UTC previous day)
   - Manual trigger available for testing

### Non-Functional Requirements

1. Reliability: Handle source failures gracefully
2. Compliance: Document and follow all ToS requirements
3. Maintainability: Separate prompt from workflow configuration
4. Transparency: Log collection summary and any issues

## Architecture

### Components

1. **GitHub Workflow** (`.github/workflows/daily-trends.yml`)
   - Cron scheduler triggering at 6:00 AM JST
   - Uses Claude Code action with prompt file
   - Permissions: write contents + pull-requests

2. **Prompt Instructions** (`.github/prompts/daily-trends-prompt.md`)
   - Step-by-step collection instructions
   - Source-specific fetching methods
   - Rating criteria and weighting logic
   - Output format specification
   - PR creation commands

3. **Compliance Documentation** (`.github/prompts/compliance-notes.md`)
   - ToS requirements for each source
   - Rate limits and User-Agent specs
   - Review schedule (quarterly)
   - Revision history

4. **Output Directory** (`trends/`)
   - Daily markdown files
   - README explaining format and automation

### Data Flow

```
Scheduler (6:00 AM JST)
  ↓
GitHub Actions Workflow
  ↓
Claude Code Action (reads prompt file)
  ↓
Data Collection:
  - Hacker News API → Top 30 stories → Filter 50+ points
  - Reddit JSON API → 12 subreddits × 25 posts → Filter 100+ upvotes
  - Hatena Bookmark → 3 categories → Filter 10+ bookmarks
  ↓
Analysis & Rating:
  - Categorize by AI/Tech/Other
  - Assign ★★★/★★/★ ratings
  - Apply 60/30/10 weighting
  ↓
Output Generation:
  - Create trends/YYYYMMDD-daily.md
  - Format with sections and summaries
  ↓
Git Operations:
  - Create branch: trends/YYYYMMDD-daily
  - Commit markdown file
  - Push to origin
  ↓
Pull Request Creation:
  - Title: "Daily Trends: YYYY-MM-DD"
  - Body: Following PR template with DoD checklist
  - Auto-label: automated, trends
```

## Implementation Details

### Source-Specific Methods

**Hacker News**:
- Endpoint: `https://hacker-news.firebaseio.com/v0/topstories.json`
- Method: Official Firebase API
- Rate: 1 second between item fetches
- Filter: 50+ points

**Reddit**:
- Endpoint: `https://www.reddit.com/r/{subreddit}/hot.json?limit=25`
- Method: Bash curl with JSON parsing
- Rate: 1 second between subreddits
- Filter: 100+ upvotes or 50+ comments
- Subreddits:
  - AI: MachineLearning, LocalLLaMA, OpenAI, ClaudeAI, AutoGPT
  - Tech: programming, SideProject, opensource, webdev
  - Other: worldnews, japan

**Hatena Bookmark**:
- URLs: `https://b.hatena.ne.jp/hotentry/{category}`
- Method: WebFetch with extraction prompt
- Rate: 2 seconds between categories
- Categories: it, technology, all
- Filter: 10+ bookmarks

### Rating System

**★★★ (High Priority)**: Direct interest match
- Examples: New AI agent framework, major LLM release, breakthrough dev tool

**★★ (Medium Priority)**: Related but indirect
- Examples: ML research paper, programming language update, tech company news

**★ (Low Priority)**: General interest
- Examples: Background reading, tangential topics

**Weighting Logic**:
- Count articles by category (AI/Tech/Other)
- Ensure ★★★ articles approximate 60/30/10 distribution
- Include ★★ and ★ articles to fill out each section

### Output Format

Markdown file structure:
1. Title with date
2. Three main sections (AI/Tech/Other) with emoji headers
3. Each section has Top Picks (★★★) and Notable (★★)
4. Each article entry includes: title link, summary, source, metrics
5. Collection Summary footer with sources and compliance status
6. Notes section for issues or observations

### Error Handling

- Source timeouts: Skip and note in summary
- Rate limiting (429): Back off and retry once, then skip
- Forbidden (403): Skip source and document in notes
- No articles meeting criteria: Create report with note about low activity
- GitHub API failure: Commit locally, PR in next run

## Testing Strategy

1. **Manual Trigger**: Use `workflow_dispatch` to test before scheduled run
2. **Compliance Check**: Verify User-Agent and rate limits in logs
3. **Output Validation**: Check markdown format and link validity
4. **PR Template**: Ensure DoD checklist is properly formatted

## Deployment Plan

1. Create all files in `.github/` and `trends/` directories
2. Commit with message: "Add daily trends automation workflow"
3. Set up GitHub Secret: `CLAUDE_CODE_OAUTH_TOKEN`
4. Test with manual workflow dispatch
5. Review first automated PR before merging
6. Monitor for first week, adjust as needed

## Maintenance

- **Weekly**: Review PR quality and coverage
- **Monthly**: Check for new relevant subreddits or sources
- **Quarterly**: Re-verify ToS compliance (update compliance-notes.md)
- **As needed**: Adjust weighting or rating criteria based on results

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Source blocks scraping | No data from that source | Use official APIs where possible; graceful degradation |
| Rate limiting penalties | Temporary access loss | Conservative delays; respect all limits |
| ToS changes | Compliance violation | Quarterly reviews; automated checks |
| Low quality results | Poor curation | Adjustable rating criteria; manual review via PR |
| GitHub Actions quota | Workflow disabled | Monitor usage; optimize fetch counts |

## Success Metrics

- Daily PR created successfully (>95% uptime)
- 10-15 articles per report (minimum)
- Weighting within ±10% of target (60/30/10)
- No ToS violations or blocks
- PR merge rate >80% (indicating quality)

## Future Enhancements

- Add more sources (dev.to, Medium, Qiita)
- AI-generated trend analysis summary
- Weekly digest roll-up
- Email notifications for high-priority items
- Machine learning to improve rating accuracy
