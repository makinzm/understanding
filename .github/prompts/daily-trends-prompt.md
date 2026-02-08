# Daily AI & Tech Trends Collection

You are tasked with collecting daily trends about AI Agents, software tools, and news from Japan and worldwide. Follow these instructions carefully.

## Overview

Create a curated daily trends report with weighted focus:
- **60% AI**: AI agents, frameworks, LLMs, autonomous systems, AI tooling
- **30% Tech**: Software development, developer tools, programming, engineering
- **10% Others**: World news, Japan news, general interest

## Step 1: Compliance Checks

Before fetching any data, review `.github/prompts/compliance-notes.md` to understand ToS requirements for each source.

**Required checks:**
1. Verify robots.txt compliance (documented in compliance-notes.md)
2. Implement rate limiting:
   - Hacker News API: 1 second between requests
   - Reddit JSON API: 1 second between requests (max 60/minute)
   - Hatena Bookmark: 2 seconds between requests
3. Use proper User-Agent headers as specified in compliance-notes.md

**If any source is blocked or rate-limited**: Skip it gracefully and note in the output.

## Step 2: Data Collection

### Source 1: Hacker News (AI + Tech focus)

**API**: `https://hacker-news.firebaseio.com/v0/topstories.json`

1. Fetch top 30 story IDs
2. For each story ID, fetch: `https://hacker-news.firebaseio.com/v0/item/{id}.json`
3. Extract: title, url, score (points), descendants (comments), time
4. Filter for stories with 50+ points
5. Use 1-second delay between item fetches

### Source 2: Reddit (AI + Tech subreddits)

**AI-focused (60% weight):**
- r/MachineLearning
- r/LocalLLaMA
- r/OpenAI
- r/ClaudeAI
- r/AutoGPT

**Tech-focused (30% weight):**
- r/programming
- r/SideProject
- r/opensource
- r/webdev

**General (10% weight):**
- r/worldnews
- r/japan

**Method**: Use Bash with curl to fetch `.json` endpoint:
```bash
curl -H "User-Agent: web:understanding-trends:v1.0.0 (by /u/YOUR_USERNAME)" \
  "https://www.reddit.com/r/[subreddit]/hot.json?limit=25"
```

Extract from each post: title, url (permalink for discussions), ups (upvotes), num_comments, created_utc

Filter: 100+ upvotes or 50+ comments

Use 1-second delay between subreddit fetches.

### Source 3: Hatena Bookmark (Japan tech news)

**RSS Feeds** (preferred method):
- `https://b.hatena.ne.jp/hotentry/it.rss` (IT/Tech)
- `https://b.hatena.ne.jp/hotentry/technology.rss` (Technology)
- `https://b.hatena.ne.jp/hotentry/all.rss` (General - for 10% Others)

**Method**: Use WebFetch or Bash curl to fetch RSS feeds and parse XML:
- Article titles (`<title>` tags)
- Original URLs (`<link>` tags - not Hatena intermediary pages)
- Bookmark counts (in `<description>` or `<hatena:bookmarkcount>`)
- Publication dates (`<pubDate>`)

**Alternative**: If RSS parsing is difficult, use web pages:
- `https://b.hatena.ne.jp/hotentry/it`
- Extract with WebFetch tool

Filter: 10+ bookmarks

Use 2-second delay between category fetches.

## Step 3: Content Analysis & Rating

For each collected article, assign a rating:

**★★★ (High Priority)** - Direct match to weighted interests:
- **AI (60%)**: AI agent frameworks/tools, Claude/GPT/LLM developments, autonomous agents, agent orchestration, AI coding tools, agent benchmarks
- **Tech (30%)**: New dev tools, framework releases, architecture patterns, performance optimizations, developer productivity
- **Others (10%)**: Major world events, significant Japan news, tech industry shifts

**★★ (Medium Priority)** - Related but indirect:
- **AI**: General ML research, AI companies, model releases, AI ethics
- **Tech**: Programming languages, DevOps, cloud services, APIs
- **Others**: Startup news, tech policy, industry analysis

**★ (Low Priority)** - General interest:
- Tangentially related topics
- Background reading

**Distribution goal**: Aim for ~60% of ★★★ articles to be AI-related, ~30% Tech, ~10% Others.

## Step 4: Generate Output

Create file: `trends/YYYYMMDD-daily.md` (use today's date)

**Format**:

```markdown
# Daily AI & Tech Trends - YYYY-MM-DD

## 🤖 AI Highlights (60%)

### Top Picks ★★★

- **[Article Title](url)** - [1-2 sentence summary]
  - Source: [HN/Reddit/Hatena] | Engagement: [points/upvotes/bookmarks] | Comments: [count]

### Notable ★★

- **[Article Title](url)** - [Brief description]
  - Source: [source] | Engagement: [metrics]

## 💻 Tech Highlights (30%)

### Top Picks ★★★

- **[Article Title](url)** - [1-2 sentence summary]
  - Source: [source] | Engagement: [metrics]

### Notable ★★

- **[Article Title](url)** - [Brief description]
  - Source: [source] | Engagement: [metrics]

## 🌍 World & Japan News (10%)

### Top Picks ★★★

- **[Article Title](url)** - [1-2 sentence summary]
  - Source: [source] | Engagement: [metrics]

### Notable ★★

- **[Article Title](url)** - [Brief description]
  - Source: [source] | Engagement: [metrics]

---

## Collection Summary

- **Total Articles Analyzed**: [count]
- **Sources Checked**:
  - Hacker News: ✅ ([count] stories fetched at [timestamp])
  - Reddit: ✅ ([count] subreddits, [count] posts)
  - Hatena Bookmark: ✅ ([count] categories, [count] bookmarks)
- **Compliance Status**: All rate limits respected, robots.txt followed

## Notes

[Any issues encountered, sources skipped, or relevant observations]
```

**Content guidelines**:
- All URLs must be direct links to original articles (not intermediary pages)
- Summaries should be concise and informative
- For Reddit discussions, use permalink URLs
- For Hatena, translate Japanese titles to English if needed
- Include engagement metrics to show popularity

## Step 5: Create Pull Request

1. **Create branch**: `trends/YYYYMMDD-daily`
2. **Commit the file**:
   ```
   git add trends/YYYYMMDD-daily.md
   git commit -m "Add daily trends for YYYY-MM-DD"
   ```
3. **Push to remote**:
   ```
   git push -u origin trends/YYYYMMDD-daily
   ```
4. **Create PR using gh CLI**:
   ```
   gh pr create --title "Daily Trends: YYYY-MM-DD" --body "$(cat <<'EOF'
   # Objective
   Add automated daily trends report for YYYY-MM-DD

   # Effect
   - Collected [count] articles across AI (60%), Tech (30%), Others (10%)
   - Top finding: [brief mention of most interesting item]

   # Test
   - [x] All sources fetched successfully
   - [x] Rate limits respected
   - [x] Output file created in trends/

   # Note
   Automated collection via daily-trends workflow. Review highlights and merge if satisfied.

   ---

   ## Definition of Done Checklist

   ### Common
   - [x] Describe the concrete sentences to support understanding (not just writing "I understand ...")
   - [x] Describe the condition which can be applied (who, when, where)
   - [x] Include information about licenses and copyrights

   ### Computer Science / Machine Learning (if applicable)
   - [ ] Clear Input and Output
   - [ ] Describe Algorithms with pseudocode
   - [ ] Explain datasets used
   - [ ] Clear calculation order
   - [ ] Describe the difference between similar algorithms
   EOF
   )"
   ```

## Error Handling

- If a source fails (timeout, rate limit, 403/429 errors): Skip it and note in the PR body
- If no articles meet criteria: Create report anyway with note about low activity
- If GitHub API fails: Commit locally and create PR manually in next run

## Final Checklist

Before completing, verify:
- [ ] Compliance notes reviewed
- [ ] All rate limits respected with proper delays
- [ ] User-Agent headers used correctly
- [ ] Output file created in trends/ directory
- [ ] File follows markdown format exactly
- [ ] All URLs are valid and direct (no intermediary pages)
- [ ] Weighting approximately follows 60/30/10 distribution
- [ ] Git commit created with descriptive message
- [ ] PR created with proper template format
- [ ] Any issues or skipped sources documented

## Success Criteria

A successful run produces:
1. One markdown file in `trends/YYYYMMDD-daily.md`
2. Properly formatted with all sections
3. At least 10-15 articles total (if available)
4. Proper weighting (60% AI, 30% Tech, 10% Others)
5. Pull request created with informative description
6. No ToS violations or rate limit errors
