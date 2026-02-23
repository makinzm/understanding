# Daily AI & Tech Trends Collection

You are tasked with collecting daily trends about AI Agents, software tools, and news from Japan and worldwide. 
Follow these instructions carefully.

# Your owner

Your owner is a software engineer who specializes in Machine Learning Algorithms.

Concerned about the rapid pace of AI development, they want to stay informed about the latest trends in AI agents, frameworks, and tools, as well as general tech news and world events.

And then he wants to share this information with his team in a concise, curated format and understand the value of the information.

## 🎯 PRIMARY GOAL: CREATE A PULL REQUEST

**YOUR TASK IS NOT COMPLETE UNTIL A PULL REQUEST IS CREATED.**

Even if data collection encounters issues:
1. Create a report documenting what happened
2. Commit the file to a new branch
3. **ALWAYS create a PR** - this is MANDATORY

Do not stop until you have executed `gh pr create` successfully.

## ⚠️ CRITICAL: NO HALLUCINATION POLICY

**DO NOT GENERATE, INVENT, OR HALLUCINATE ANY CONTENT.**

- You MUST run `python3 scripts/fetch_trends.py` to fetch real data
- You MUST use ONLY articles from the JSON output
- You MUST NOT create fake URLs, article titles, or engagement metrics
- You MUST NOT add articles about non-existent products (GPT-5, Claude 4, etc.) unless they actually appear in the fetched data
- You MUST validate all output against the source JSON

**If the fetch script fails and returns no data, DO NOT proceed with fake content. Document the failure instead.**

## Overview

Create a curated daily trends report with two sections:
- **Pick Up**: A short hand-curated selection of the most interesting/relevant items, with a personal reason for recommending each. Prioritize ML, AI, and security research given the owner's background.
- **All**: The full list of fetched articles, organized by source.

Sources covered: Hacker News, Reddit, Hatena Bookmark, Wiz Blog, arXiv cs.LG.

**All content must come from real data fetched by the script.**

## Step 1: Compliance Checks

Before fetching any data, review `.github/prompts/compliance-notes.md` to understand ToS requirements for each source.

**Required checks:**
1. Verify robots.txt compliance (documented in compliance-notes.md)
2. Implement rate limiting:
   - Hacker News API: 1 second between requests
   - Reddit JSON API: 1 second between requests (max 60/minute)
   - Hatena Bookmark: 5 seconds between requests (per robots.txt Crawl-delay)
3. Use proper User-Agent headers as specified in compliance-notes.md

**If any source is blocked or rate-limited**: Skip it gracefully and note in the output.

## Step 2: Fetch Real Data (REQUIRED - NO HALLUCINATION)

**⚠️ CRITICAL: You MUST run the fetch script to get real data. DO NOT generate, invent, or hallucinate any articles, URLs, or content.**

### Run the Fetch Script

Execute the Python fetch script to collect real data from all sources:

```bash
python3 scripts/fetch_trends.py > /tmp/trends-data.json
```

This script will:
- Fetch Hacker News top stories (50+ points)
- Fetch Reddit posts from AI/Tech/General subreddits (100+ upvotes or 50+ comments)
- Fetch Hatena Bookmark hot entries from RSS feeds (10+ bookmarks)
- Fetch Wiz Blog posts via RSS feed (no rate limit required)
- Fetch arXiv cs.LG recent submissions (first 50 entries, 15 s crawl-delay)
- Respect all rate limits and ToS requirements
- Output structured JSON with all fetched articles

### Validate the Fetched Data

After running the script, verify the output:

1. **Check the JSON structure**:
   ```bash
   jq '.summary' /tmp/trends-data.json
   ```
   Expected output: `{"hn_count": N, "reddit_count": M, "hatena_count": P, "wiz_count": Q, "arxiv_count": R, "total": X}`

2. **Verify article counts**:
   - If total is 0: The fetch failed, check error messages
   - If total is < 10: Data collection may be incomplete, note in report
   - If total is > 10: Proceed with analysis

3. **Sample check URLs** (verify at least 3 URLs are real):
   ```bash
   jq -r '.hacker_news[0].url, .reddit[0].url, .hatena[0].url, .wiz_blog[0].url, .arxiv_cslg[0].url' /tmp/trends-data.json | head -5
   ```

**If the fetch script fails**: Document the failure in the report and DO NOT generate fake data. Create a report noting the technical issue.

## Step 3: Parse and Analyze Real Data

**⚠️ CRITICAL: Use ONLY the data from /tmp/trends-data.json. DO NOT add, modify, or invent any articles.**

### Load and Parse the JSON

```bash
jq '.' /tmp/trends-data.json
```

The JSON structure:
```json
{
  "hacker_news":  [{"title": "...", "url": "...", "score": N, "descendants": M, "source": "HackerNews"}],
  "reddit":       [{"title": "...", "url": "...", "ups": N, "num_comments": M, "category": "AI/Tech/General", "subreddit": "..."}],
  "hatena":       [{"title": "...", "url": "...", "bookmarks": N, "category": "Tech/General"}],
  "wiz_blog":     [{"title": "...", "url": "...", "published": "...", "description": "...", "source": "WizBlog"}],
  "arxiv_cslg":   [{"arxiv_id": "...", "title": "...", "url": "...", "subjects": "...", "source": "arXiv_csLG"}]
}
```

### Select Pick Up Articles

Scan all articles and select **5–10 items** for the Pick Up section.

**Selection criteria** (owner is a software engineer specializing in ML):
- Prioritize: ML research papers, LLM/agent advances, cloud security research (Wiz), new frameworks and tools
- Also consider: high-engagement HN/Reddit posts on AI, notable Hatena Tech entries, arXiv papers with novel methods
- Write a concrete `Why I recommend:` sentence for each pick — what makes it interesting or actionable

## Step 4: Generate Output

**⚠️ Directory Structure**: Use year/month subdirectories for better organization.

Create file: `trends/YYYY/MM/YYYYMMDD-daily.md` (use today's date)

Example: For 2026-02-08, create `trends/2026/02/20260208-daily.md`

**Before creating the file**:
1. Create directories if they don't exist: `mkdir -p trends/YYYY/MM`
2. Verify the JSON data is loaded: `test -f /tmp/trends-data.json`

**Format**:

````markdown
# Daily AI & Tech Trends - YYYY-MM-DD

## Pick Up

1. [Article Title](url)
   - Why I recommend: [1–2 sentences — what makes this interesting or actionable for an ML engineer]
2. [Article Title](url)
   - Why I recommend:
<!-- 5–10 items total -->

## All

### Hacker News

| Title | Score | Comments |
|-------|------:|---------:|
| [Article Title](url) | 123 | 45 |

### Reddit

| Title | Subreddit | Upvotes | Comments |
|-------|-----------|--------:|---------:|
| [Article Title](permalink) | r/MachineLearning | 456 | 78 |

### Hatena

| Title | Bookmarks |
|-------|---------:|
| [Article Title](url) | 34 |

### Wiz Blog

| Title | Published |
|-------|-----------|
| [Article Title](url) | YYYY-MM-DD |

### arXiv cs.LG

| Title | arXiv ID |
|-------|----------|
| [Article Title](url) | 2602.XXXXX |

---

## Fetch Status

| Source | Status | Count |
|--------|--------|------:|
| Hacker News | ✅ | N |
| Reddit | ✅ | N |
| Hatena | ✅ | N |
| Wiz Blog | ✅ | N |
| arXiv cs.LG | ✅ | N |
| **Total** | | **N** |

<!-- Any issues encountered, sources skipped, or relevant observations -->
````

**Content guidelines**:
- **Use ONLY URLs from /tmp/trends-data.json** — no fake or generated links
- For Reddit discussions, use the `permalink` field from JSON
- For Hatena, keep original Japanese titles (add English translation in parentheses if helpful)
- Engagement metrics must match the JSON exactly (do not round or estimate)
- arXiv: use `arxiv_id` field for the ID column, `url` for the link
- Wiz Blog: use `published` field for the date column; strip time if present

**Anti-Hallucination Checklist**:
- [ ] Every article title comes from the JSON data (not invented)
- [ ] Every URL is from the JSON data (verify no example.com, placeholder.com, or fake domains)
- [ ] Article counts in Fetch Status match `jq '.summary' /tmp/trends-data.json`
- [ ] No articles about "GPT-5", "Claude 4", or other non-existent products (unless actually in the data)
- [ ] Engagement metrics match the JSON exactly (not estimated or inflated)

## Step 5: Validate and Create Pull Request

### Pre-Commit Validation

Before committing, verify the report is valid (BUT CONTINUE TO PR EVEN IF VALIDATION FAILS):

```bash
VALIDATION_PASSED=1

# 1. Check file exists in correct location
if ! test -f trends/YYYY/MM/YYYYMMDD-daily.md; then
  echo "⚠️  WARNING: File not in correct location"
  VALIDATION_PASSED=0
fi

# 2. Verify no placeholder/fake URLs
if grep -E "(example\.com|placeholder\.com|fake\.com|test\.com|openai\.com/blog/gpt-5)" trends/YYYY/MM/YYYYMMDD-daily.md; then
  echo "⚠️  WARNING: Found fake/placeholder URLs"
  VALIDATION_PASSED=0
fi

# 3. Check article counts match
REPORT_COUNT=$(grep -c "^- \*\*\[" trends/YYYY/MM/YYYYMMDD-daily.md 2>/dev/null || echo "0")
JSON_COUNT=$(jq '.summary.total' /tmp/trends-data.json 2>/dev/null || echo "0")
echo "Report has $REPORT_COUNT articles, JSON has $JSON_COUNT articles"

# 4. Verify real domains are present
if ! grep -E "(news\.ycombinator\.com|reddit\.com|hatena\.ne\.jp|github\.com|arxiv\.org|wiz\.io)" trends/YYYY/MM/YYYYMMDD-daily.md > /dev/null 2>&1; then
  echo "⚠️  WARNING: No real domains found"
  VALIDATION_PASSED=0
fi

if [ $VALIDATION_PASSED -eq 1 ]; then
  echo "✅ Validation passed"
else
  echo "⚠️  Validation had warnings - documenting in PR body and continuing"
fi

# IMPORTANT: Even if validation fails, continue to create PR
```

### Create Branch and Commit

1. **Create branch**: `trends/YYYYMMDD-daily`
   ```bash
   git checkout -b trends/YYYYMMDD-daily
   ```

2. **Add and commit**:
   ```bash
   git add trends/YYYY/MM/YYYYMMDD-daily.md
   git commit -m "Add daily trends for YYYY-MM-DD"
   ```

3. **Push to remote**:
   ```bash
   git push -u origin trends/YYYYMMDD-daily
   ```

4. **Create PR using gh CLI** (THIS IS MANDATORY - RETRY IF NEEDED):

   First, prepare the PR body:
   ```bash
   cat > /tmp/pr-body.txt <<'EOF'
   # Objective
   Add automated daily trends report for YYYY-MM-DD

   # Effect
   - Collected [actual_count] real articles
   - Top finding: [brief mention of most interesting item from real data]
   - Sources: Hacker News ([hn_count]), Reddit ([reddit_count]), Hatena ([hatena_count]), Wiz Blog ([wiz_count]), arXiv cs.LG ([arxiv_count])

   # Test
   - [x] Data fetched from real sources (not hallucinated)
   - [x] All URLs validated (no fake domains)
   - [x] Article counts match JSON output
   - [x] Rate limits respected
   - [x] Output file created in trends/YYYY/MM/

   # Note
   Automated collection via daily-trends workflow. All data verified against source APIs.

   ## Data Validation
   - Script output: [paste jq '.summary' /tmp/trends-data.json output]
   - No hallucinated content (verified with grep checks)
   - File location: trends/YYYY/MM/YYYYMMDD-daily.md

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
   ```

   Then create the PR with retry logic:
   ```bash
   # Retry up to 3 times
   SUCCESS=0
   for i in 1 2 3; do
     echo "Attempt $i to create PR..."
     if gh pr create --title "Daily Trends: YYYY-MM-DD" --body-file /tmp/pr-body.txt; then
       echo "✅ PR created successfully!"
       SUCCESS=1
       break
     else
       echo "⚠️  PR creation failed, attempt $i/3"
       sleep 5
     fi
   done

   if [ $SUCCESS -eq 0 ]; then
     echo "❌ Failed to create PR after 3 attempts"
     echo "Please check the logs and create PR manually:"
     echo "  gh pr create --title 'Daily Trends: YYYY-MM-DD' --body-file /tmp/pr-body.txt"
     exit 1
   fi
   ```

**Replace placeholders in PR body** (in /tmp/pr-body.txt before creating):
- `[actual_count]`: Total from `jq '.summary.total' /tmp/trends-data.json`
- `[hn_count]`, `[reddit_count]`, `[hatena_count]`, `[wiz_count]`, `[arxiv_count]`: From `jq '.summary' /tmp/trends-data.json`

## Error Handling

- **If fetch script fails completely**: DO NOT generate fake data. Create a minimal report documenting the failure **AND STILL CREATE A PR**.
- **If a source fails** (timeout, rate limit, 403/429 errors): The script will skip it. Note the missing source in the PR body **AND CONTINUE TO PR CREATION**.
- **If no articles meet criteria**: Create report with available data and note about low activity. DO NOT pad with fake articles. **BUT STILL CREATE THE PR**.
- **If GitHub API fails**: Retry `gh pr create` up to 3 times with 5-second delays. If still failing, document the error and attempt using `gh api` directly.

**CRITICAL**: Never stop before creating a PR. Even if everything fails, create a PR with a failure report. The PR is the evidence that the workflow ran.

**Never, under any circumstances, generate fake articles to fill gaps in data collection.**

## Final Checklist

**🎯 MANDATORY (DO NOT SKIP):**
- [ ] **PR created**: With actual counts and validation output in description - **THIS IS THE PRIMARY GOAL**
- [ ] **Git commit created**: With descriptive message including date

**Quality checks (verify if possible, but don't block PR creation):**
- [ ] **Fetch script executed**: `python3 scripts/fetch_trends.py` was run successfully
- [ ] **Real data loaded**: `/tmp/trends-data.json` exists and contains articles
- [ ] **No hallucination**: All articles come from JSON data (verified with grep checks)
- [ ] **URLs validated**: No fake domains (example.com, placeholder.com, non-existent products)
- [ ] **Counts match**: Report article count matches JSON summary total
- [ ] **Directory structure**: File created in `trends/YYYY/MM/YYYYMMDD-daily.md` (not flat trends/)
- [ ] **Compliance respected**: Rate limits followed (script handles this automatically)
- [ ] **Format correct**: Markdown follows template exactly
- [ ] **Metrics accurate**: Engagement numbers match JSON exactly (not rounded/estimated)
- [ ] **Pick Up quality**: 5–10 items selected with concrete "Why I recommend" text
- [ ] **Issues documented**: Any skipped sources or problems noted in PR body

**Remember: Even if some quality checks fail, ALWAYS create the PR to document what happened.**

## Success Criteria

**MINIMUM SUCCESS (MANDATORY):**
1. **Pull request created**: With actual counts, validation output, and data summary - **THIS IS THE ONLY TRULY REQUIRED OUTPUT**
2. **File created**: `trends/YYYY/MM/YYYYMMDD-daily.md` with year/month subdirectories (even if minimal/error report)

**IDEAL SUCCESS (aim for these but don't block PR if issues occur):**
3. **Real data fetched**: `python3 scripts/fetch_trends.py` executed successfully
4. **No hallucination**: All articles, URLs, and metrics from real JSON data
5. **Validation passed**: grep checks confirm no fake domains or placeholder content
6. **Proper format**: Markdown follows template with all sections
7. **Good coverage**: At least 10-15 articles total (if available from sources)
8. **Pick Up curated**: 5–10 items with concrete "Why I recommend" text, reflecting ML/AI focus
9. **Compliance maintained**: No ToS violations, all rate limits respected (handled by script)

**What constitutes a "failed" run:**
- **ONLY if PR is not created** - this is the only true failure
- Everything else is a degraded success (create PR documenting the issues)

**If data collection fails:**
- DO NOT proceed with fake data
- DO create a minimal report documenting the failure
- DO create a PR explaining what went wrong
- The PR itself is proof the workflow ran and can be debugged
