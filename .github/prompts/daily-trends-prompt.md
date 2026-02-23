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

Create a curated daily trends report with weighted focus:
- **60% AI**: AI agents, frameworks, LLMs, autonomous systems, AI tooling
- **30% Tech**: Software development, developer tools, programming, engineering
- **10% Others**: World news, Japan news, general interest

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
- Respect all rate limits and ToS requirements
- Output structured JSON with all fetched articles

### Validate the Fetched Data

After running the script, verify the output:

1. **Check the JSON structure**:
   ```bash
   jq '.summary' /tmp/trends-data.json
   ```
   Expected output: `{"hn_count": N, "reddit_count": M, "hatena_count": P, "total": X}`

2. **Verify article counts**:
   - If total is 0: The fetch failed, check error messages
   - If total is < 10: Data collection may be incomplete, note in report
   - If total is > 10: Proceed with analysis

3. **Sample check URLs** (verify at least 3 URLs are real):
   ```bash
   jq -r '.hacker_news[0].url, .reddit[0].url, .hatena[0].url' /tmp/trends-data.json | head -3
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
  "hacker_news": [{"title": "...", "url": "...", "score": N, "descendants": M, "source": "HackerNews"}],
  "reddit": [{"title": "...", "url": "...", "ups": N, "num_comments": M, "category": "AI/Tech/General", "subreddit": "..."}],
  "hatena": [{"title": "...", "url": "...", "bookmarks": N, "category": "Tech/General"}]
}
```

### Categorize and Rate Each Article

For each article in the JSON, assign a category and rating:

**Categories**:
- **🤖 AI (60% target)**: AI agents, LLMs, Claude/GPT, autonomous systems, AI frameworks, ML research
- **💻 Tech (30% target)**: Dev tools, programming, frameworks, DevOps, architecture, performance
- **🌍 World/Japan (10% target)**: World news, Japan news, tech policy, general interest

**Ratings**:
- **★★★ (High Priority)**: Direct match to category focus, high engagement, significant impact
- **★★ (Medium Priority)**: Related but indirect, moderate engagement
- **★ (Low Priority)**: Tangentially related, lower engagement

**Rating criteria**:
- Hacker News: 200+ points = ★★★, 100-199 = ★★, 50-99 = ★
- Reddit: 500+ upvotes = ★★★, 200-499 = ★★, 100-199 = ★
- Hatena: 50+ bookmarks = ★★★, 20-49 = ★★, 10-19 = ★

**Distribution goal**: Aim for ~60% AI, ~30% Tech, ~10% Others in final selection.

## Step 4: Generate Output

**⚠️ Directory Structure**: Use year/month subdirectories for better organization.

Create file: `trends/YYYY/MM/YYYYMMDD-daily.md` (use today's date)

Example: For 2026-02-08, create `trends/2026/02/20260208-daily.md`

**Before creating the file**:
1. Create directories if they don't exist: `mkdir -p trends/YYYY/MM`
2. Verify the JSON data is loaded: `test -f /tmp/trends-data.json`

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
- **Use ONLY URLs from /tmp/trends-data.json** - no fake or generated links
- All URLs must be direct links to original articles (not intermediary pages)
- Summaries should be concise and informative (1-2 sentences per article)
- For Reddit discussions, use the `permalink` field from JSON
- For Hatena, keep original Japanese titles with English translation in parentheses if needed
- Include engagement metrics exactly as they appear in the JSON (don't round or estimate)

**Anti-Hallucination Checklist**:
- [ ] Every article title comes from the JSON data (not invented)
- [ ] Every URL is from the JSON data (verify no example.com, placeholder.com, or fake domains)
- [ ] Article counts match JSON summary (HN count, Reddit count, Hatena count)
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
if ! grep -E "(news\.ycombinator\.com|reddit\.com|hatena\.ne\.jp|github\.com|arxiv\.org)" trends/YYYY/MM/YYYYMMDD-daily.md > /dev/null 2>&1; then
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
   - Collected [actual_count] real articles across AI ([ai_count]), Tech ([tech_count]), Others ([other_count])
   - Top finding: [brief mention of most interesting item from real data]
   - Sources: Hacker News ([hn_count]), Reddit ([reddit_count]), Hatena ([hatena_count])

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
- `[ai_count]`, `[tech_count]`, `[other_count]`: Actual distribution from your analysis
- `[hn_count]`, `[reddit_count]`, `[hatena_count]`: From `jq '.summary' /tmp/trends-data.json`

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
- [ ] **Weighting approximate**: ~60% AI, ~30% Tech, ~10% Others (based on available data)
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
8. **Approximate weighting**: ~60% AI, ~30% Tech, ~10% Others (based on real data)
9. **Compliance maintained**: No ToS violations, all rate limits respected (handled by script)

**What constitutes a "failed" run:**
- **ONLY if PR is not created** - this is the only true failure
- Everything else is a degraded success (create PR documenting the issues)

**If data collection fails:**
- DO NOT proceed with fake data
- DO create a minimal report documenting the failure
- DO create a PR explaining what went wrong
- The PR itself is proof the workflow ran and can be debugged
