# ToS Compliance Notes for Automated Workflows

This document archives the terms of service compliance requirements for all sources used in the daily trends collection and auto-summarize-papers workflows.

## Hacker News

**API**: Firebase API (`https://hacker-news.firebaseio.com/v0/`)
- **Status**: ✅ Official API, fully allowed
- **Official Documentation**: https://github.com/HackerNews/API
- **Terms of Service**: https://www.ycombinator.com/legal/ (Y Combinator Legal)
- **User-Agent**: `Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)`
- **robots.txt**: Not applicable (using official API)
- **Last Checked**: 2026-02-08

## Reddit

**API**: Public JSON API (`.json` endpoint)
- **Status**: ✅ Public read-only access allowed
- **Official Documentation**: https://www.reddit.com/dev/api
- **Terms of Service**: https://www.redditinc.com/policies/user-agreement
- **API Rules**: https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki
- **Developer Platform Agreement**: https://www.redditinc.com/policies/data-api-terms
- **User-Agent**: `web:understanding-trends:v1.0.0 (by /u/YOUR_REDDIT_USERNAME)`
  - **ACTION REQUIRED**: Replace `YOUR_REDDIT_USERNAME` with actual Reddit username
- **robots.txt**: https://www.reddit.com/robots.txt
- **Last Checked**: 2026-02-08
- **Notes**:
  - Using public `.json` endpoints (no OAuth required)
  - Only accessing public subreddits
  - No voting, posting, or authenticated actions

## Hatena Bookmark

**API**: Public bookmark pages (Note: Hatena provides RSS feeds which may be preferred)
- **Status**: ⚠️ Web scraping (no official API for hotentry, but RSS available)
- **Official Site**: https://b.hatena.ne.jp/
- **Terms of Service**: https://www.hatena.ne.jp/rule (Japanese)
- **Privacy Policy**: https://policies.hatena.ne.jp/privacy (Japanese)
- **URL Pattern**: `https://b.hatena.ne.jp/hotentry/[category]`
- **Alternative (RSS)**: `https://b.hatena.ne.jp/hotentry/[category].rss`
- **Rate Limits**: `Crawl-delay: 5` specified in robots.txt — implement 5-second delays minimum
- **User-Agent**: `Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)`
- **robots.txt**: https://b.hatena.ne.jp/robots.txt
- **Last Checked**: 2026-02-08
- **Notes**:
  - Prefer RSS feeds when possible (more stable, intended for automation)
  - Only accessing public bookmark pages (no login required)
  - Using WebFetch tool which respects standard web etiquette
  - If blocked or rate-limited, skip this source

## Wiz Blog (fetch_trends workflow)

**Access**: RSS feed (`https://www.wiz.io/feed/rss.xml`)
- **Status**: ✅ Fully allowed; robots.txt permits all crawlers unconditionally
- **Official Site**: https://www.wiz.io/blog
- **Terms of Service**: https://legal.wiz.io/legal#terms-of-use
- **Privacy Policy**: https://legal.wiz.io/legal#privacy-policy
- **User-Agent**: `Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)`
- **robots.txt**: https://www.wiz.io/robots.txt
  - `User-agent: *` → `Allow: /` (no path restrictions, no crawl-delay)
- **Last Checked**: 2026-02-23
- **Notes**:
  - Uses RSS feed (`/feed/rss.xml`) for reliable structured parsing — avoids JS-rendered HTML
  - No rate limiting required, but single request per run is sufficient
  - Only fetching public blog content

## arXiv (auto-summarize-papers workflow)

**Access**: Paper content fetched via ar5iv (HTML mirror), not arxiv.org directly
- **Status**: ✅ ar5iv is very permissive; arxiv.org direct access not needed
- **Official Site**: https://arxiv.org/
- **Terms of Service**: https://arxiv.org/help/policies/terms_of_use
- **arxiv.org robots.txt**: https://arxiv.org/robots.txt
  - `Crawl-delay: 15` for `User-agent: *`
  - Allows: `/abs`, `/pdf`, `/html`, `/archive`, `/list`, `/year`, `/catchup`
  - **Do NOT scrape arxiv.org directly** — use ar5iv instead
- **ar5iv robots.txt**: https://ar5iv.labs.arxiv.org/robots.txt
  - Only disallows `/log/` — all paper HTML paths are allowed
- **Last Checked**: 2026-02-11
- **Notes**:
  - The `summarize-arxiv-paper` skill fetches `https://ar5iv.labs.arxiv.org/html/<ID>`, which is fully allowed
  - Never fetch `https://arxiv.org/pdf/` or bulk-scrape arxiv.org

## arXiv cs.LG Listing (fetch_trends workflow)

**Access**: HTML listing page scraped directly from `https://arxiv.org/list/cs.LG/recent`
- **Status**: ⚠️ Allowed with strict rate limiting — `/list` is explicitly permitted but `Crawl-delay: 15` applies
- **Official Site**: https://arxiv.org/
- **Terms of Service**: https://arxiv.org/help/policies/terms_of_use
- **User-Agent**: `Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)`
- **robots.txt**: https://arxiv.org/robots.txt
  - `User-agent: *` → `Crawl-delay: 15`
  - `Allow: /list` (listing pages explicitly permitted)
  - `Disallow: /search`, `/api`, `/pdf` (do not touch these)
- **Last Checked**: 2026-02-23
- **Notes**:
  - `fetch_arxiv_csLG()` sleeps 15 seconds before every request (`time.sleep(15)`)
  - Only fetches a **single page** (first 50 entries) per run — no pagination
  - Parses `<dt>/<dd>` HTML to extract arXiv ID, title, and primary subject
  - For full paper content, use ar5iv via the `summarize-arxiv-paper` skill (not this workflow)

## General Compliance Principles

1. **Respect robots.txt**: Always check before first fetch
2. **Rate Limiting**: Implement appropriate delays between requests
3. **User-Agent**: Always use proper identification
4. **Public Only**: Never access authenticated or private content
5. **Graceful Degradation**: If a source fails or blocks, skip it and continue
6. **Monitoring**: Log any 429 (rate limit) or 403 (forbidden) responses
7. **Review**: Re-check ToS quarterly (every 3 months)

## Next Review Date

**2026-05-08** (3 months from creation)

## Revision History

- 2026-02-08: Initial documentation
- 2026-02-11: Fix Hatena crawl-delay from 2s to 5s (per robots.txt); add arXiv/ar5iv compliance section
- 2026-02-23: Add Wiz Blog (RSS, no restrictions) and arXiv cs.LG listing (Crawl-delay: 15, single-page) sections for fetch_trends workflow
