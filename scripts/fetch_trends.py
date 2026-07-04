#!/usr/bin/env python3
"""
Daily Trends Data Fetcher
Fetches data from Hacker News, Reddit, and Hatena Bookmark with proper rate limiting
"""

import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from xml.etree import ElementTree as ET


def log(message):
    """Print progress messages to stderr so stdout stays valid JSON when redirected."""
    print(message, file=sys.stderr, flush=True)

# User-Agent strings
HN_UA = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"
REDDIT_UA = "web:understanding-trends:v1.0.0 (by /u/trends_bot)"
HATENA_UA = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"
# Wiz blog: robots.txt allows all (Allow: /), RSS feed available at /feed/rss.xml
WIZ_UA = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"
# arXiv: robots.txt Crawl-delay: 15 (User-agent: *), /list path is allowed
ARXIV_UA = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"


def fetch_json(url, user_agent, timeout=10):
    """Fetch JSON from URL with proper User-Agent"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        log(f"Error fetching {url}: {e}")
        return None


def fetch_text(url, user_agent, timeout=10):
    """Fetch text/XML from URL with proper User-Agent"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode()
    except Exception as e:
        log(f"Error fetching {url}: {e}")
        return None


def fetch_hacker_news():
    """Fetch Hacker News top stories with 50+ points"""
    log("[1/5] Fetching Hacker News...")
    stories = []

    # Get top story IDs
    top_ids = fetch_json(
        "https://hacker-news.firebaseio.com/v0/topstories.json", HN_UA
    )
    if not top_ids:
        return stories

    # Fetch details for first 30 stories (with rate limiting)
    for i, story_id in enumerate(top_ids[:30]):
        if i > 0:
            time.sleep(1)  # 1-second delay between requests

        story = fetch_json(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", HN_UA
        )
        if story and story.get("score", 0) >= 50:
            stories.append(
                {
                    "title": story.get("title", ""),
                    "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                    "score": story.get("score", 0),
                    "descendants": story.get("descendants", 0),
                    "time": story.get("time", 0),
                    "source": "HackerNews",
                }
            )

    log(f"  → Fetched {len(stories)} stories")
    return stories


def fetch_reddit():
    """Fetch Reddit posts from AI, Tech, and General subreddits"""
    log("[2/5] Fetching Reddit...")
    posts = []

    # AI-focused (60% weight)
    ai_subs = ["MachineLearning", "LocalLLaMA", "OpenAI", "ClaudeAI", "AutoGPT"]
    # Tech-focused (30% weight)
    tech_subs = ["programming", "SideProject", "opensource", "webdev"]
    # General (10% weight)
    general_subs = ["worldnews", "japan"]

    all_subs = [
        (*ai_subs, "AI"),
        (*tech_subs, "Tech"),
        (*general_subs, "General"),
    ]

    # Flatten and categorize
    for sub_list, category in [
        (ai_subs, "AI"),
        (tech_subs, "Tech"),
        (general_subs, "General"),
    ]:
        for sub in sub_list:
            time.sleep(1)  # 1-second delay between requests
            log(f"  → Fetching r/{sub}...")

            data = fetch_json(
                f"https://old.reddit.com/r/{sub}/hot.json?limit=25", REDDIT_UA
            )
            if not data or "data" not in data:
                continue

            for child in data["data"].get("children", []):
                post = child.get("data", {})
                ups = post.get("ups", 0)
                comments = post.get("num_comments", 0)

                # Filter: 100+ upvotes OR 50+ comments
                if ups >= 100 or comments >= 50:
                    posts.append(
                        {
                            "title": post.get("title", ""),
                            "url": post.get("url", ""),
                            "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
                            "ups": ups,
                            "num_comments": comments,
                            "created_utc": post.get("created_utc", 0),
                            "subreddit": post.get("subreddit", ""),
                            "category": category,
                            "source": "Reddit",
                        }
                    )

    log(f"  → Fetched {len(posts)} posts")
    return posts


def fetch_hatena():
    """Fetch Hatena Bookmark hot entries from RSS feeds"""
    log("[3/5] Fetching Hatena Bookmark...")
    bookmarks = []

    categories = [
        ("https://b.hatena.ne.jp/hotentry/it.rss", "Tech"),
    ]

    for url, category in categories:
        time.sleep(5)  # 5-second delay per Hatena robots.txt Crawl-delay directive
        log(f"  → Fetching {url}...")

        xml_content = fetch_text(url, HATENA_UA)
        if not xml_content:
            continue

        try:
            root = ET.fromstring(xml_content)
            # This feed is RSS 1.0 (RDF): <item>, <title>, <link> live in the
            # default namespace declared on <rdf:RDF>, and the bookmark count
            # is a dedicated <hatena:bookmarkcount> element, not free text.
            RSS_NS = "http://purl.org/rss/1.0/"
            HATENA_NS = "http://www.hatena.ne.jp/info/xmlns#"

            for item in root.findall(f".//{{{RSS_NS}}}item"):
                title_elem = item.find(f"{{{RSS_NS}}}title")
                link_elem = item.find(f"{{{RSS_NS}}}link")
                count_elem = item.find(f"{{{HATENA_NS}}}bookmarkcount")

                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ""
                    link = link_elem.text or ""
                    bookmarks_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

                    # Filter: 10+ bookmarks
                    if bookmarks_count >= 10:
                        bookmarks.append(
                            {
                                "title": title,
                                "url": link,
                                "bookmarks": bookmarks_count,
                                "category": category,
                                "source": "Hatena",
                            }
                        )
        except Exception as e:
            log(f"  Error parsing XML: {e}")
            continue

    log(f"  → Fetched {len(bookmarks)} bookmarks")
    return bookmarks


def fetch_wiz_blog():
    """Fetch Wiz Security Blog via RSS feed.
    robots.txt: Allow: / for all agents (no restrictions, no Crawl-delay).
    Uses RSS feed at https://www.wiz.io/feed/rss.xml for reliable parsing.
    """
    log("[4/5] Fetching Wiz Blog...")
    articles = []

    xml_content = fetch_text("https://www.wiz.io/feed/rss.xml", WIZ_UA)
    if not xml_content:
        return articles

    try:
        root = ET.fromstring(xml_content)
        # Handle both RSS 2.0 (<rss><channel><item>) and Atom (<feed><entry>)
        channel = root.find("channel")
        if channel is not None:
            items = channel.findall("item")
        else:
            # Atom feed: entries are direct children or under namespace
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"
            items = root.findall(f"{ns}entry")

        for item in items:
            # RSS 2.0 fields
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")
            desc_elem = item.find("description")

            # Atom fallback fields
            if title_elem is None:
                ns = item.tag.split("}")[0] + "}" if item.tag.startswith("{") else ""
                title_elem = item.find(f"{ns}title")
                link_elem = item.find(f"{ns}link")
                pub_date_elem = item.find(f"{ns}published") or item.find(f"{ns}updated")
                desc_elem = item.find(f"{ns}summary") or item.find(f"{ns}content")

            if title_elem is None:
                continue

            # ElementTree already unwraps CDATA sections into .text, so no
            # further stripping of "<![CDATA[" markers is needed here — doing
            # it with lstrip/rstrip (which strip character *sets*, not a
            # literal prefix/suffix) was eating real leading letters like the
            # "A" in "AI" or the "T" in "The".
            title = (title_elem.text or "").strip()
            # <link> in Atom may use href attribute instead of text
            if link_elem is not None:
                url = (link_elem.text or link_elem.get("href") or "").strip()
            else:
                url = ""
            published = (pub_date_elem.text or "").strip() if pub_date_elem is not None else ""
            description = (desc_elem.text or "").strip() if desc_elem is not None else ""

            if title and url:
                articles.append({
                    "title": title,
                    "url": url,
                    "published": published,
                    "description": description,
                    "source": "WizBlog",
                })
    except Exception as e:
        log(f"  Error parsing Wiz RSS: {e}")

    # Feed is newest-first and contains the blog's entire history (600+ entries);
    # keep only the most recent ones for a "daily trends" snapshot.
    articles = articles[:20]

    log(f"  → Fetched {len(articles)} articles")
    return articles


def fetch_arxiv_csLG():
    """Fetch recent arXiv cs.LG (Machine Learning) paper listings.
    robots.txt: Crawl-delay: 15 (User-agent: *), /list path is explicitly allowed.
    Fetches a single page to respect crawl-delay; sleeps 15 s before request.
    """
    import html
    import re

    log("[5/5] Fetching arXiv cs.LG/recent...")
    papers = []

    time.sleep(15)  # Respect Crawl-delay: 15 per arxiv.org robots.txt
    page_html = fetch_text("https://arxiv.org/list/cs.LG/recent", ARXIV_UA)
    if not page_html:
        return papers

    # arXiv listing pages use <dt>/<dd> pairs.
    # <dt> contains the sequence number and arXiv ID link.
    # <dd> contains title (class="list-title"), authors, and subjects.
    # arxiv.org markup mixes single/double quotes and sometimes spaces around `=`
    # (e.g. `href ="/abs/..."` or `class='list-title mathjax'`), so quote style
    # and surrounding whitespace can't be assumed fixed.
    dt_dd_re = re.compile(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", re.DOTALL)
    id_re = re.compile(r'href\s*=\s*[\'"]/abs/(\d{4}\.\d{4,5})[\'"]')
    # Title: <div class='list-title mathjax'><span class='descriptor'>Title:</span>\nPAPER TITLE</div>
    title_re = re.compile(
        r'class\s*=\s*[\'"]list-title[^\'"]*[\'"][^>]*>.*?class\s*=\s*[\'"]descriptor[\'"][^>]*>.*?</span>\s*(.*?)\s*</div>',
        re.DOTALL | re.IGNORECASE,
    )
    # Subjects: <span class="primary-subject">Machine Learning (cs.LG)</span>
    subject_re = re.compile(
        r'class\s*=\s*[\'"]primary-subject[\'"][^>]*>(.*?)</span>', re.DOTALL | re.IGNORECASE
    )
    tag_re = re.compile(r"<[^>]+>")

    for m in dt_dd_re.finditer(page_html):
        dt_content = m.group(1)
        dd_content = m.group(2)

        # Extract arXiv ID (prefer from <dd>, fall back to <dt>)
        id_match = id_re.search(dd_content) or id_re.search(dt_content)
        if not id_match:
            continue
        arxiv_id = id_match.group(1)

        # Extract title
        title_match = title_re.search(dd_content)
        if not title_match:
            continue
        title = html.unescape(re.sub(r"\s+", " ", tag_re.sub("", title_match.group(1))).strip())

        # Extract primary subject
        subj_match = subject_re.search(dd_content)
        subjects = html.unescape(re.sub(r"\s+", " ", tag_re.sub("", subj_match.group(1))).strip()) if subj_match else ""

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "subjects": subjects,
            "source": "arXiv_csLG",
        })

    log(f"  → Fetched {len(papers)} papers")
    return papers


def main():
    """Main function to fetch all trends data"""
    log("=== Daily Trends Data Fetcher ===\n")
    start_time = time.time()

    # Fetch from all sources
    hn_stories = fetch_hacker_news()
    reddit_posts = fetch_reddit()
    hatena_bookmarks = fetch_hatena()
    wiz_articles = fetch_wiz_blog()
    arxiv_papers = fetch_arxiv_csLG()

    # Combine results
    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hacker_news": hn_stories,
        "reddit": reddit_posts,
        "hatena": hatena_bookmarks,
        "wiz_blog": wiz_articles,
        "arxiv_cslg": arxiv_papers,
        "summary": {
            "hn_count": len(hn_stories),
            "reddit_count": len(reddit_posts),
            "hatena_count": len(hatena_bookmarks),
            "wiz_count": len(wiz_articles),
            "arxiv_count": len(arxiv_papers),
            "total": len(hn_stories) + len(reddit_posts) + len(hatena_bookmarks) + len(wiz_articles) + len(arxiv_papers),
        },
    }

    elapsed = time.time() - start_time
    log(f"\n=== Collection Complete ===")
    log(f"Total articles: {result['summary']['total']}")
    log(f"Time elapsed: {elapsed:.1f} seconds")

    # Output JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
