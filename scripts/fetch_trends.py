#!/usr/bin/env python3
"""
Daily Trends Data Fetcher
Fetches data from Hacker News, Reddit, and Hatena Bookmark with proper rate limiting
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from xml.etree import ElementTree as ET

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
        print(f"Error fetching {url}: {e}", flush=True)
        return None


def fetch_text(url, user_agent, timeout=10):
    """Fetch text/XML from URL with proper User-Agent"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode()
    except Exception as e:
        print(f"Error fetching {url}: {e}", flush=True)
        return None


def fetch_hacker_news():
    """Fetch Hacker News top stories with 50+ points"""
    print("[1/5] Fetching Hacker News...", flush=True)
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

    print(f"  → Fetched {len(stories)} stories", flush=True)
    return stories


def fetch_reddit():
    """Fetch Reddit posts from AI, Tech, and General subreddits"""
    print("[2/5] Fetching Reddit...", flush=True)
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
            print(f"  → Fetching r/{sub}...", flush=True)

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

    print(f"  → Fetched {len(posts)} posts", flush=True)
    return posts


def fetch_hatena():
    """Fetch Hatena Bookmark hot entries from RSS feeds"""
    print("[3/5] Fetching Hatena Bookmark...", flush=True)
    bookmarks = []

    categories = [
        ("https://b.hatena.ne.jp/hotentry/it", "Tech"),
    ]

    for url, category in categories:
        time.sleep(5)  # 5-second delay per Hatena robots.txt Crawl-delay directive
        print(f"  → Fetching {url}...", flush=True)

        xml_content = fetch_text(url, HATENA_UA)
        if not xml_content:
            continue

        try:
            root = ET.fromstring(xml_content)
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")

                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ""
                    link = link_elem.text or ""

                    # Try to extract bookmark count from description
                    desc_elem = item.find("description")
                    bookmarks_count = 10  # Default

                    if desc_elem is not None and desc_elem.text:
                        # Simple heuristic: look for numbers in description
                        import re

                        numbers = re.findall(r"\d+", desc_elem.text)
                        if numbers:
                            bookmarks_count = int(numbers[0])

                    # Filter: 10+ bookmarks (or just include all from hot entries)
                    if bookmarks_count >= 10 or link:
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
            print(f"  Error parsing XML: {e}", flush=True)
            continue

    print(f"  → Fetched {len(bookmarks)} bookmarks", flush=True)
    return bookmarks


def fetch_wiz_blog():
    """Fetch Wiz Security Blog via RSS feed.
    robots.txt: Allow: / for all agents (no restrictions, no Crawl-delay).
    Uses RSS feed at https://www.wiz.io/feed/rss.xml for reliable parsing.
    """
    print("[4/5] Fetching Wiz Blog...", flush=True)
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

            title = (title_elem.text or "").strip().lstrip("<![CDATA[").rstrip("]]>")
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
        print(f"  Error parsing Wiz RSS: {e}", flush=True)

    print(f"  → Fetched {len(articles)} articles", flush=True)
    return articles


def fetch_arxiv_csLG():
    """Fetch recent arXiv cs.LG (Machine Learning) paper listings.
    robots.txt: Crawl-delay: 15 (User-agent: *), /list path is explicitly allowed.
    Fetches a single page to respect crawl-delay; sleeps 15 s before request.
    """
    import re

    print("[5/5] Fetching arXiv cs.LG/recent...", flush=True)
    papers = []

    time.sleep(15)  # Respect Crawl-delay: 15 per arxiv.org robots.txt
    html = fetch_text("https://arxiv.org/list/cs.LG/recent", ARXIV_UA)
    if not html:
        return papers

    # arXiv listing pages use <dt>/<dd> pairs.
    # <dt> contains the sequence number and arXiv ID link.
    # <dd> contains title (class="list-title"), authors, and subjects.
    dt_dd_re = re.compile(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", re.DOTALL)
    id_re = re.compile(r'href="/abs/(\d{4}\.\d{4,5})"')
    # Title: <div class="list-title mathjax"><span class="descriptor">Title:</span>\nPAPER TITLE</div>
    title_re = re.compile(
        r'class="list-title[^"]*"[^>]*>.*?class="descriptor"[^>]*>.*?</span>\s*(.*?)\s*</div>',
        re.DOTALL | re.IGNORECASE,
    )
    # Subjects: <span class="primary-subject">Machine Learning (cs.LG)</span>
    subject_re = re.compile(
        r'class="primary-subject"[^>]*>(.*?)</span>', re.DOTALL | re.IGNORECASE
    )
    tag_re = re.compile(r"<[^>]+>")

    for m in dt_dd_re.finditer(html):
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
        title = re.sub(r"\s+", " ", tag_re.sub("", title_match.group(1))).strip()

        # Extract primary subject
        subj_match = subject_re.search(dd_content)
        subjects = re.sub(r"\s+", " ", tag_re.sub("", subj_match.group(1))).strip() if subj_match else ""

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "subjects": subjects,
            "source": "arXiv_csLG",
        })

    print(f"  → Fetched {len(papers)} papers", flush=True)
    return papers


def main():
    """Main function to fetch all trends data"""
    print("=== Daily Trends Data Fetcher ===\n", flush=True)
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
    print(f"\n=== Collection Complete ===", flush=True)
    print(f"Total articles: {result['summary']['total']}", flush=True)
    print(f"Time elapsed: {elapsed:.1f} seconds", flush=True)

    # Output JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
