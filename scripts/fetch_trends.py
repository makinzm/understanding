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
from datetime import datetime, timezone
from xml.etree import ElementTree as ET


def log(message):
    """Print progress/diagnostic messages to stderr so stdout stays pure JSON."""
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
            # Hatena's hotentry feed is RSS 1.0 (RDF) and declares a default
            # xmlns, so <item>/<title>/<link> parse as namespaced elements
            # (e.g. "{http://purl.org/rss/1.0/}item") rather than bare tags.
            # Match by local name (suffix after "}") to stay namespace-agnostic.
            def local_name(tag):
                return tag.rsplit("}", 1)[-1]

            def find_child(elem, name):
                for child in elem:
                    if local_name(child.tag) == name:
                        return child
                return None

            for item in root.iter():
                if local_name(item.tag) != "item":
                    continue
                title_elem = find_child(item, "title")
                link_elem = find_child(item, "link")

                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ""
                    link = link_elem.text or ""

                    # This RSS 1.0 feed has no dedicated bookmark-count
                    # element — only title/link/description/date. Scanning
                    # the description for digits is unreliable (it matches
                    # unrelated numbers like dates within the article text),
                    # so report the count as unknown (None) rather than
                    # fabricating one. Inclusion in "hotentry" already
                    # implies it cleared Hatena's own popularity threshold.
                    if link:
                        bookmarks.append(
                            {
                                "title": title,
                                "url": link,
                                "bookmarks": None,
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

        # The feed contains the blog's full history (600+ posts), not just
        # recent ones. It's ordered newest-first, so cap to the most recent
        # entries to keep this a "daily trends" source rather than a full
        # archive dump.
        for item in items[:20]:
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

            # ElementTree already decodes CDATA sections into plain text, so
            # no manual unwrapping is needed here. (A prior version called
            # .lstrip("<![CDATA[") / .rstrip("]]>"), but str.strip family
            # treats its argument as a character set, not a literal prefix —
            # that silently ate real leading/trailing letters from titles,
            # e.g. "The Wiz Red Agent..." became "he Wiz Red Agent...".
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

    log(f"  → Fetched {len(articles)} articles")
    return articles


def fetch_arxiv_csLG():
    """Fetch recent arXiv cs.LG (Machine Learning) paper listings.
    robots.txt: Crawl-delay: 15 (User-agent: *), /list path is explicitly allowed.
    Fetches a single page to respect crawl-delay; sleeps 15 s before request.
    """
    import html as html_module
    import re

    log("[5/5] Fetching arXiv cs.LG/recent...")
    papers = []

    time.sleep(15)  # Respect Crawl-delay: 15 per arxiv.org robots.txt
    html = fetch_text("https://arxiv.org/list/cs.LG/recent", ARXIV_UA)
    if not html:
        return papers

    # arXiv listing pages use <dt>/<dd> pairs.
    # <dt> contains the sequence number and arXiv ID link.
    # <dd> contains title (class="list-title"), authors, and subjects.
    dt_dd_re = re.compile(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", re.DOTALL)
    # arXiv's listing HTML mixes single/double quotes and sometimes has a
    # stray space before "=" (e.g. `href ="/abs/..."`), so quotes and
    # whitespace around "=" must both be tolerated.
    id_re = re.compile(r'href\s*=\s*["\']/abs/(\d{4}\.\d{4,5})["\']')
    # Title: <div class='list-title mathjax'><span class='descriptor'>Title:</span>\nPAPER TITLE</div>
    title_re = re.compile(
        r'class=[\'"]list-title[^\'"]*[\'"][^>]*>.*?class=[\'"]descriptor[\'"][^>]*>.*?</span>\s*(.*?)\s*</div>',
        re.DOTALL | re.IGNORECASE,
    )
    # Subjects: <span class="primary-subject">Machine Learning (cs.LG)</span>
    subject_re = re.compile(
        r'class=[\'"]primary-subject[\'"][^>]*>(.*?)</span>', re.DOTALL | re.IGNORECASE
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
        title = html_module.unescape(re.sub(r"\s+", " ", tag_re.sub("", title_match.group(1))).strip())

        # Extract primary subject
        subj_match = subject_re.search(dd_content)
        subjects = html_module.unescape(re.sub(r"\s+", " ", tag_re.sub("", subj_match.group(1))).strip()) if subj_match else ""

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
