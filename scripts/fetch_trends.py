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
    print("[1/3] Fetching Hacker News...", flush=True)
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
    print("[2/3] Fetching Reddit...", flush=True)
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
                f"https://www.reddit.com/r/{sub}/hot.json?limit=25", REDDIT_UA
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
    print("[3/3] Fetching Hatena Bookmark...", flush=True)
    bookmarks = []

    categories = [
        ("https://b.hatena.ne.jp/hotentry/it.rss", "Tech"),
        ("https://b.hatena.ne.jp/hotentry/technology.rss", "Tech"),
        ("https://b.hatena.ne.jp/hotentry/all.rss", "General"),
    ]

    for url, category in categories:
        time.sleep(2)  # 2-second delay for Hatena
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


def main():
    """Main function to fetch all trends data"""
    print("=== Daily Trends Data Fetcher ===\n", flush=True)
    start_time = time.time()

    # Fetch from all sources
    hn_stories = fetch_hacker_news()
    reddit_posts = fetch_reddit()
    hatena_bookmarks = fetch_hatena()

    # Combine results
    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hacker_news": hn_stories,
        "reddit": reddit_posts,
        "hatena": hatena_bookmarks,
        "summary": {
            "hn_count": len(hn_stories),
            "reddit_count": len(reddit_posts),
            "hatena_count": len(hatena_bookmarks),
            "total": len(hn_stories) + len(reddit_posts) + len(hatena_bookmarks),
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
