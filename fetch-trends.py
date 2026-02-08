#!/usr/bin/env python3
"""
Daily Trends Data Collector
Fetches articles from Hacker News, Reddit, and Hatena Bookmark
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

def fetch_url(url: str, user_agent: str) -> str:
    """Fetch URL with proper user agent"""
    req = urllib.request.Request(url, headers={'User-Agent': user_agent})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def fetch_hacker_news() -> List[Dict[str, Any]]:
    """Fetch top stories from Hacker News"""
    print("Fetching Hacker News stories...")
    user_agent = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"

    # Get top story IDs
    stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    stories_data = fetch_url(stories_url, user_agent)
    if not stories_data:
        return []

    story_ids = json.loads(stories_data)[:30]  # Top 30
    articles = []

    for story_id in story_ids:
        time.sleep(1)  # 1-second delay between requests
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item_data = fetch_url(item_url, user_agent)

        if item_data:
            item = json.loads(item_data)
            # Filter for stories with 50+ points
            if item.get('type') == 'story' and item.get('score', 0) >= 50:
                articles.append({
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'score': item.get('score'),
                    'by': item.get('by'),
                    'time': item.get('time'),
                    'descendants': item.get('descendants', 0)
                })
                print(f"  Added: {item.get('title')} ({item.get('score')} points)")

    print(f"Fetched {len(articles)} HN stories")
    return articles

def fetch_reddit() -> List[Dict[str, Any]]:
    """Fetch hot posts from Reddit subreddits"""
    print("Fetching Reddit posts...")
    user_agent = "web:understanding-trends:v1.0.0 (by /u/trends_bot)"

    subreddits = [
        # AI (60%)
        'MachineLearning', 'LocalLLaMA', 'OpenAI', 'ClaudeAI', 'AutoGPT',
        # Tech (30%)
        'programming', 'SideProject', 'opensource', 'webdev',
        # General (10%)
        'worldnews', 'japan'
    ]

    articles = []

    for subreddit in subreddits:
        time.sleep(1)  # 1-second delay between requests
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        data = fetch_url(url, user_agent)

        if data:
            try:
                reddit_data = json.loads(data)
                posts = reddit_data.get('data', {}).get('children', [])

                for post in posts:
                    post_data = post.get('data', {})
                    upvotes = post_data.get('ups', 0)
                    comments = post_data.get('num_comments', 0)

                    # Filter: 100+ upvotes OR 50+ comments
                    if upvotes >= 100 or comments >= 50:
                        articles.append({
                            'subreddit': subreddit,
                            'title': post_data.get('title'),
                            'url': post_data.get('url'),
                            'permalink': f"https://reddit.com{post_data.get('permalink')}",
                            'upvotes': upvotes,
                            'comments': comments,
                            'author': post_data.get('author'),
                            'created_utc': post_data.get('created_utc')
                        })
                        print(f"  Added: {post_data.get('title')} ({upvotes} upvotes)")
            except Exception as e:
                print(f"Error parsing Reddit data for r/{subreddit}: {e}")

    print(f"Fetched {len(articles)} Reddit posts")
    return articles

def fetch_hatena() -> List[Dict[str, Any]]:
    """Fetch entries from Hatena Bookmark RSS feeds"""
    print("Fetching Hatena Bookmark entries...")
    user_agent = "Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"

    feeds = [
        ('https://b.hatena.ne.jp/hotentry/it.rss', 'Tech-IT'),
        ('https://b.hatena.ne.jp/hotentry/technology.rss', 'Tech-Technology'),
        ('https://b.hatena.ne.jp/hotentry/all.rss', 'General')
    ]

    articles = []

    for feed_url, category in feeds:
        time.sleep(2)  # 2-second delay between requests
        rss_data = fetch_url(feed_url, user_agent)

        if rss_data:
            try:
                root = ET.fromstring(rss_data)

                # Parse RSS feed
                for item in root.findall('.//item'):
                    title = item.find('title')
                    link = item.find('link')
                    description = item.find('description')
                    pub_date = item.find('pubDate')

                    # Try to extract bookmark count from description
                    bookmarks = 0
                    if description is not None and description.text:
                        # Hatena typically includes bookmark count in description
                        desc_text = description.text
                        # Look for patterns like "10 users" or bookmark counts
                        import re
                        match = re.search(r'(\d+)\s*users?', desc_text)
                        if match:
                            bookmarks = int(match.group(1))

                    # Filter: 10+ bookmarks (if we can't determine, include it)
                    if bookmarks >= 10 or bookmarks == 0:
                        articles.append({
                            'category': category,
                            'title': title.text if title is not None else '',
                            'url': link.text if link is not None else '',
                            'bookmarks': bookmarks if bookmarks > 0 else 'unknown',
                            'pub_date': pub_date.text if pub_date is not None else ''
                        })
                        if title is not None:
                            print(f"  Added: {title.text} ({bookmarks if bookmarks > 0 else '?'} bookmarks)")
            except Exception as e:
                print(f"Error parsing Hatena RSS for {category}: {e}")

    print(f"Fetched {len(articles)} Hatena entries")
    return articles

def main():
    """Main execution function"""
    print("Starting daily trends data collection...")
    print("=" * 60)

    # Collect data from all sources
    hn_data = fetch_hacker_news()
    print("=" * 60)

    reddit_data = fetch_reddit()
    print("=" * 60)

    hatena_data = fetch_hatena()
    print("=" * 60)

    # Create output structure
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hacker_news": hn_data,
        "reddit": reddit_data,
        "hatena": hatena_data,
        "summary": {
            "hn_count": len(hn_data),
            "reddit_count": len(reddit_data),
            "hatena_count": len(hatena_data),
            "total": len(hn_data) + len(reddit_data) + len(hatena_data)
        }
    }

    # Save to file
    output_file = "/home/runner/work/understanding/understanding/trends-data-raw.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nData collection complete!")
    print(f"Summary:")
    print(f"  Hacker News: {output['summary']['hn_count']} articles")
    print(f"  Reddit: {output['summary']['reddit_count']} posts")
    print(f"  Hatena: {output['summary']['hatena_count']} entries")
    print(f"  Total: {output['summary']['total']} items")
    print(f"\nData saved to: {output_file}")

if __name__ == "__main__":
    main()
