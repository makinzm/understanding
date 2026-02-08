#!/bin/bash

# Daily Trends Data Fetcher
# Fetches data from Hacker News, Reddit, and Hatena Bookmark
# Respects rate limits and ToS compliance

set -euo pipefail

OUTPUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUTPUT_DIR"' EXIT

echo "=== Fetching Daily Trends ===" >&2
echo "Output directory: $OUTPUT_DIR" >&2
echo "" >&2

# User-Agent strings
HN_UA="Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"
REDDIT_UA="web:understanding-trends:v1.0.0 (by /u/trends_bot)"
HATENA_UA="Mozilla/5.0 (compatible; TrendBot/1.0; +https://github.com/understanding/trends)"

# ====== Hacker News ======
echo "[1/3] Fetching Hacker News..." >&2
HN_FILE="$OUTPUT_DIR/hn.json"
echo '{"stories":[]}' > "$HN_FILE"

# Fetch top story IDs
STORY_IDS=$(curl -s -H "User-Agent: $HN_UA" \
  "https://hacker-news.firebaseio.com/v0/topstories.json" | \
  jq -r '.[:30][]')

# Fetch details for each story (with 1-second delay)
FIRST=1
for id in $STORY_IDS; do
  if [ "$FIRST" -eq 0 ]; then
    sleep 1  # Rate limiting: 1 second between requests
  fi
  FIRST=0

  STORY=$(curl -s -H "User-Agent: $HN_UA" \
    "https://hacker-news.firebaseio.com/v0/item/$id.json")

  # Only include stories with 50+ points
  SCORE=$(echo "$STORY" | jq -r '.score // 0')
  if [ "$SCORE" -ge 50 ]; then
    echo "$STORY" | jq -c '.' >> "$OUTPUT_DIR/hn_stories_raw.json"
  fi
done

# Combine into JSON array
if [ -f "$OUTPUT_DIR/hn_stories_raw.json" ]; then
  jq -s '.' "$OUTPUT_DIR/hn_stories_raw.json" > "$HN_FILE"
else
  echo '[]' > "$HN_FILE"
fi

echo "  → Fetched $(jq 'length' "$HN_FILE") stories" >&2

# ====== Reddit ======
echo "[2/3] Fetching Reddit..." >&2
REDDIT_FILE="$OUTPUT_DIR/reddit.json"
echo '[]' > "$REDDIT_FILE"

# AI-focused subreddits (60% weight)
AI_SUBS=("MachineLearning" "LocalLLaMA" "OpenAI" "ClaudeAI" "AutoGPT")

# Tech-focused subreddits (30% weight)
TECH_SUBS=("programming" "SideProject" "opensource" "webdev")

# General subreddits (10% weight)
GENERAL_SUBS=("worldnews" "japan")

ALL_SUBS=("${AI_SUBS[@]}" "${TECH_SUBS[@]}" "${GENERAL_SUBS[@]}")

for sub in "${ALL_SUBS[@]}"; do
  sleep 1  # Rate limiting: 1 second between requests

  echo "  → Fetching r/$sub..." >&2

  POSTS=$(curl -s -H "User-Agent: $REDDIT_UA" \
    "https://www.reddit.com/r/$sub/hot.json?limit=25" | \
    jq -c '[.data.children[].data | select(.ups >= 100 or .num_comments >= 50) | {
      title: .title,
      url: .url,
      permalink: ("https://www.reddit.com" + .permalink),
      ups: .ups,
      num_comments: .num_comments,
      created_utc: .created_utc,
      subreddit: .subreddit,
      category: (if .subreddit == "MachineLearning" or .subreddit == "LocalLLaMA" or .subreddit == "OpenAI" or .subreddit == "ClaudeAI" or .subreddit == "AutoGPT" then "AI" elif .subreddit == "worldnews" or .subreddit == "japan" then "General" else "Tech" end)
    }]')

  # Merge with existing results
  TEMP=$(mktemp)
  jq -s 'add' "$REDDIT_FILE" <(echo "$POSTS") > "$TEMP"
  mv "$TEMP" "$REDDIT_FILE"
done

echo "  → Fetched $(jq 'length' "$REDDIT_FILE") posts" >&2

# ====== Hatena Bookmark ======
echo "[3/3] Fetching Hatena Bookmark..." >&2
HATENA_FILE="$OUTPUT_DIR/hatena.json"
echo '[]' > "$HATENA_FILE"

# Fetch IT category RSS
sleep 2
IT_RSS=$(curl -s -H "User-Agent: $HATENA_UA" \
  "https://b.hatena.ne.jp/hotentry/it.rss" || echo "")

if [ -n "$IT_RSS" ]; then
  echo "$IT_RSS" | grep -oP '<item>.*?</item>' | while IFS= read -r item; do
    TITLE=$(echo "$item" | grep -oP '<title>\K[^<]+' | head -1)
    LINK=$(echo "$item" | grep -oP '<link>\K[^<]+' | head -1)
    BOOKMARKS=$(echo "$item" | grep -oP 'users?|bookmarks?' | wc -l)

    if [ "$BOOKMARKS" -ge 10 ] || [ -n "$LINK" ]; then
      echo "{\"title\":\"$TITLE\",\"url\":\"$LINK\",\"bookmarks\":$BOOKMARKS,\"category\":\"Tech\"}" >> "$OUTPUT_DIR/hatena_raw.json"
    fi
  done 2>/dev/null || true
fi

# Fetch Technology category RSS
sleep 2
TECH_RSS=$(curl -s -H "User-Agent: $HATENA_UA" \
  "https://b.hatena.ne.jp/hotentry/technology.rss" || echo "")

if [ -n "$TECH_RSS" ]; then
  echo "$TECH_RSS" | grep -oP '<item>.*?</item>' | while IFS= read -r item; do
    TITLE=$(echo "$item" | grep -oP '<title>\K[^<]+' | head -1)
    LINK=$(echo "$item" | grep -oP '<link>\K[^<]+' | head -1)
    BOOKMARKS=$(echo "$item" | grep -oP 'users?|bookmarks?' | wc -l)

    if [ "$BOOKMARKS" -ge 10 ] || [ -n "$LINK" ]; then
      echo "{\"title\":\"$TITLE\",\"url\":\"$LINK\",\"bookmarks\":$BOOKMARKS,\"category\":\"Tech\"}" >> "$OUTPUT_DIR/hatena_raw.json"
    fi
  done 2>/dev/null || true
fi

# Combine Hatena results
if [ -f "$OUTPUT_DIR/hatena_raw.json" ]; then
  jq -s '.' "$OUTPUT_DIR/hatena_raw.json" > "$HATENA_FILE" 2>/dev/null || echo '[]' > "$HATENA_FILE"
fi

echo "  → Fetched $(jq 'length' "$HATENA_FILE" 2>/dev/null || echo 0) bookmarks" >&2

# ====== Output Combined Results ======
echo "" >&2
echo "=== Collection Complete ===" >&2

# Output JSON with all data
jq -n \
  --argjson hn "$(cat "$HN_FILE")" \
  --argjson reddit "$(cat "$REDDIT_FILE")" \
  --argjson hatena "$(cat "$HATENA_FILE")" \
  '{
    hacker_news: $hn,
    reddit: $reddit,
    hatena: $hatena,
    summary: {
      hn_count: ($hn | length),
      reddit_count: ($reddit | length),
      hatena_count: ($hatena | length),
      total: (($hn | length) + ($reddit | length) + ($hatena | length))
    }
  }'
