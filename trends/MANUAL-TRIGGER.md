# How to Manually Trigger Daily Trends Workflow

The daily trends workflow runs automatically at 6:00 AM JST every day. However, you can also trigger it manually/ad-hoc whenever needed.

## Method 1: GitHub Web Interface (Recommended)

1. **Navigate to Actions**:
   - Go to your repository on GitHub
   - Click the **"Actions"** tab at the top

2. **Select the Workflow**:
   - In the left sidebar, find and click **"Daily AI & Tech Trends"**

3. **Run Workflow**:
   - Click the **"Run workflow"** button (on the right side)
   - Select the branch (usually `main` or your current branch)
   - Click the **"Run workflow"** button in the dropdown

4. **Monitor Progress**:
   - The workflow will start immediately
   - You'll see it appear in the workflow runs list
   - Click on it to see live logs and progress

## Method 2: GitHub CLI (Command Line)

If you have [GitHub CLI](https://cli.github.com/) installed:

```bash
# Trigger the workflow on the main branch
gh workflow run daily-trends.yml --ref main

# Check the status
gh run list --workflow=daily-trends.yml --limit 5

# Watch the latest run in real-time
gh run watch
```

## Method 3: GitHub API (Advanced)

Using `curl` with GitHub API:

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_personal_access_token"

# Trigger the workflow
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/YOUR_USERNAME/understanding/actions/workflows/daily-trends.yml/dispatches \
  -d '{"ref":"main"}'
```

## When to Use Manual Triggering

**Testing scenarios**:
- After updating the prompt file (`.github/prompts/daily-trends-prompt.md`)
- After modifying the workflow itself
- Testing new sources or filters
- Verifying ToS compliance changes

**Ad-hoc collection**:
- When you want trends for a specific day (outside 6:00 AM schedule)
- When the automatic run failed and you want to retry
- When you want to collect trends for comparison at different times of day

**Recovery scenarios**:
- If the scheduled run failed due to API issues
- If you missed a day and want to backfill
- If GitHub Actions was down during the scheduled time

## Important Notes

- Each manual run creates a new PR (just like the scheduled run)
- The filename will be based on the date when the workflow runs
- If you run it multiple times on the same day, you'll get multiple PRs with the same date
- The workflow respects all rate limits and ToS requirements, whether triggered manually or automatically

## Troubleshooting

**"Run workflow" button is grayed out**:
- You need write access to the repository
- Make sure you're logged in to GitHub

**Workflow fails immediately**:
- Check that `CLAUDE_CODE_OAUTH_TOKEN` secret is set
- Verify the secret has the correct value

**Workflow succeeds but no PR created**:
- Check the workflow logs for errors
- Verify branch permissions allow PR creation
- Check if there were any source failures

## See Also

- [Workflow file](../.github/workflows/daily-trends.yml)
- [Prompt instructions](../.github/prompts/daily-trends-prompt.md)
- [Compliance notes](../.github/prompts/compliance-notes.md)
- [GitHub Actions documentation](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow)
