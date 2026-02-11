# Automatic arXiv Paper Summarization from Issues

You are tasked with automatically finding and summarizing arXiv papers from open GitHub issues. Follow these instructions carefully.

## 🎯 PRIMARY GOAL: CREATE A PULL REQUEST WITH PAPER SUMMARY

**YOUR TASK IS NOT COMPLETE UNTIL A PULL REQUEST IS CREATED.**

Steps:
1. Find an open issue with an arXiv paper URL
2. Summarize the paper using the `/summarize-arxiv-paper` skill
3. Create a PR and link it to the issue
4. If the issue is not a paper, retry with the next issue (up to 3 attempts)

Do not stop until you have created a PR with a paper summary.

## Step 1: Find Open Issues with arXiv Papers

### Fetch Open Issues

Use the GitHub CLI to get open issues (sorted by oldest first):

```bash
gh issue list --state open --limit 20 --json number,title,body,labels | jq 'sort_by(.number)'
```

### Identify Paper Issues

Look for issues that contain arXiv URLs in the body or title. Valid patterns:
- `https://arxiv.org/abs/YYMM.NNNNN`
- `https://arxiv.org/pdf/YYMM.NNNNN`
- `http://arxiv.org/abs/YYMM.NNNNN`

### Selection Strategy

1. **Sort by issue number** (oldest first) to handle backlog
2. **Extract arXiv URL** from the issue body or title
3. **Skip non-paper issues** (blog posts, Zenn articles, general discussions, etc.)
4. **Skip issues that already have a PR** (open or merged) to avoid duplication
5. **Retry up to 3 times** if the first issue is not a paper or already has a PR

Example filtering logic:
```bash
# Get issues (oldest first) and filter for arXiv URLs
gh issue list --state open --limit 20 --json number,title,body | \
  jq -r 'sort_by(.number) | .[] | select(.body | test("arxiv.org/(abs|pdf)/[0-9]{4}\\.[0-9]{4,5}")) | "\(.number)|\(.title)|\(.body)"' | \
  head -1
```

If no arXiv URL is found in the first issue, try the next one (up to 3 attempts total).

### Deduplication Check

**Before processing any issue**, verify no PR already exists for it:

```bash
# Check if an open or merged PR already references this issue
EXISTING_PR=$(gh pr list --state all --limit 50 --json number,title,body \
  | jq -r ".[] | select(.body | test(\"Closes #$ISSUE_NUMBER\")) | .number" \
  | head -1)

if [ -n "$EXISTING_PR" ]; then
  echo "⏭️  Issue #$ISSUE_NUMBER already has PR #$EXISTING_PR — skipping"
  # Move on to the next candidate issue
fi

# Also check if the branch already exists on the remote
PAPER_ID=$(echo "$ARXIV_URL" | grep -oP '\d{4}\.\d{4,5}')
BRANCH_NAME="paper/arxiv-$PAPER_ID"

if git ls-remote --exit-code --heads origin "$BRANCH_NAME" > /dev/null 2>&1; then
  echo "⏭️  Branch $BRANCH_NAME already exists — skipping"
  # Move on to the next candidate issue
fi
```

If either check finds a match, skip this issue and try the next one (up to the retry limit).

## Step 2: Extract arXiv URL and Issue Information

Once you find a valid paper issue:

1. **Store the issue number**: You'll need this to link the PR
2. **Extract the arXiv URL**: Use regex or string matching
3. **Verify the URL format**: Should match `https://arxiv.org/abs/YYMM.NNNNN`

Example:
```bash
ISSUE_NUMBER=257
ARXIV_URL="https://arxiv.org/abs/1910.10683"
ISSUE_TITLE="T5"
```

## Step 3: Summarize the Paper

### Use the Summarize Skill

**CRITICAL**: Use the `/summarize-arxiv-paper` skill to create the paper summary.

```bash
# This will be executed via the Skill tool
/summarize-arxiv-paper $ARXIV_URL
```

The skill will:
- Fetch the paper from ar5iv
- Extract all content (title, authors, abstract, sections, experiments, etc.)
- Generate a comprehensive Markdown summary
- Save it to `machine-learning/YYYY/<filename>.md`
- Follow all DoD requirements

**IMPORTANT**: The skill should handle all the work. You just need to invoke it with the arXiv URL.

## Step 4: Create Branch and Commit

After the skill completes:

1. **Verify the file was created**:
   ```bash
   # Check if file exists
   find machine-learning -name "*.md" -mmin -5 | head -1
   ```

2. **Create a branch** with descriptive name:
   ```bash
   # Extract paper ID from URL
   PAPER_ID=$(echo "$ARXIV_URL" | grep -oP '\d{4}\.\d{4,5}')
   BRANCH_NAME="paper/arxiv-$PAPER_ID"

   git checkout -b "$BRANCH_NAME"
   ```

3. **Commit the changes**:
   ```bash
   git add machine-learning/
   git add .claude/skills/  # In case skill file was updated

   git commit -m "$(cat <<EOF
Add paper summary from Issue #$ISSUE_NUMBER: $ISSUE_TITLE

Summarize arXiv:$PAPER_ID - $ISSUE_TITLE

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
   ```

## Step 5: Create Pull Request with Issue Link

### Push to Remote

```bash
git push -u origin "$BRANCH_NAME"
```

### Create PR with Issue Reference

**CRITICAL**: Use `Closes #<issue_number>` in the PR body to automatically close the issue when merged.

```bash
# Prepare PR body
cat > /tmp/pr-body.txt <<EOF
## Objective

Automatically summarize arXiv paper from Issue #$ISSUE_NUMBER.

## Effect

- **Paper**: [$ISSUE_TITLE]($ARXIV_URL)
- **Summary file**: \`machine-learning/YYYY/<filename>.md\`
- **Issue**: Closes #$ISSUE_NUMBER

This PR includes a comprehensive summary following the project's DoD requirements:
- Concrete, detailed explanations (not vague statements)
- Clear input/output specifications with tensor dimensions
- Algorithm descriptions with mathematical formulations
- Datasets explicitly listed
- Comparisons with similar/related methods

## Test

- Review the summary for completeness and accuracy
- Verify all mathematical formulations have proper dimensions
- Check that DoD requirements are met (see checklist below)
- Confirm the paper URL matches the issue

## Note

Automatically generated via the \`auto-summarize-papers\` workflow.

Closes #$ISSUE_NUMBER

---

## Definition of Done Checklist

### Common
- [x] Describe the concrete sentences to support understanding (not just writing "I understand ...")
- [x] Describe the condition which can be applied (who, when, where)
- [x] Include information about licenses and copyrights

### Computer Science / Machine Learning
- [x] Clear Input and Output
- [x] Describe Algorithms with pseudocode
- [x] Explain datasets used
- [x] Clear calculation order
- [x] Describe the difference between similar algorithms
EOF

# Create PR with retry logic
SUCCESS=0
for i in 1 2 3; do
  echo "Attempt $i to create PR..."
  if gh pr create --base main --title "Add paper summary: $ISSUE_TITLE (arXiv:$PAPER_ID)" --body-file /tmp/pr-body.txt; then
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
  exit 1
fi
```

**Important**: The `Closes #$ISSUE_NUMBER` syntax in the PR body will automatically close the issue when the PR is merged.

## Step 6: Handle Retry Logic for Non-Paper Issues

If the first issue doesn't contain an arXiv URL:

1. **Log the skip**:
   ```bash
   echo "⚠️  Issue #$ISSUE_NUMBER is not a paper (no arXiv URL found), trying next issue..."
   ```

2. **Try the next issue** (up to 3 total attempts):
   ```bash
   ATTEMPT=1
   MAX_ATTEMPTS=3
   FOUND_PAPER=0

   while [ $ATTEMPT -le $MAX_ATTEMPTS ] && [ $FOUND_PAPER -eq 0 ]; do
     # Get the next open issue (oldest first)
     ISSUE_DATA=$(gh issue list --state open --limit 20 --json number,title,body | \
       jq -r "sort_by(.number) | .[$ATTEMPT] | select(.body | test(\"arxiv.org/(abs|pdf)/[0-9]{4}\\.[0-9]{4,5}\"))")

     if [ -n "$ISSUE_DATA" ]; then
       FOUND_PAPER=1
       # Process this issue
     else
       echo "⚠️  Attempt $ATTEMPT: No paper found, trying next..."
       ATTEMPT=$((ATTEMPT + 1))
     fi
   done

   if [ $FOUND_PAPER -eq 0 ]; then
     echo "❌ No paper issues found after $MAX_ATTEMPTS attempts"
     exit 1
   fi
   ```

3. **If all 3 attempts fail**: Create a summary issue documenting that no papers were found

## Error Handling

### If Paper Fetch Fails

If the `/summarize-arxiv-paper` skill fails:
1. Check the error message
2. Verify the arXiv URL is correct
3. Try the ar5iv URL directly
4. If still failing, comment on the issue and move to the next one

### If No Paper Issues Found

If no open issues contain arXiv URLs after 3 attempts:
1. **Success condition**: No work needed, exit gracefully
2. **Log message**: "No paper issues found in the backlog"
3. **Exit code**: 0 (success, nothing to do)

```bash
echo "✅ No paper issues found in the backlog. All papers are summarized!"
exit 0
```

### If PR Creation Fails

If `gh pr create` fails after 3 retries:
1. Check if branch was pushed successfully
2. Verify GitHub token permissions
3. Log the error and exit with failure code

## Issue Types to Skip

**Skip these issue types** (not papers):
- Blog posts (dev.to, medium.com, zenn.dev, note.com, qiita.com)
- General articles (not arXiv)
- Discussion threads (no URL in body)
- Tutorial links
- Tool documentation
- Social media posts (x.com, twitter.com)

**Only process issues with**:
- `arxiv.org/abs/` or `arxiv.org/pdf/` URLs
- Clear paper references

## Example Workflow

```bash
# 1. Get open issues (oldest first) that contain arXiv URLs
ISSUES=$(gh issue list --state open --limit 20 --json number,title,body \
  | jq 'sort_by(.number) | [.[] | select(.body | test("arxiv.org/(abs|pdf)/[0-9]{4}\\.[0-9]{4,5}"))]')

ISSUE_COUNT=$(echo "$ISSUES" | jq 'length')

if [ "$ISSUE_COUNT" -eq 0 ]; then
  echo "✅ No paper issues found"
  exit 0
fi

# 2. Iterate through candidate issues, skipping ones that already have a PR
FOUND=0
for i in $(seq 0 $((ISSUE_COUNT - 1))); do
  PAPER_ISSUE=$(echo "$ISSUES" | jq -r ".[$i] | \"\(.number)|\(.title)|\(.body)\"")
  ISSUE_NUMBER=$(echo "$PAPER_ISSUE" | cut -d'|' -f1)
  ISSUE_TITLE=$(echo "$PAPER_ISSUE" | cut -d'|' -f2)
  ARXIV_URL=$(echo "$PAPER_ISSUE" | grep -oP 'https?://arxiv\.org/(abs|pdf)/[0-9]{4}\.[0-9]{4,5}' | head -1)
  ARXIV_URL=$(echo "$ARXIV_URL" | sed 's|/pdf/|/abs/|')
  PAPER_ID=$(echo "$ARXIV_URL" | grep -oP '\d{4}\.\d{4,5}')
  BRANCH_NAME="paper/arxiv-$PAPER_ID"

  # 3. Deduplication: skip if a PR already closes this issue
  EXISTING_PR=$(gh pr list --state all --limit 50 --json number,title,body \
    | jq -r ".[] | select(.body | test(\"Closes #$ISSUE_NUMBER\")) | .number" | head -1)
  if [ -n "$EXISTING_PR" ]; then
    echo "⏭️  Issue #$ISSUE_NUMBER already has PR #$EXISTING_PR — skipping"
    continue
  fi

  # 4. Deduplication: skip if branch already exists on the remote
  if git ls-remote --exit-code --heads origin "$BRANCH_NAME" > /dev/null 2>&1; then
    echo "⏭️  Branch $BRANCH_NAME already exists — skipping"
    continue
  fi

  echo "📄 Found paper: Issue #$ISSUE_NUMBER - $ISSUE_TITLE"
  echo "🔗 URL: $ARXIV_URL"
  FOUND=1
  break
done

if [ "$FOUND" -eq 0 ]; then
  echo "✅ All paper issues already have PRs. Nothing to do."
  exit 0
fi

# 5. Summarize using skill
# (This will be done via Skill tool invocation)

# 6. Create PR with issue link
# (Follow Step 5 above)
```

## Final Checklist

**🎯 MANDATORY:**
- [ ] **Paper issue found**: With valid arXiv URL (or gracefully exit if none)
- [ ] **Deduplication checked**: Confirmed no existing PR closes this issue and the branch does not already exist
- [ ] **Paper summarized**: Using `/summarize-arxiv-paper` skill
- [ ] **File created**: In `machine-learning/YYYY/<filename>.md`
- [ ] **PR created**: With `Closes #<issue_number>` in body
- [ ] **Branch pushed**: To remote repository
- [ ] **Commit made**: With descriptive message referencing issue

**Quality checks:**
- [ ] **Summary complete**: All sections filled per DoD requirements
- [ ] **Math formatted**: LaTeX/MathJax notation correct
- [ ] **Dimensions specified**: All tensor/matrix dimensions explicit
- [ ] **Datasets listed**: All datasets from experiments documented
- [ ] **Comparisons included**: Differences from similar algorithms explained
- [ ] **License info**: arXiv license and copyright in meta section

## Success Criteria

**MINIMUM SUCCESS:**
1. **PR created and linked to issue**: With `Closes #<issue_number>`
2. **Paper summary file created**: Following project conventions

**IDEAL SUCCESS:**
3. **DoD requirements met**: All checklist items satisfied
4. **Retry handled gracefully**: If first issue not a paper, try next (up to 3)
5. **No errors**: Smooth execution from issue → summary → PR

**Graceful exit conditions:**
- No open issues with arXiv papers: Exit with success (nothing to do)
- All candidate issues already have PRs or branches: Exit with success (nothing to do)
- All 3 retry attempts found non-paper issues: Exit with message

**What constitutes failure:**
- Paper issue found but summary failed (after retries)
- Summary created but PR creation failed (after retries)
- Any errors in git operations (commit, push)
