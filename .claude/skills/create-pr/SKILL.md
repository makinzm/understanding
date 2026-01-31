---
name: create-pr
description: Create a GitHub pull request using gh CLI with the project's PR template and DoD checklist
disable-model-invocation: true
allowed-tools: Bash(gh *), Bash(git *)
---

Create a pull request for the current branch using the `gh` CLI.

## Steps

1. Run `git status` and `git log main..HEAD --oneline` to understand what changes are included.
2. Run `git diff main...HEAD` to review the full diff.
3. Check if the branch has been pushed to remote. If not, push with `git push -u origin HEAD`.
4. Draft the PR body using the template below, filling in each section based on the changes.
5. Create the PR with `gh pr create --base main --title "<title>" --body "<body>"`.
6. Return the PR URL.

## PR Body Template

```
## Objective
<What this PR aims to accomplish>

## Effect
<What changes were made and their impact>

## Test
<How to verify the changes>

## Note
<Any additional context or caveats>

---

## Definition of Done Checklist

### Common
- [ ] Describe the concrete sentences to support understanding (not just writing "I understand ...")
- [ ] Describe the condition which can be applied (who, when, where)
- [ ] Include information about licenses and copyrights

### Computer Science / Machine Learning (if applicable)
- [ ] Clear Input and Output
- [ ] Describe Algorithms with pseudocode
- [ ] Explain datasets used
- [ ] Clear calculation order
- [ ] Describe the difference between similar algorithms
```
