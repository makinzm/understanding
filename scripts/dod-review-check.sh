#!/usr/bin/env bash
# Stop hook (repo-local): block completion if a deliverable (.md file
# written/edited this session, under a tracked content directory) exists
# but the DoD review Agents relevant to ITS scope haven't been invoked.
#
# Scope-aware by design: unlike a naive "require every dod-* Agent on any
# .md file" check, this maps a touched file's top-level directory to the
# scopes that actually apply to it (dod/deliverables-dod.json ->
# sync-dod-skills.py) and only requires Agents whose scopes intersect
# that set -- so e.g. an ML-pseudocode check is never demanded on a dance
# document, and a dance-provenance check is never demanded on a chess
# document.
#
# Re-checks the transcript on every Stop attempt (no stop_hook_active
# bypass) so it keeps blocking across however many retries it takes until
# the missing Agents are actually dispatched and show up in the transcript.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/.claude/skills"

input=$(cat)
transcript_path=$(echo "$input" | jq -r '.transcript_path // ""')

if [ -z "$transcript_path" ] || [ ! -f "$transcript_path" ]; then
    exit 0
fi

# Deliverable files = .md files written/edited this session under a
# tracked content directory. Anything else (.claude/, dod/, docs/,
# scripts/, trends/) is this factory's own infra or auto-generated, not a
# reviewable deliverable.
deliverable_files=$(grep -oE '"name":"(Write|Edit)","input":\{"file_path":"[^"]*\.md"' "$transcript_path" 2>/dev/null \
    | sed -E 's/.*"file_path":"([^"]*)"/\1/' \
    | grep -E '/(machine-learning|software-engineering|foods|dance|chess)/' \
    | sort -u || true)

if [ -z "$deliverable_files" ]; then
    exit 0
fi

# Map each touched file's top-level content directory to its scopes, and
# union them across all deliverable files touched this session.
required_scopes=""
while IFS= read -r f; do
    case "$f" in
        */machine-learning/*) dirscope="machine-learning" ;;
        */dance/*) dirscope="dance" ;;
        */chess/*) dirscope="chess" ;;
        *) dirscope="" ;;  # software-engineering, foods -> common scope only
    esac
    required_scopes="$required_scopes common $dirscope"
done <<< "$deliverable_files"
required_scopes=$(echo "$required_scopes" | tr ' ' '\n' | grep -v '^$' | sort -u)

# Build the set of Agents whose declared scopes intersect required_scopes,
# reading each generated Skill's stable "**Scopes:**" line.
required_agents=""
for skill_dir in "$SKILLS_DIR"/dod-understanding-*/; do
    [ -d "$skill_dir" ] || continue
    id=$(basename "$skill_dir")
    id=${id#dod-understanding-}
    scopes_line=$(grep -m1 '^\*\*Scopes:\*\*' "$skill_dir/SKILL.md" 2>/dev/null | sed -E 's/\*\*Scopes:\*\* //')
    for s in $(echo "$scopes_line" | tr ',' ' '); do
        if echo "$required_scopes" | grep -qxF "$s"; then
            required_agents="$required_agents dod-understanding-$id"
            break
        fi
    done
done
required_agents=$(echo "$required_agents" | tr ' ' '\n' | grep -v '^$' | sort -u)

if [ -z "$required_agents" ]; then
    exit 0
fi

invoked_agents=$(grep -oE '"subagent_type":"dod-understanding-[A-Za-z0-9_-]*"' "$transcript_path" 2>/dev/null \
    | sed -E 's/.*"subagent_type":"([^"]*)"/\1/' \
    | sort -u || true)

missing=$(comm -23 <(echo "$required_agents") <(echo "$invoked_agents") 2>/dev/null | grep -v '^$' || true)

if [ -z "$missing" ]; then
    exit 0
fi

missing_count=$(echo "$missing" | wc -l | tr -d ' ')
file_lines=$(echo "$deliverable_files" | sed 's/^/  - /')
missing_lines=$(echo "$missing" | sed 's/^/  - /')
scopes_line_display=$(echo "$required_scopes" | tr '\n' ' ')

cat >&2 <<EOF
[DoD review not done] The following deliverable(s) were written/edited this session:

${file_lines}

Required scopes for these files: ${scopes_line_display}

The following scope-relevant dod-understanding-* Agents have not been invoked yet (${missing_count}):

${missing_lines}

Before reporting this done, invoke each one individually via the Agent tool
(subagent_type: "dod-understanding-<id>") and confirm PASS/FAIL. This check
re-evaluates every time, so it stays blocked until the Agents actually show
up in the transcript.
EOF
exit 2
