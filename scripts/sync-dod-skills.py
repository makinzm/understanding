#!/usr/bin/env python3
"""Materialize one Skill AND one Task-tool review Agent per DoD item from
dod/deliverables-dod.json.

Adapted from makinzm/dot-claude's own DoD Factory generator
(scripts/sync-dod-skills.py there), but scope-aware: each item carries a
`scopes` list (e.g. ["common"], ["dance"], ["chess"]) saying which
top-level topic directories it applies to. That list is rendered into the
generated Skill/Agent so a reviewer can see applicability without reading
the JSON manifest, and dod-review-check.sh uses it to only demand the
Agents relevant to the directory actually touched (never, e.g., an
ML-pseudocode check on a dance document).

Ids are prefixed dod-understanding-<id> (not bare dod-<id>) so they can
never collide with makinzm/dot-claude's own global dod-* ids, which also
run on this machine via its machine-wide Stop hook.

No more and no fewer: the set of dod-understanding-*-prefixed skill
directories under skills/, and the set of dod-understanding-*.md agent
files under agents/, are each kept in exact sync with the "items" list in
deliverables-dod.json. Re-run this script after editing that JSON.
"""

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOD_JSON = REPO_ROOT / "dod" / "deliverables-dod.json"
DEFAULT_SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
DEFAULT_AGENTS_ROOT = REPO_ROOT / ".claude" / "agents"

DOD_PREFIX = "dod-understanding-"

SKILL_TEMPLATE = """---
name: dod-understanding-{id}
description: {description}
---

# DoD item: {title}

{check}

**Scopes:** {scopes_str}

## Usage

Before reporting a document in one of the scopes above as done, confirm it satisfies this
check. If it does not, fix it before reporting completion.
"""

AGENT_TEMPLATE = """---
name: dod-understanding-{id}
description: "{description}"
model: sonnet
color: yellow
---

You are the **DoD review Agent for "{title}"**. You check ONLY whether a document
satisfies this single DoD item -- nothing else.

## Most important rule

> **You review this one item only; do not comment on other aspects or style.**
> Do not fix anything yourself. Report PASS/FAIL and, if FAIL, a specific reason.
> Leave the fix to whoever asked for the review.

## Scopes

This item applies to: {scopes_str}. If the document under review is outside these scopes,
say so and return PASS without further comment (this check does not apply).

## Item to check

{check}

## Output format

- State `PASS` or `FAIL` explicitly.
- If `FAIL`, point to the specific part of the document that fails this item and why.
"""


def load_items(dod_json: Path) -> list[dict]:
    with dod_json.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def _scopes_str(item: dict) -> str:
    return ", ".join(item.get("scopes", [])) or "common"


def sync(items: list[dict], skills_root: Path) -> tuple[list[str], list[str]]:
    skills_root.mkdir(parents=True, exist_ok=True)

    wanted_ids = {item["id"] for item in items}
    existing_dod_ids = {
        p.name[len(DOD_PREFIX) :]
        for p in skills_root.iterdir()
        if p.is_dir() and p.name.startswith(DOD_PREFIX)
    }

    removed = sorted(existing_dod_ids - wanted_ids)
    for stale_id in removed:
        shutil.rmtree(skills_root / f"{DOD_PREFIX}{stale_id}")

    written = []
    for item in items:
        skill_dir = skills_root / f"{DOD_PREFIX}{item['id']}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        description = (
            f"Check a document against the DoD item '{item['title']}' "
            f"(scopes: {_scopes_str(item)})."
        )
        content = SKILL_TEMPLATE.format(
            id=item["id"],
            title=item["title"],
            check=item["check"],
            description=description,
            scopes_str=_scopes_str(item),
        )
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        written.append(item["id"])

    return written, removed


def sync_agents(items: list[dict], agents_root: Path) -> tuple[list[str], list[str]]:
    agents_root.mkdir(parents=True, exist_ok=True)

    wanted_ids = {item["id"] for item in items}
    existing_dod_ids = {
        p.name[len(DOD_PREFIX) : -len(".md")]
        for p in agents_root.iterdir()
        if p.is_file() and p.name.startswith(DOD_PREFIX) and p.name.endswith(".md")
    }

    removed = sorted(existing_dod_ids - wanted_ids)
    for stale_id in removed:
        (agents_root / f"{DOD_PREFIX}{stale_id}.md").unlink()

    written = []
    for item in items:
        description = (
            f"DoD review Agent for '{item['title']}' "
            f"(scopes: {_scopes_str(item)}). Invoke individually when checking a document."
        )
        content = AGENT_TEMPLATE.format(
            id=item["id"],
            title=item["title"],
            check=item["check"],
            description=description,
            scopes_str=_scopes_str(item),
        )
        (agents_root / f"{DOD_PREFIX}{item['id']}.md").write_text(content, encoding="utf-8")
        written.append(item["id"])

    return written, removed


def main() -> int:
    items = load_items(DEFAULT_DOD_JSON)
    written, removed = sync(items, DEFAULT_SKILLS_ROOT)
    agents_written, agents_removed = sync_agents(items, DEFAULT_AGENTS_ROOT)
    print(f"DoD items: {len(items)}")
    print(f"Skills written: {len(written)} -> {', '.join(written)}")
    if removed:
        print(f"Skills removed (no longer in DoD list): {len(removed)} -> {', '.join(removed)}")
    print(f"Agents written: {len(agents_written)} -> {', '.join(agents_written)}")
    if agents_removed:
        print(
            f"Agents removed (no longer in DoD list): {len(agents_removed)} -> "
            f"{', '.join(agents_removed)}"
        )
    if len(written) != len(items) or len(agents_written) != len(items):
        print("ERROR: generated skill/agent count does not match DoD item count", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
