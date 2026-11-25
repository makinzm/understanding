# Customer Journey Map: Document Factory

Persona: **a future contributor to this repository** — most realistically future-self, returning after weeks/months away, or an AI coding agent (Claude Code) picking up the repo cold. They want to add a new document (or a whole new topic) and have it meet this repo's quality bar without re-deriving the rules from scratch.

This map is backed by the actual state of the repo as of this writing (2026-11-25), not an aspirational description. It must be revisited whenever `dod/deliverables-dod.json`, `.claude/CLAUDE.md`, `README.md`, or `scripts/sync-dod-skills.py` change in a way that affects any stage below.

## Stage 1: Discover that this repo expects DoD-driven documents

- **Information needed:** That a quality checklist exists at all, before writing anything.
- **Reference:** Top-level `README.md` ("DoD" and "Document Factory" sections); `.claude/CLAUDE.md` / `AGENTS.md` (same file via symlink) "Document Factory" section.
- **Current state: Done.** Both are now in place and link to each other and to `dod/en.md`/`dod/ja.md`.

## Stage 2: Learn what DoD applies to *this* directory specifically

- **Information needed:** Which checklist items apply to the directory they're writing in — a contributor adding a dance document should not have to guess that ML-specific items (pseudocode, datasets) don't apply to them.
- **Reference:** `dod/en.md` / `dod/ja.md` (scope-tagged per section); each topic's own `README.md` has a "Document Factory" paragraph naming its own scope's items explicitly (`dance/README.md`, `chess/README.md`); `dod/deliverables-dod.json`'s `scopes` field is the ground truth if the mirrors ever drift.
- **Current state: Done.** `machine-learning/`, `dance/`, `chess/` each have explicit scoped items in their own `README.md`. `software-engineering/` and `foods/` now also have a "Document Factory" paragraph stating explicitly that they are `common`-scope only, rather than leaving a contributor to infer that from the absence of a section.

## Stage 3: Write the document

- **Information needed:** The expected document structure (Meta Information, Overview, Key Concepts, Comparison, References) and file-naming convention.
- **Reference:** The `create-topic` Skill's "Document Template" (`.claude/skills/create-topic/SKILL.md`); each topic `README.md`'s own "Document Template" section; `scripts/title-converter.sh` for filenames.
- **Current state: Done.** This existed before this change and was followed for the new `dance/genre/HipHop/`, `dance/genre/Locking/`, and `chess/` documents.

## Stage 4: Get the document reviewed

- **Information needed:** Which specific Agents to invoke, and that skipping this step will actually be caught, not just discouraged by a comment in a doc nobody reads.
- **Reference:** `.claude/agents/dod-understanding-<id>.md` (one per DoD item, `subagent_type: dod-understanding-<id>` via the Task/Agent tool); `scripts/dod-review-check.sh`, registered as a `Stop` hook in `.claude/settings.json`.
- **Current state: Partial, with an important caveat discovered while building this.** The hook's scope-aware logic was manually verified correct (synthetic transcripts) to require exactly the common+dance set for a `dance/` file, the common+chess set for a `chess/` file, and common-only for a `software-engineering/` file — never the full set of 14 Agents regardless of relevance. However, **both the Agent dispatch and the Stop hook itself only work from a Claude Code session whose project root is this repo** (`~/workspace/understanding`) — a session rooted elsewhere (as the one that built this factory was, rooted in a separate dotfiles repo) cannot see `dod-understanding-*` as an available `subagent_type` at all, and this repo's `.claude/settings.json` Stop hook never activates for it either. The documents added alongside this factory were reviewed by manual self-check against each item's exact check text instead (by the same session that wrote them, which is a weaker guarantee than an independent Agent's review). A future contributor must open/continue their session with this repo as the project root for Stage 4 to work as designed.
- Separately, enforcement is session-scoped (it only checks files written/edited in the *current* session's transcript) — it does not retroactively audit documents already in the repo, including the pre-existing `dance/general/*.md` and `dance/genre/KPOP/README.md`, which do not meet even the Common items today (they are informal link lists). That gap is accepted as out of scope for this change, not silently hidden.

## Stage 5: Ship

- **Information needed:** How to get the reviewed document merged.
- **Reference:** `create-pr` Skill; `.github/pull_request_template.md` (Objective/Effect/Test/Note + DoD checklist).
- **Current state: Done.** The PR template's "Definition of Done Checklist" now also lists Dance and Chess sections, so a contributor relying on the PR template alone (not `dod/en.md`) still sees the scoped items relevant to what they changed.

## Stage 6: Maintain

- **Information needed:** How to add a new DoD item or a new topic's scope later without breaking the 1:1 Skill/Agent invariant.
- **Reference:** `.claude/CLAUDE.md`'s "Document Factory" section (edit JSON → re-run `scripts/sync-dod-skills.py`, tests via `scripts/test-dod-sync.sh` / `make test` → update `dod/en.md`/`dod/ja.md` by hand).
- **Current state: Done.** The generator is test-covered (`scripts/test_sync_dod_skills.py`), and running it is idempotent and safe (it removes stale Skills/Agents for deleted items).

## Known gaps (tracked here, not hidden)

1. Pre-existing `dance/` content (`general/`, `genre/KPOP/`) does not meet even the Common DoD items (it predates this Document Factory and is an informal link list) and is not retroactively flagged by the new hook, since the hook is session-scoped rather than a full-repo audit.

This is left as follow-up work rather than silently absorbed into this change's scope.
