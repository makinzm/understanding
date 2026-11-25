## Definition of Done

I define "done" as having a clear and concise explanation of the topic that can be easily understood by someone with a basic knowledge of the subject.

**Source of truth**: this checklist is a human-readable mirror of [`dod/deliverables-dod.json`](deliverables-dod.json), which is machine-generated into one Skill + one Agent per item (`.claude/skills/dod-understanding-<id>/`, `.claude/agents/dod-understanding-<id>.md`) via `scripts/sync-dod-skills.py`. Edit the JSON, then re-run that script; update this file to match by hand.

Each item below applies only to the directory scopes shown in parentheses. "Common" items apply everywhere; directory-specific items apply only within that top-level directory (e.g. Machine Learning items do not apply to `dance/` or `chess/`, and vice versa). Directories without a dedicated section yet (`software-engineering/`, `foods/`) are covered by Common only.

### Table of Contents

- [0. Common](#0-common)
- [1. Machine Learning](#1-computer-science--machine-learning)
- [2. Dance](#2-dance)
- [3. Chess](#3-chess)


### 0. Common (scope: common)
- [x] Describe the concrete sentences to support understanding ( not just writing "I understand ..." )
- [x] Describe the condition which can be applied ( who, when, where )
- [x] Include information about licenses and copyrights

### 1. Computer Science / Machine Learning (scope: machine-learning)

- [x] Clear Input and Output
- [x] Describe Algorithms with pseudocode
- [x] Explain datasets used
- [x] Clear calculation order
- [x] Describe the difference between similar algorithms

### 2. Dance (scope: dance)

- [x] State each move's provenance (origin/originator and era when sources establish one), and explicitly flag when sources disagree or are too thin to confirm a detail
- [x] Describe concrete body mechanics (which joints/body parts move, on which beat/count), not vague language like "feel the groove"
- [x] State which dance style/family each move belongs to (e.g. Hip-Hop, Locking, Popping, Breaking)

### 3. Chess (scope: chess)

- [x] Do not explain how individual pieces move (assumed prior knowledge) — focus on strategic principles and named concepts
- [x] Explain why each principle, opening family, or endgame technique works, not only its name
- [x] State which broader category each named opening family or endgame technique belongs to (e.g. open vs. closed game; king-side vs. queen-side endgame)
