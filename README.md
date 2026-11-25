# understanding

This repository describe what I understand about various topics in computer science, programming, software development and machine learning and so on.

## Topics

- [`machine-learning/`](machine-learning/README.md) — ML papers and concepts
- [`software-engineering/`](software-engineering/README.md) — Software engineering topics
- [`foods/`](foods/README.md) — Food culture
- [`dance/`](dance/README.md) — Street dance fundamentals, by genre
- [`chess/`](chess/README.md) — Chess opening and endgame strategy (no piece-movement rules)

## DoD

- [Definition of Done](dod/en.md)
- [Definition of Done (日本語)](dod/ja.md)

## Document Factory

This repository is a "Document Factory": [`dod/deliverables-dod.json`](dod/deliverables-dod.json) is the source of truth for DoD checklist items, each tagged with which directories it applies to (a `common` item applies everywhere; a topic-specific item, e.g. for `machine-learning/`, `dance/`, or `chess/`, applies only there, since each directory's quality bar genuinely differs). `scripts/sync-dod-skills.py` materializes each item into one Skill and one review Agent, and `scripts/dod-review-check.sh` enforces — scope-aware — that the relevant Agents actually get run against a document before it is reported done. See `.claude/CLAUDE.md` for the full mechanism.
