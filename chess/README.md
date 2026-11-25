# Chess

Documentation of chess strategic fundamentals: opening principles and endgame technique. This directory assumes the reader already knows how the pieces move and does not re-explain that — it focuses on strategy (why a move/plan is good), not rules.

## Directory Structure

Organized by game phase, since opening and endgame strategy are governed by largely different principles (development/king safety vs. king activity/pawn promotion).

```
chess/
├── opening/
│   ├── opening-principles.md          # control center, development, king safety — and why each works
│   └── opening-families-overview.md   # categorization of named opening families, not move-by-move theory
└── ending/
    ├── endgame-principles.md          # king activity, opposition, zugzwang, the rule of the square
    └── key-endgame-techniques.md      # Lucena/Philidor positions, basic mating patterns by name
```

### Document Template

Each document in this directory should follow this structure:

- **Meta Information**: Source URL, LICENSE, and Reference for any quoted material, at the top
- **Overview**: What the concept/category is and who would use it
- **Key Concepts**: Each principle/technique explained with *why* it works, not just its name, and which broader category it belongs to
- **Comparison**: How it differs from similar principles/techniques
- **References**: Links to the sources used

### File Naming

- Use kebab-case for all filenames (e.g., `opening-principles.md`)
- Generate filenames with: `echo "Title" | bash scripts/title-converter.sh`

## Document Factory

This directory is covered by the `chess` scope in this repository's [Document Factory](../dod/deliverables-dod.json): the `common` DoD items apply here, plus `chess-no-piece-movement-rules` (no document in this directory may explain how a piece moves), `chess-principle-reasoning`, and `chess-concept-category`. See the top-level README's "Document Factory" section for how this is enforced.
