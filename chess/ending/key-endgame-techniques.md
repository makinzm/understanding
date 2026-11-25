# Meta Information

- LICENSE: N/A (general chess strategy knowledge, not quoted from a single copyrighted source)
- Reference: Standard endgame theory as consolidated across widely-used references such as Basic Chess Endings (Fine) and Dvoretsky's Endgame Manual; this document names and conceptually describes well-known named endgame positions/patterns rather than giving move-by-move solutions.

# Key Endgame Techniques

## Overview

Beyond the general principles in `endgame-principles.md`, certain specific endgame positions and mating patterns are well-known enough to have their own names, because they recur constantly and have a known correct result (win or draw) with correct play. This document describes what each one is and why it matters conceptually, without giving the exact winning move sequence (that is calculation practice, not strategic understanding).

## Key Concepts

### Lucena position

**Category:** Rook-and-pawn vs. rook endgame technique (winning technique for the side with the extra pawn).

**Why it matters:** The Lucena position is a specific rook-and-pawn configuration (the pawn is one step from promotion, cut off from the opposing king by the defending rook) that is a known win for the stronger side, using a technique nicknamed "building a bridge" — the attacking rook shields the king's advance to the promotion square from the defending rook's checks. Recognizing the Lucena position tells a player the position is winnable even before calculating the exact moves.

### Philidor position

**Category:** Rook-and-pawn vs. rook endgame technique (defensive drawing technique for the side without the extra pawn).

**Why it matters:** The Philidor position is a specific defensive setup where the defending rook holds a particular rank in front of the advancing pawn, preventing the attacking king from making progress and holding a draw despite being down a pawn. It is the defensive counterpart to the Lucena position — recognizing it tells the defending player the position is a known draw if the correct rank is held.

### Basic mate: King + Queen vs. King

**Category:** Basic forced-mate pattern (overwhelming material advantage).

**Why it matters:** This is the simplest material-advantage mate to learn: the queen restricts the lone king's available squares while the attacking king approaches to assist, since the queen alone cannot force mate against a king that has the whole board to flee into. Learning this pattern establishes the general technique (restrict the king's box, then bring your own king closer) reused in more complex mates.

### Basic mate: King + Rook vs. King

**Category:** Basic forced-mate pattern (overwhelming material advantage).

**Why it matters:** Similar in spirit to King+Queen vs. King but slower, because a rook covers less of the board at once than a queen and so needs more moves (and more precise king coordination) to force the lone king to the edge and then mate it. This pattern is the standard introduction to "driving the king to the edge," a technique reused in many other endgames.

### Fortress

**Category:** General defensive endgame concept (applies across many material imbalances, not one specific position).

**Why it matters:** A fortress is a defensive setup where the weaker side's king and remaining pieces occupy squares that the stronger side cannot break into or make progress against, resulting in a draw despite being materially behind. Recognizing a fortress tells the defending player that passive, unmoving defense — rather than trying to trade material down further — is the correct plan.

## Comparison with Related Concepts

- **Lucena vs. Philidor**: These are the two reference positions for the same rook-and-pawn endgame, one from the winning side's perspective (Lucena) and one from the defending side's (Philidor) — a player should recognize both, since which one applies depends on whether they are ahead or behind the pawn.
- **King+Queen vs. King vs. King+Rook vs. King**: Both are "drive the lone king to the edge" patterns; the queen version is faster/more forcing because the queen covers more squares per move, which is why it is usually taught first.
- **Fortress vs. Lucena/Philidor**: Lucena and Philidor are specific piece-and-pawn configurations with known, calculable results; a fortress is a broader pattern-recognition concept that can apply to many different material combinations, not a single named position.

## Applicability

These techniques apply to a player who already understands general endgame principles (king activity, opposition, zugzwang) and is converting (or defending) a specific, common material imbalance — most often a single extra pawn in a rook endgame, or an overwhelming material advantage against a lone king.

## References

- General endgame theory as consolidated in standard references (e.g. Basic Chess Endings, Dvoretsky's Endgame Manual); no single source is quoted verbatim.
