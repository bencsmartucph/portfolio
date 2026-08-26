# chess-coach — personal chess coaching pipeline

A pipeline that turns my [chess.com](https://chess.com) games into interactive,
engine-verified coaching reviews. Each game gets a narrative explanation of what
went wrong (and what went right), move-by-move engine analysis, and a
drag-and-try board for exploring alternatives.

→ **[Open the sample review](sample-review.html)** (self-contained HTML — download
  and open locally, or view on GitHub Pages)

## What it does

1. **Ingest** — pulls games from the chess.com public API, deduplicates, stores
   in SQLite.
2. **Moves** — parses PGN into a per-ply move table with clock times and
   metadata.
3. **Engine pass** — Stockfish evaluates every position at depth 12, producing
   centipawn-loss and best-move data.
4. **Review** — an LLM reads the engine data and writes a coaching narrative:
   identifies the 3–5 key moments, explains *why* each move was wrong in plain
   language, names the better plan, and validates its own claims against the
   engine lines.
5. **Web** — renders each review as a self-contained HTML page with an
   interactive board, eval graph, move-quality badges, and coach bubbles
   modeled on chess.com's Game Review UI.

## What the review includes

- **Narrative** — a plain-English story of the game: what the opening set up,
  where it went wrong, and what the engine thread was.
- **Key moments** — each mistake gets: the engine's best move, a punish line
  showing how the opponent exploits the error, arrow/square overlays on the
  board, and a one-sentence coaching cue.
- **Best find** — the one move where the player matched the engine's uniquely
  best option.
- **Pattern check** — cross-references each game against a persistent claims
  table (e.g., "missed-tactic rate is above peer baseline") to track whether
  weaknesses are real trends or noise.
- **Validation stamp** — every factual claim in the coach prose is
  machine-checked against the Stockfish evaluations; flagged if it can't be
  verified.

## Stack

Python 3.14 · SQLite · Stockfish 17 · Claude (coaching narrative + validation) ·
vanilla JS/CSS (no framework, no build step — each review is one `.html` file).

## Skills demonstrated

- Data pipeline design (incremental ingest → engine → LLM → render)
- LLM-as-judge pattern (the coach writes prose, then a validation pass
  checks every claim against engine data)
- Self-contained frontend with no build tooling
- Domain modeling (claims table, pattern tracking across games)

---

Built with Claude as a pair-programmer; the pipeline design, domain modeling,
and chess judgment are mine.
