# Research v4.3 Gate 4 Batch Results

Run date: 2026-08-28
Branch: `feature/research-v4.3`
Execution: sequential, local inference concurrency `1`

## Result

Gate 4 completed across the full usable transcript corpus.

- 14 usable transcripts have persisted terminal research states.
- 12 videos are `research_ready` with verified Stage F packages.
- 2 videos are `analysis_failed` with preserved diagnostic packages and exact
  unresolved-coverage reasons.
- 0 videos are missing a research-manifest record.
- 0 videos remain queued or analyzing.
- 0 videos were silently skipped.
- The one transcription failure in the video manifest was not an analysis
  candidate and was not included in the usable-corpus denominator.

## Gate 4 run outcomes

| Video | Terminal state | Calculations | Formulas retained | Non-symbolic | Unresolved | Package SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `As1a2VgbdWg` — Analyze a Balance Sheet | `research_ready` | 5 | 1 | 4 | 0 | `031ab55b0e0621382c2ead6739dd9f991c99188c2f07e62c6bf8f2d8d6d31d4c` |
| `cWzgk-8QHKk` — What is a Bond | `analysis_failed` | 10 | 1 | 6 | 3 | — |
| `KQ2bfwHMrnM` — Components of a Bond | `research_ready` | 3 | 1 | 2 | 0 | `32a3ad082d320bb522420b8d0190c9ef0dcef3cb6de4fa16dc0d01b7726e9c72` |
| `v6zhQuGCfoM` — What is the Stock Market | `research_ready` | 0 | 0 | 0 | 0 | `8e86e2d5c20357d0b3a5eebe370c2f4b9740c4f2a320b764a0fb556a953c32b9` |
| `Edx5UR4yFOo` — Stock Market Crash and Bubbles | `research_ready` | 8 | 0 | 8 | 0 | `a6eb98aee00a674d615748e29e022af3e126b0233716471f92e1002eec7c965e` |
| `2QGqXeDOkIU` — What is the FED | `research_ready` | 5 | 1 | 4 | 0 | `f77f694a9381b7c85f0d2b1e2ea37d9f0e56dc6f067f3dc240278eb77e62d8e2` |

## Fail-closed record

`cWzgk-8QHKk` was not promoted because `CALC_0005`, `CALC_0009`, and
`CALC_0010` remained `insufficient_source_detail` after one bounded inventory
repair. The manifest records `analysis_failed` and points to
`Research v43 Diagnostics/cWzgk-8QHKk`.

Together with the Gate 3 failure for `csy91a1iDKU`, this leaves two explicit
regression fixtures for future generic recovery work. Neither failure was
allowed to fall back to an older package.

## Full-corpus verification

Every one of the 12 `research_ready` packages passed current source SHA,
prompt-version, required-artifact, artifact-hash, package-digest, formula
coverage, citation, and visual-provenance verification.

Notable batch behavior included:

- bounded adaptive retries for dense inventory chunks;
- safe zero-calculation narrative finalization;
- item-level rejection of unsupported narrative numbers and broad citations;
- rejection of seven no-equation visual frames followed by independently
  validated transcript fallback; and
- persisted `analysis_failed` status when coverage could not be resolved.

## Release status

Gates 1 through 4 are complete. The two fail-closed regression diagnostics were
reviewed and replayed on 2026-08-29. `csy91a1iDKU` recovered completely and was
finalized, bringing the usable-corpus state to 13 `research_ready` and one
`analysis_failed`. `cWzgk-8QHKk` now has only one unresolved item:
`CALC_0005`, intentionally retained as `insufficient_source_detail` because the
source does not state a unit-consistent reusable formula.

The remaining release work is to review and commit the accepted implementation
and result records, then tag `research-v4.3` when approved.
