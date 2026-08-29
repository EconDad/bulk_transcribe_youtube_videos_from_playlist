# Research v4.3 Gate 3 Pilot Results

Run date: 2026-08-28
Branch: `feature/research-v4.3`
Execution: sequential, local inference concurrency `1`

## Result

Gate 3 passed its operational acceptance criteria:

- five additional usable videos were processed sequentially;
- every video reached an allowed persisted terminal state;
- no pipeline run ended in an uncaught exception;
- no video was silently skipped;
- diagnostic coverage and final-package QC artifacts were persisted.

Four videos reached `research_ready`. One video correctly failed closed as
`analysis_failed` because formula coverage remained unresolved after the
bounded recovery attempts.

## Per-video outcomes

| Video | Terminal state | Calculations | Formulas retained | Non-symbolic | Unresolved | Package SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `vJASGyPn-vg` — Value a Small Business | `research_ready` | 12 | 3 | 9 | 0 | `2e9bd3353ae57d75b5986b7df670858c3bea7586ba6e47a6718e5635cac46ddb` |
| `yfWbSbrKkcQ` — Balance Sheet and Margin of Safety | `research_ready` | 9 | 6 | 3 | 0 | `c9f6ccfc1de5cefdf2acd0a9d0f8cf21a11f31b4bd58bea62d5b5090f52b67df` |
| `csy91a1iDKU` — What is a Share | `analysis_failed` | 12 | 9 | 0 | 3 | — |
| `PUvkPoDavnI` — Finding Basic Stock Terms | `research_ready` | 7 | 6 | 1 | 0 | `6c52e3cc24fbb86ad042e0ca69d6de6226a88d580a7c26bfe7fc4c4ab04f7964` |
| `_uQjGz6jp2E` — Warren Buffett Stock Basics | `research_ready` | 12 | 7 | 5 | 0 | `b06997ebed3f22c505ca4db52a77bf02a4d752e506fc38ac7589cfb6717d52b8` |

## Fail-closed record

`csy91a1iDKU` retained nine validated formulas but was not promoted because
three calculation events remained unresolved:

- `CALC_0001`: insufficient source detail after the bounded inventory audit;
- `CALC_0007`: proposed candidates did not pass expression-node entailment;
- `CALC_0012`: extraction remained schema-invalid after one repair.

The manifest records `analysis_failed` and points to the preserved diagnostic
package at `Research v43 Diagnostics/csy91a1iDKU`.

## Notable recovery coverage

- Adaptive inventory output-budget retries completed successfully.
- Empty extraction responses were retried without aborting a video.
- Broad narrative citations were localized or rejected item-by-item.
- Synthesis evidence references were pruned deterministically when needed.
- One visual task produced no parser-valid frame equation; the invalid visual
  candidates were rejected and a separately grounded transcript fallback
  passed AST-node entailment.

## Next gate

Gate 4 is the full sequential batch. The failed pilot diagnostic should remain
as a regression fixture; Gate 4 does not require weakening the fail-closed
coverage rules.
