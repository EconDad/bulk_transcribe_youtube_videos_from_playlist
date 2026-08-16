# Research v4.3 Stage F — Finalization and Narrative Integration

Baseline: `a7b42363d290f01ca5b86aaaea8c79138d731145` (`research-v4.3-stage-e3`)

## Purpose

Stage F turns a verified v4.3 calculation/formula diagnostic package into the
reader-facing per-video research package. It does not weaken the Stage A-E
formula gates and does not reuse the legacy v4.1 formula-recovery code.

The implementation remains domain-neutral. Narrative synthesis may consume
only transcript-grounded narrative evidence plus formulas already retained by
the v4.3 AST/entailment/visual pipeline.

## F.1 scope

1. Extract narrative evidence from bounded transcript chunks.
2. Validate every narrative evidence item deterministically:
   - citation range exists and is narrow;
   - numbers and dates are present in the cited source range;
   - prose is English reader-facing text;
   - prompt/debug/schema leakage is rejected.
3. Synthesize a four-sentence executive summary, distinct takeaways, and
   transcript-ordered sections with `qwen3:8b`, `think=false`.
4. Permit synthesis to reference retained formulas as validated evidence, but
   never expose machine identifiers in reader-facing prose.
5. Build the final `Processed Research/<video_id>/` package atomically with:
   - `_READY` written last;
   - `metadata.json`;
   - `research.json`;
   - `research.md`;
   - `formulas.json`;
   - `source_map.json`;
   - `calculation_inventory.json`;
   - `formula_entailment.json`;
   - `formula_coverage.json`;
   - `rejected_formulas.json`;
   - `visual_evidence.json`.
6. Verify artifact hashes, package digest, source SHA, build version, citation
   references, formula coverage, and stated-visual provenance before a package
   is accepted.

## Fail-closed rule

F.1 does not write a final `research_ready` package when the upstream v4.3
formula coverage report has `passed=false`. A diagnostic run may still exist,
but it is not promoted to `Processed Research`.

This preserves the Stage E acceptance while making remaining coverage issues
explicit. Numerical examples and other unresolved inventory classifications
will be addressed generically in the next acceptance repair rather than by
lowering formula or visual validation standards.

## Narrative evidence contract

Each evidence item contains:

```json
{
  "evidence_id": "N0001",
  "topic": "Reader-facing topic",
  "text": "A complete source-grounded sentence.",
  "explanation": "A complete source-grounded sentence explaining why it matters.",
  "start_segment": 10,
  "end_segment": 12
}
```

Formula evidence is generated only from retained formula records and is marked
separately so the synthesis model cannot turn rejected candidates into prose.

## Final acceptance for F.1

Unit gate:

- all existing tests pass;
- new narrative and package tests pass;
- stale source SHA is rejected;
- bad package hash is rejected;
- unsupported narrative numbers are rejected;
- out-of-order sections are rejected;
- stated-visual formula without accepted visual provenance is rejected;
- incomplete formula coverage blocks finalization.

Local integration gate:

- run the conceptual QC video and confirm no invented formula enters the final
  package;
- run the long narrative QC video and confirm ordered synthesis and no package
  crash;
- retain the YTM Stage E diagnostic as a regression oracle until its remaining
  coverage items are resolved generically.

Do not run the full playlist batch in F.1.
