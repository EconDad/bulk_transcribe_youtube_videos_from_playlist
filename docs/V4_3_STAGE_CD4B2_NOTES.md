# v4.3 Stage C-D.4B.2 — Bounded Evidence Audit

This milestone adds a selective, source-grounded audit between merged inventory
discovery and formula extraction.

- Raw calculation-inventory model checkpoints remain valid.
- The deterministic visual-equation audit from 4B.1 remains first.
- Only formula-expected, nonvisual items whose current span fails deterministic
  variable/operation evidence checks are sent to the evidence-audit model.
- Each selected item receives a bounded ±8 segment neighborhood.
- The audit can keep the range, minimally expand it, or downgrade a result-only
  observation to `formula_expected=false`.
- Every model evidence quote is deterministically checked against the cited
  transcript range.
- One bounded repair is allowed for malformed audit output.
- A failed repair preserves the original item and records `audit_failed`.
- Decisions are appended to `inventory_audit.json`.
- Changed item content invalidates extraction automatically through item SHA.
- Entailment is version-invalidated because subtraction evidence now recognizes
  generic `loss`, `lose`, and `lost` language.

No subject-specific equations, variables, or aliases are included.
