# v4.3 Stage C-D.4B.1 — Lexical and Visual Routing Audit

This milestone is deterministic and domain-neutral.

- Adds `dividing` to division cue recognition.
- Detects direct transcript announcements of displayed equations/formulas.
- Audits the merged calculation inventory while preserving raw inventory model
  checkpoints.
- Routes visual review using `visual_equation_cue`.
- Persists `inventory_audit.json` as an optional diagnostic artifact.
- Updates the atomic artifact writer so optional diagnostic artifacts are
  written, hashed, and covered by package integrity metadata.
- Leaves the extraction prompt/checkpoint version unchanged.
- Bumps entailment validation/checkpoint version for the corrected lexical cue.
- Defers bounded-neighborhood evidence/span auditing to Stage C-D.4B.2.
