# v4.3 Stage C-D.4B.2.2 — Matcher Hardening

This milestone hardens deterministic source matching before further acceptance.

- Numeric inventory variables require exact normalized numeric literals.
- Percent values preserve the percent distinction.
- Operation cues require lexical boundaries; short cues no longer match inside unrelated words.
- Generic `profit`, `gain`, and `gained` are recognized as subtraction/difference cues alongside loss vocabulary.
- Automatic deterministic expansion can use evidence no farther than three transcript segments from the original event.
- More distant evidence falls back to the bounded model audit rather than being stitched automatically.
- Audit action strings receive conservative edit-distance normalization only when the intended allowed action is unique and at most two edits away.
- Audit, entailment, and package versions advance so corrected matcher behavior invalidates stale checkpoints.

No subject-specific equations, variables, or aliases are added.
