# v4.3 Stage C-D.4B.3.1 — Span-Grounded Reconciliation + Fail-Closed Audit

This milestone addresses two real-video defects observed after Stage C-D.4B.3.

## Span-grounded reconciliation

Model-selected evidence IDs now bound one contiguous candidate source span
together with the original inventory item. Revised variables and operations
are validated against every transcript segment inside that span rather than
only against the explicitly enumerated IDs.

The audit prompt and its one repair prompt now state that evidence IDs may be
used as outer endpoints when support for the same calculation event is
distributed across nearby segments. The bounded neighborhood and same-event
constraints remain unchanged.

## Fail-closed audit routing

If both the primary inventory audit and its single bounded repair fail
validation, the item remains formula-expected but is routed directly to
`insufficient_source_detail`. Formula extraction and entailment are skipped.

This prevents an uncorrected inventory item from producing a retained formula
after its own source-grounding audit has failed.

Raw inventory checkpoints remain reusable. Evidence-audit and entailment
versions are advanced. Changed reconciled item hashes continue to invalidate
formula extraction automatically.

No subject-specific equations, aliases, or domain rules are introduced.
