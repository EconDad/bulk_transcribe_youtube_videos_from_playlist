# v4.3 Stage C-D.4B.2.1 — Deterministic-First Evidence Audit

This revision repairs the real-video interface failure observed in Stage C-D.4B.2.

- Python first searches the bounded neighborhood for every missing inventory
  variable and recognized operation cue.
- Variable matching includes small domain-neutral singular/plural normalization.
- When all claims are found, Python computes the minimal contiguous expansion
  without a model call.
- Remaining cases use a reduced model schema containing only calculation ID,
  action, evidence segment IDs, and reason.
- The model no longer supplies transcript quotes, evidence-kind labels, or
  start/end ranges.
- Python copies exact source text and computes the final range.
- Model expansion is accepted only if the resulting range deterministically
  grounds the existing inventory variables and operation cues.
- Otherwise one repair is allowed, with downgrade as the supported terminal
  path when a reusable symbolic relationship is not source-grounded.
- Entailment checkpoints are version-invalidated because audited spans may
  change even when a formula candidate is textually identical.

No subject-specific equations, variables, or aliases are introduced.
