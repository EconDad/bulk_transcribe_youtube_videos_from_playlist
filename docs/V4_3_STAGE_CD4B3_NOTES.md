# v4.3 Stage C-D.4B.3 — Inventory Claim Reconciliation

Stage C-D.4B.2.2 hardened deterministic matching. This milestone addresses the
remaining case where the raw inventory claim itself is inaccurate or too narrow.

## Contract

After deterministic expansion fails, the bounded audit may now choose:

- `reconcile`: preserve the same calculation event while correcting its
  source-grounded variables, canonical operations, and/or bounded span;
- `downgrade_non_symbolic`: retain the event as quantitative context without
  requiring a symbolic formula.

The deterministic `expand` action remains internal and legacy-compatible.

For reconciliation, the model supplies only segment IDs, revised variable
claims, revised canonical operation claims, and a reason. It does not supply a
formula, quotes, or start/end ranges. Python:

1. validates every evidence segment against the bounded neighborhood;
2. copies canonical transcript text itself;
3. validates every revised variable against selected source evidence;
4. validates every revised operation against the shared operation-cue map;
5. computes the final contiguous span;
6. rejects no-op or ungrounded reconciliations;
7. changes the `CalculationItem`, causing downstream extraction checkpoints to
   invalidate automatically by item SHA.

The prompt explicitly forbids replacing an event with a different neighboring
example or arithmetic sub-step and prefers reusable source noun phrases over
numeric literals where the transcript supplies them.

No subject-specific formulas, aliases, variables, or recovery rules are added.
