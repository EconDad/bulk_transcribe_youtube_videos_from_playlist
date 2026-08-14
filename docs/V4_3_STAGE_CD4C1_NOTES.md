# v4.3 Stage C-D.4C.1 — Calculation-Scoped Formula Identity

Stage 4C is frozen as accepted. This milestone changes only coverage-level
formula identity.

Formula IDs are semantic names local to one calculation event. The globally
unique formula key is `(calculation_id, formula_id)`.

This allows numerical examples and a separately grounded general expression
to use the same semantic formula ID without colliding. A duplicate remains
invalid when the same calculation contains the same formula ID more than once.

Coverage lookup, unreferenced-formula validation, and retained-formula counts
all use scoped keys. Formula expressions, formula IDs, parsed identifiers,
inventory, extraction, entailment, and source evidence are not rewritten.

Canonical/general-example classification remains out of scope for this
correctness milestone.

Revision 2 preserves the legacy wrong-owner diagnostic for a scoped formula lookup miss when exactly one other calculation owns that formula_id. The actual identity remains `(calculation_id, formula_id)`.
