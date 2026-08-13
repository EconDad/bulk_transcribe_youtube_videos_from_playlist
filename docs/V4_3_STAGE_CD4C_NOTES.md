# v4.3 Stage C-D.4C — Deterministic Extraction Schema Normalization

Stage C-D.4C freezes the accepted 4B.3.1 inventory/audit architecture and
changes only the formula-extraction validation layer.

Before strict `FormulaCandidate` validation, Python safely parses the existing
ASCII expression and derives its exact identifier set. Variable definitions
whose symbols do not occur in the parsed expression are removed. This includes
conceptual aliases for quantities represented as numeric literals.

The ASCII expression, LaTeX, derivation type, source claims, and derivation
steps are never rewritten.

Identical duplicate definitions for a required identifier are deduplicated.
Conflicting duplicate definitions remain an error and use the existing single
bounded repair attempt. Missing required AST identifiers are never synthesized.

The extraction prompt itself is unchanged, so successful raw extraction
checkpoints remain reusable and are reparsed under the corrected validator.
Only the package version advances to Stage C-D.4C.

No inventory, evidence-audit, entailment, visual-recovery, or subject-specific
logic is modified.
