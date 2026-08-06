# v4.3 Stage C-D.4A — Schema Repair and Runner Control

This patch remains domain-neutral and preserves the Stage C-D.1 inventory
checkpoints.

## Changes

- Introduces an independent extraction prompt version.
- Bypasses extraction for nonvisual inventory items with
  `formula_expected=false`.
- Restricts extraction dispositions to those valid for the inventory item.
- Requires one variable definition for every ASCII identifier, including the
  left-hand result, with no extras.
- Performs one source-bounded extraction repair after deterministic validation
  failure.
- Supplies immutable AST node templates to entailment.
- Performs one entailment repair only when immutable node expressions or
  operations are altered.
- Preserves nonfatal rejection for structurally invalid entailment responses;
  these do not trigger a second model call.
- Allows redundant exact grounding for identifiers that exist in the complete
  candidate formula.
- Keeps the deterministic validators strict.

## Deliberately deferred

- Evidence-backed inventory audit and span expansion.
- Autonomous visual frame recovery.
- Narrative synthesis and production-runner integration.
