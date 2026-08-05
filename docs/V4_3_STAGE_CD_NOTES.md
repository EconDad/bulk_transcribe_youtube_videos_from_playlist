# v4.3 Stage C–D Integration

This milestone implements the isolated calculation-inventory and formula-
entailment path without modifying `run_research_analysis.py`.

## Added

- JSON-only Ollama model client with deterministic settings
- Per-calculation formula extraction
- Shared AST validation for every proposed formula
- Expression-node entailment with exact source quotes
- Dependency support for multi-operator formulas
- Domain-neutral arithmetic cue validation
- Formula-coverage reconciliation
- Atomic diagnostic packages with `_READY` written last
- Package hash, source-SHA, and prompt-version verification
- Isolated `run_research_v43.py` runner

## Diagnostic package

```text
Research v43 Diagnostics/<video_id>/
├── _READY
├── metadata.json
├── calculation_inventory.json
├── formulas.json
├── formula_entailment.json
├── formula_coverage.json
├── rejected_formulas.json
└── model_invocations.json
```

## Intentional limitations

- Visual frame extraction and equation reading are deferred to Stage E.
- A visual-only calculation is persisted as `visual_review_required`.
- This milestone does not write the final narrative research package.
- This milestone does not replace the working v4.1.1 production runner.

## Production-code constraints

The implementation contains no hard-coded subject formulas, finance-specific
recovery rules, or subject-specific alias tables. Deterministic logic is
limited to expression grammar, generic arithmetic relationships, source ranges,
exact quote validation, coverage, and package integrity.
