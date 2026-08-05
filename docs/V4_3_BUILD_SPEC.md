# Phase 3 Research Pipeline — v4.3 Build Specification

Status: **Authoritative next build**
Supersedes: v4.2 as a production/batch target
Model routing baseline:

- Extraction and reasoning: `qwen3:8b`, `think=true`
- Narrative synthesis: `qwen3:8b`, `think=false`
- Formula and citation validators remain deterministic
- Local inference concurrency: `1`

## 1. Purpose

v4.3 must provide a general-purpose, source-grounded research pipeline for
independent videos. It must discover calculations and equations from the
source material without embedding domain-specific formulas in application
code.

The implementation must not contain hard-coded recovery rules for bond yield,
YTM, EPS, gross profit, CAGR, discounted cash flow, or any other subject-matter
formula. Domain knowledge may be proposed by the model only when grounded in
the transcript or visual source and must pass deterministic validation.

## 2. Non-goals

v4.3 must not:

- Add one-off formula rules for each QC video.
- Treat the presence of any single formula as complete formula coverage.
- Infer an exact formula from general conceptual language.
- silently inject textbook equations absent from the source.
- Accept an old research package after a newer run fails.
- Depend on OCR as the primary visual-reading mechanism.
- Execute arbitrary expressions or model-produced code.

## 3. Architecture

### 3.1 Stage A — Calculation inventory

Before formula extraction, Qwen3 must identify every calculation event in the
transcript.

Required inventory schema:

```json
{
  "calculation_id": "CALC_0001",
  "name": "Human-readable calculation name",
  "source_mode": "spoken | visual_cue | mixed",
  "start_segment": 10,
  "end_segment": 14,
  "variables_mentioned": [
    "first quantity",
    "second quantity"
  ],
  "operations_mentioned": [
    "addition",
    "division"
  ],
  "visual_equation_cue": false,
  "formula_expected": true,
  "reason": "The speaker explicitly says to divide one quantity by another."
}
```

The calculation inventory is the formula-coverage contract for the video.

### 3.2 Stage B — Formula candidate extraction

For each calculation inventory item, Qwen3 proposes zero or more normalized
formula candidates.

Each candidate must include:

```json
{
  "calculation_id": "CALC_0001",
  "formula_id": "snake_case_identifier",
  "name": "Human-readable name",
  "ascii": "result = expression",
  "latex": "LaTeX expression",
  "derivation_type": "stated | derived | approximation | stated_visual",
  "variables": [
    {
      "symbol": "snake_case_symbol",
      "meaning": "Source-grounded meaning",
      "unit": "unit or empty"
    }
  ],
  "derivation_steps": [
    "Complete sentence."
  ],
  "source_claims": [
    {
      "start_segment": 10,
      "end_segment": 12,
      "relationship": "division"
    }
  ]
}
```

A candidate may be absent when the source describes a concept but does not
provide enough information to normalize a formula. That inventory item remains
unresolved until explicitly classified.

### 3.3 Stage C — General expression parser

One parser must validate spoken, derived, approximation, and visual formulas.

Allowed syntax:

- One assignment operator: `=`
- Named snake_case variables
- Numeric constants
- `+`, `-`, `*`, `/`, `^`
- Parentheses
- Unary `+` and `-`
- Whitelisted functions only:
  - `sum`
  - `sqrt`
  - `log`
  - `exp`
  - `abs`
  - `min`
  - `max`

The parser must:

- Parse into an abstract syntax tree.
- Reject arbitrary function calls, attributes, indexing, imports, and code.
- Record every variable and operation node.
- Produce a canonical normalized expression.
- Never evaluate untrusted model output with `eval`.

The same parser must be used for all formula source types.

### 3.4 Stage D — Expression-level entailment

Every formula AST node must map to evidence.

Required entailment record:

```json
{
  "formula_id": "current_yield",
  "nodes": [
    {
      "node_id": "NODE_001",
      "expression": "annual_coupon_payment / purchase_price",
      "operation": "division",
      "operand_evidence": {
        "annual_coupon_payment": [50, 50],
        "purchase_price": [51, 51]
      },
      "operator_evidence": [50, 52],
      "status": "entailed"
    }
  ]
}
```

Rules:

- Every variable must have distinct source support.
- Every operation must have direct source support or a validated algebraic
  derivation.
- A derived node may depend on previously validated nodes.
- Approximation formulas must be labeled `approximation`.
- Visual formulas must be labeled `stated_visual`.
- Unsupported nodes invalidate the candidate, not the entire video.
- Rejections must be preserved with exact reasons.

### 3.5 Stage E — Visual-equation recovery

A visual recovery task is created when the calculation inventory finds source
language such as:

- “Here is the equation.”
- “As shown in the formula.”
- “This equation.”
- “The calculation on the screen.”

Workflow:

1. Resolve the video timestamp from transcript segments.
2. Extract a short frame sequence around the cue.
3. Select candidate frames using sharpness and scene-change heuristics.
4. Send candidate frames to a vision-capable local model through a provider
   interface.
5. Request:
   - exact visual transcription;
   - normalized ASCII;
   - LaTeX;
   - variable legend;
   - confidence;
   - visible attribution.
6. Run a second independent verification pass against the selected frame.
7. Require agreement or route the item to `visual_review_required`.
8. Store:
   - selected image;
   - image SHA-256;
   - timestamp;
   - transcript cue;
   - extraction model;
   - verification model;
   - confidence;
   - attribution.

Manual `visual_formulas.json` files remain supported only as a reviewed
override and test fixture, not as the standard production path.

### 3.6 Stage F — Coverage reconciliation

Every calculation inventory item must end in exactly one state:

- `formula_retained`
- `non_symbolic_calculation`
- `insufficient_source_detail`
- `visual_review_required`
- `formula_rejected`

A package cannot pass QC when an item marked `formula_expected=true` remains
unresolved.

Coverage summary schema:

```json
{
  "identified_calculations": 5,
  "formulas_retained": 3,
  "non_symbolic_calculations": 1,
  "visual_review_required": 1,
  "unresolved": 0
}
```

A nonzero formula count alone is not sufficient.

### 3.7 Stage G — Narrative synthesis

Narrative synthesis uses only validated evidence and reconciled calculation
items.

Qwen3 settings:

- Model: `qwen3:8b`
- `think=false`
- Temperature: `0`
- Structured JSON output
- Four-sentence executive summary
- Distinct takeaways
- Sections ordered by transcript progression
- No machine-style variable names in prose
- No unsupported numbers or dates
- No prompt/debug language

### 3.8 Stage H — Package writing

Required package:

```text
Processed Research/<video_id>/
├── _READY
├── metadata.json
├── research.json
├── research.md
├── formulas.json
├── source_map.json
├── calculation_inventory.json
├── formula_entailment.json
├── formula_coverage.json
├── rejected_formulas.json
└── visual_evidence.json
```

`_READY` must be written last.

## 4. QC requirements

### 4.1 Package freshness and integrity

QC must require:

- `metadata.prompt_version` equals the requested build version.
- `metadata.source_package_sha256` equals the current transcript package SHA.
- `_READY.source_package_sha256` equals the current transcript package SHA.
- Manifest state is `research_ready`.
- Required artifacts exist.
- Artifact hashes match metadata.
- Package digest matches `_READY`.
- A failed newer run cannot fall back to an older package.

### 4.2 Formula coverage

QC must compare calculation inventory items to reconciliation results.

QC fails when:

- A required formula is missing.
- A visual equation cue remains unresolved.
- An expression node lacks entailment.
- A formula has undefined variables.
- A source citation is too broad or irrelevant.
- A derived formula lacks derivation steps.
- A stated visual formula lacks a verified image.

### 4.3 Narrative quality

QC fails for:

- Repeated takeaways.
- Unsupported numeric values or dates.
- CJK or mixed-language artifacts in English output.
- Prompt, validation, or debug leakage.
- Sections out of transcript order.
- Machine-readable identifiers in narrative prose.
- Missing package artifacts.

QC must return a readable failure report and never raise an uncaught
`FileNotFoundError`.

## 5. Domain-neutral deterministic rules

Allowed deterministic logic:

- Expression grammar and AST validation
- Variable-definition completeness
- Unit consistency checks
- Numeric grounding
- Citation localization
- Distinct evidence matching
- Operation entailment
- Formula coverage reconciliation
- Package freshness and hashes
- Output-language and prose-quality checks

Prohibited deterministic logic:

- Hard-coded bond formulas
- Hard-coded accounting formulas
- Hard-coded valuation formulas
- Subject-specific variable recovery
- Formula injection based on video title
- Formula injection based on known textbook relationships

## 6. Diagnostics

Every rejected formula must record:

```json
{
  "calculation_id": "CALC_0001",
  "candidate": {
    "ascii": "..."
  },
  "stage": "parser | entailment | coverage | visual_verification",
  "reason": "Exact rejection reason",
  "source_segments": [10, 14],
  "timestamp": "UTC timestamp"
}
```

The run summary must report:

- calculations identified;
- candidates proposed;
- formulas retained;
- formulas rejected;
- visual cues detected;
- visual formulas verified;
- unresolved inventory items;
- final package status.

## 7. Test strategy

Tests must be domain-neutral and include synthetic examples for:

- Single-operation formula
- Multi-operator formula
- Nested parentheses
- Exponents
- Whitelisted function
- Rejected arbitrary function
- Derived formula with multiple evidence nodes
- Approximation formula
- Visual formula cue
- Missing visual image
- Conflicting visual extraction
- Formula inventory with partial coverage
- Conceptual “difference between” that is not arithmetic
- Stale package after failed rerun
- Missing package
- Source SHA change
- Transcript-order narrative validation

Tests may use bond examples as one fixture, but production code must contain no
bond-specific recovery paths.

## 8. Migration from v4.2

1. Keep:
   - Qwen3 dual-mode routing
   - Package writer atomicity
   - Citation localization
   - Numeric grounding
   - Rejected-formula diagnostics
   - Visual source provenance concepts
2. Remove:
   - `recover_transcript_formulas()` domain-specific specifications
   - Bond-specific alias table entries
   - Formula-count-only coverage logic
3. Replace:
   - Single-operator transcript parser with the common AST parser
   - Manual visual sidecar as default with autonomous visual task processing
   - Formula-empty QC with inventory reconciliation
4. Add:
   - Calculation inventory artifacts
   - Expression-node entailment artifacts
   - Freshness-aware QC
   - Visual review queue

## 9. Acceptance plan

### Gate 1 — Unit tests

- All pre-v4.3 tests pass after migration.
- New generalized tests pass.
- No subject-specific formula names appear in production recovery code.

### Gate 2 — Three-video acceptance

Run:

1. Formula-heavy bond/YTM video
2. Conceptual value-investing video
3. Long narrative-heavy video

Required:

- Formula-heavy video: full inventory coverage, including verified visual
  equation.
- Conceptual video: no invented formulas and no false coverage failure.
- Long video: coherent ordered narrative and no package crash.

### Gate 3 — Batch pilot

Run five additional videos sequentially with concurrency `1`.

Required:

- No uncaught exceptions.
- Every video ends as `research_ready`, `analysis_failed`, or
  `visual_review_required`.
- No silent skips.
- QC results are persisted.

### Gate 4 — Full batch

Only after Gates 1–3 pass.

## 10. Definition of done

v4.3 is complete when:

- The model discovers calculations without domain-specific recovery code.
- All formula source types share one expression parser.
- Multi-operator spoken and visual formulas are supported.
- Formula AST nodes are individually entailed.
- Formula coverage is reconciled against a calculation inventory.
- Visual equations are recovered or explicitly queued for review.
- QC verifies package freshness and integrity.
- The three-video acceptance batch passes.
- The implementation is committed and tagged `research-v4.3`.
