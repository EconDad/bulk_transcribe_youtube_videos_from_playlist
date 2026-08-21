#!/usr/bin/env python3
"""Resilient v4.3 diagnostic entrypoint for Stage F acceptance.

This keeps the frozen core diagnostic runner intact while layering only bounded,
domain-neutral recoveries proven useful by real-video integration:
- retry inventory generation once when thinking exhausts the response budget;
- repair calculation-array ordering without changing item contents;
- downgrade comparison-only and worked-example inventory events when bounded
  source lacks a reusable arithmetic operation;
- downgrade incomplete arithmetic antecedents when the source never grounds a
  result concept, avoiding invented left-hand variables;
- downgrade numeric outcomes whose full bounded source still lacks a reusable
  symbolic relationship;
- allow deterministic inventory expansion across the already-bounded audit
  neighborhood;
- re-run the same semantic downgrade pass after inventory evidence expansion so
  independently grounded nearby operands cannot become a false relationship;
- complete missing formula variable metadata without changing expressions;
- widen node operation evidence only to the hull of already-cited groundings;
- localize paraphrased entailment quotes to exact text inside unchanged cited
  ranges before rerunning strict validation.
"""

from __future__ import annotations

import run_research_v43 as base

from research_v43.gate2_recovery import (
    audit_with_gate2_semantic_downgrades,
    find_deterministic_expansion_gate2,
    validate_entailment_response_with_gate2_quote_repair,
)
from research_v43.inventory_recovery import (
    AdaptiveInventoryOllamaClient,
    parse_inventory_response_with_order_repair,
)
from research_v43.semantic_recovery import (
    parse_formula_extraction_response_with_variable_completion,
)


_ORIGINAL_RUN_INVENTORY_EVIDENCE_AUDIT = base._run_inventory_evidence_audit


def _run_inventory_evidence_audit_with_gate2_postcheck(
    *,
    inventory,
    segments,
    **kwargs,
):
    """Apply Gate 2 association checks to the final evidence-audited spans.

    The frozen evidence audit may legitimately expand an inventory item by
    collecting independently grounded variables and operation cues from nearby
    segments. Gate 2's stricter association check therefore has to see that
    expanded span before formula extraction; otherwise a relationship that was
    rejected in a post-audit preview can escape during a real pipeline run.
    """

    audited, evidence_records = _ORIGINAL_RUN_INVENTORY_EVIDENCE_AUDIT(
        inventory=inventory,
        segments=segments,
        **kwargs,
    )
    postchecked, gate2_records = audit_with_gate2_semantic_downgrades(
        inventory=audited,
        segments=segments,
    )
    return postchecked, tuple((*evidence_records, *gate2_records))


def main(argv=None) -> int:
    base.OllamaJsonClient = AdaptiveInventoryOllamaClient
    base.parse_inventory_response = parse_inventory_response_with_order_repair
    base.audit_visual_equation_cues = audit_with_gate2_semantic_downgrades
    base.find_deterministic_expansion = find_deterministic_expansion_gate2
    base._run_inventory_evidence_audit = (
        _run_inventory_evidence_audit_with_gate2_postcheck
    )
    base.parse_formula_extraction_response = (
        parse_formula_extraction_response_with_variable_completion
    )
    base.validate_entailment_response = (
        validate_entailment_response_with_gate2_quote_repair
    )
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
