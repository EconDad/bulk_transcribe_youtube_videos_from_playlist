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
- complete missing formula variable metadata without changing expressions;
- widen node operation evidence only to the hull of already-cited groundings
  inside the validated calculation span.
"""

from __future__ import annotations

import run_research_v43 as base

from research_v43.inventory_recovery import (
    AdaptiveInventoryOllamaClient,
    parse_inventory_response_with_order_repair,
)
from research_v43.operation_fragment_recovery import (
    audit_with_incomplete_operation_fragment_downgrade,
)
from research_v43.semantic_recovery import (
    parse_formula_extraction_response_with_variable_completion,
    validate_entailment_response_with_grounding_hull_repair,
)


def main(argv=None) -> int:
    base.OllamaJsonClient = AdaptiveInventoryOllamaClient
    base.parse_inventory_response = parse_inventory_response_with_order_repair
    base.audit_visual_equation_cues = (
        audit_with_incomplete_operation_fragment_downgrade
    )
    base.parse_formula_extraction_response = (
        parse_formula_extraction_response_with_variable_completion
    )
    base.validate_entailment_response = (
        validate_entailment_response_with_grounding_hull_repair
    )
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
