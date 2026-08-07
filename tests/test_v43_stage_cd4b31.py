from __future__ import annotations

import inspect
import json
import unittest

import run_research_v43
from research_v43.calculation_inventory import CalculationItem
from research_v43.inventory_evidence_audit import (
    InventoryEvidenceAuditError,
    build_inventory_evidence_audit_prompt,
    parse_inventory_evidence_audit_response,
)


def make_item(**updates):
    raw = {
        "calculation_id": "CALC_0001",
        "name": "Distributed arithmetic relationship",
        "source_mode": "spoken",
        "start_segment": 4,
        "end_segment": 4,
        "variables_mentioned": ["old variable"],
        "operations_mentioned": ["addition"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "Inventory claim needs bounded reconciliation.",
    }
    raw.update(updates)
    return CalculationItem.from_mapping(raw)


class SpanGroundedFailClosedTests(unittest.TestCase):
    def test_reconcile_validates_entire_contiguous_candidate_span(self):
        item = make_item()
        segments = [
            {"text": "Noise."},
            {"text": "The first amount is 20."},
            {"text": "Noise."},
            {"text": "Multiply the first amount by the item count."},
            {"text": "The result is 80."},
            {"text": "The item count is 4."},
            {"text": "Noise."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [1, 5],
            "revised_variables_mentioned": [
                "first amount",
                "item count",
            ],
            "revised_operations_mentioned": ["multiplication"],
            "reason": "The complete relationship spans the bounded context.",
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=6,
        )
        self.assertEqual(decision.evidence_segment_ids, (1, 5))

    def test_reconcile_still_rejects_claim_absent_from_contiguous_span(self):
        item = make_item()
        segments = [
            {"text": "Noise."},
            {"text": "The first amount is 20."},
            {"text": "Noise."},
            {"text": "Multiply the first amount by the item count."},
            {"text": "The result is 80."},
            {"text": "The item count is 4."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [1, 5],
            "revised_variables_mentioned": [
                "first amount",
                "missing quantity",
            ],
            "revised_operations_mentioned": ["multiplication"],
            "reason": "Attempted reconciliation.",
        }
        with self.assertRaisesRegex(
            InventoryEvidenceAuditError,
            "revised variables are not grounded",
        ):
            parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=5,
            )

    def test_prompt_explains_contiguous_span_endpoints(self):
        item = make_item()
        prompt = build_inventory_evidence_audit_prompt(
            item=item,
            neighborhood_segments=[
                {"segment_id": 3, "text": "Multiply the two amounts."},
                {"segment_id": 4, "text": "The result is shown."},
            ],
            selection_reasons=("missing evidence",),
        ).casefold()
        self.assertIn("contiguous", prompt)
        self.assertIn("outer segment endpoints", prompt)

    def test_audit_failure_index_contains_only_failed_records(self):
        records = [
            {
                "calculation_id": "CALC_0001",
                "action": "audit_failed",
                "validation_error": "bad evidence",
            },
            {
                "calculation_id": "CALC_0002",
                "action": "reconcile",
            },
        ]
        failures = run_research_v43._audit_failures_by_id(records)
        self.assertEqual(set(failures), {"CALC_0001"})

    def test_run_pipeline_checks_audit_failure_before_extraction(self):
        source = inspect.getsource(run_research_v43.run_pipeline)
        guard = source.index(
            "audit_failure = audit_failures.get(item.calculation_id)"
        )
        extraction = source.index(
            'extraction_stage = f"formula_extraction {item.calculation_id}"'
        )
        self.assertLess(guard, extraction)


if __name__ == "__main__":
    unittest.main()
