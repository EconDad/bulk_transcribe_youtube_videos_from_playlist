from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.inventory_evidence_audit import (
    AuditAction,
    InventoryEvidenceAuditError,
    apply_inventory_audit_decision,
    build_inventory_evidence_audit_prompt,
    parse_inventory_evidence_audit_response,
)


def make_item(**updates):
    raw = {
        "calculation_id": "CALC_0001",
        "name": "Compute a total",
        "source_mode": "spoken",
        "start_segment": 1,
        "end_segment": 1,
        "variables_mentioned": ["total value"],
        "operations_mentioned": ["addition"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The source describes a quantitative result.",
    }
    raw.update(updates)
    return CalculationItem.from_mapping(raw)


class ClaimReconciliationTests(unittest.TestCase):
    def test_reconcile_corrects_wrong_operation_and_variables(self):
        item = make_item()
        segments = [
            {
                "text": (
                    "Take the number of items, 4, and multiply it by "
                    "the item value, 5."
                )
            },
            {"text": "That gives a total value of 20."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "revised_variables_mentioned": [
                "number of items",
                "item value",
            ],
            "revised_operations_mentioned": ["multiplication"],
            "reason": "The source states a multiplication procedure.",
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=1,
        )
        self.assertEqual(decision.action, AuditAction.RECONCILE)
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertEqual((updated.start_segment, updated.end_segment), (0, 1))
        self.assertEqual(
            updated.variables_mentioned,
            ("number of items", "item value"),
        )
        self.assertEqual(updated.operations_mentioned, ("multiplication",))
        self.assertTrue(updated.formula_expected)

    def test_reconcile_can_expand_same_claims_beyond_auto_radius(self):
        item = make_item(
            start_segment=8,
            end_segment=8,
            variables_mentioned=["starting amount", "deduction"],
            operations_mentioned=["subtraction"],
        )
        segments = [
            {"text": "The starting amount is 100."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "The deduction is 20."},
            {"text": "Context."},
            {"text": "Subtract the deduction from the starting amount."},
            {"text": "The result is 80."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 5, 7, 8],
            "revised_variables_mentioned": [
                "starting amount",
                "deduction",
            ],
            "revised_operations_mentioned": ["subtraction"],
            "reason": "The wider bounded discourse states the full procedure.",
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=8,
        )
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertEqual((updated.start_segment, updated.end_segment), (0, 8))
        self.assertEqual(updated.operations_mentioned, ("subtraction",))

    def test_reconcile_rejects_ungrounded_revised_variable(self):
        item = make_item()
        segments = [
            {"text": "Multiply the count by the unit amount."},
            {"text": "That gives the total value."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "revised_variables_mentioned": ["missing quantity"],
            "revised_operations_mentioned": ["multiplication"],
            "reason": "Attempted correction.",
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
                neighborhood_end=1,
            )

    def test_reconcile_rejects_noncanonical_operation(self):
        item = make_item()
        segments = [
            {"text": "Multiply the count by the unit amount."},
            {"text": "That gives the total value."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "revised_variables_mentioned": ["count", "unit amount"],
            "revised_operations_mentioned": ["comparison"],
            "reason": "Attempted correction.",
        }
        with self.assertRaisesRegex(
            InventoryEvidenceAuditError,
            "revised operations are not canonical",
        ):
            parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=1,
            )

    def test_downgrade_requires_empty_revised_claim_arrays(self):
        item = make_item()
        segments = [{"text": "Only the final result is 20."}, {"text": "Done."}]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "downgrade_non_symbolic",
            "evidence_segment_ids": [1],
            "revised_variables_mentioned": ["total value"],
            "revised_operations_mentioned": [],
            "reason": "Result only.",
        }
        with self.assertRaisesRegex(
            InventoryEvidenceAuditError,
            "requires empty revised claim arrays",
        ):
            parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=1,
            )

    def test_prompt_is_domain_neutral_and_exposes_canonical_operations(self):
        item = make_item()
        prompt = build_inventory_evidence_audit_prompt(
            item=item,
            neighborhood_segments=[
                {"segment_id": 0, "text": "Multiply two measurements."},
                {"segment_id": 1, "text": "The result is shown."},
            ],
            selection_reasons=["operation claim is unsupported"],
        )
        lowered = prompt.lower()
        self.assertIn("reconcile", lowered)
        self.assertIn("canonical operations", lowered)
        self.assertNotIn("coupon", lowered)
        self.assertNotIn("yield to maturity", lowered)


if __name__ == "__main__":
    unittest.main()
