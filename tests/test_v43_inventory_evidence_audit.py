from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.inventory_evidence_audit import (
    InventoryEvidenceAuditError,
    apply_inventory_audit_decision,
    build_inventory_evidence_audit_prompt,
    item_needs_evidence_audit,
    parse_inventory_evidence_audit_response,
)


def make_item(**updates):
    raw = {
        "calculation_id": "CALC_0001",
        "name": "Normalize a measurement",
        "source_mode": "spoken",
        "start_segment": 1,
        "end_segment": 1,
        "variables_mentioned": ["total amount", "item count"],
        "operations_mentioned": ["division"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The source describes a normalized measurement.",
    }
    raw.update(updates)
    return CalculationItem.from_mapping(raw)


class InventoryEvidenceAuditTests(unittest.TestCase):
    def test_selects_item_when_operand_is_outside_current_span(self):
        item = make_item()
        segments = [
            {"text": "The item count is 4."},
            {"text": "The result is 5."},
            {"text": "Divide the total amount by the item count."},
        ]
        needed, reasons = item_needs_evidence_audit(
            item=item,
            segments=segments,
        )
        self.assertTrue(needed)
        self.assertTrue(reasons)

    def test_skips_item_when_current_span_contains_claimed_evidence(self):
        item = make_item(start_segment=0, end_segment=0)
        segments = [
            {
                "text": (
                    "Divide the total amount by the item count "
                    "to obtain the result."
                )
            }
        ]
        needed, reasons = item_needs_evidence_audit(
            item=item,
            segments=segments,
        )
        self.assertFalse(needed)
        self.assertEqual(reasons, ())

    def test_valid_expand_is_applied(self):
        item = make_item()
        segments = [
            {"text": "The item count is 4."},
            {"text": "The result is 5."},
            {"text": "Divide the total amount by the item count."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "expand",
            "start_segment": 0,
            "end_segment": 2,
            "reason": "Nearby source supplies the missing relationship.",
            "evidence": [
                {
                    "kind": "relationship",
                    "start_segment": 2,
                    "end_segment": 2,
                    "quote": "Divide the total amount by the item count.",
                },
                {
                    "kind": "operand",
                    "start_segment": 0,
                    "end_segment": 0,
                    "quote": "The item count is 4.",
                },
            ],
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=2,
        )
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertEqual(
            (updated.start_segment, updated.end_segment),
            (0, 2),
        )
        self.assertTrue(updated.formula_expected)

    def test_downgrade_sets_formula_expected_false(self):
        item = make_item(
            variables_mentioned=["observed result"],
            operations_mentioned=[],
        )
        segments = [
            {"text": "The observed result is 17."},
            {"text": "That is the final outcome."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "downgrade_non_symbolic",
            "start_segment": 1,
            "end_segment": 1,
            "reason": "Only a final outcome is stated in the bounded source.",
            "evidence": [
                {
                    "kind": "result",
                    "start_segment": 1,
                    "end_segment": 1,
                    "quote": "That is the final outcome.",
                }
            ],
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=1,
        )
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertFalse(updated.formula_expected)

    def test_non_exact_evidence_quote_is_rejected(self):
        item = make_item()
        segments = [
            {"text": "The item count is 4."},
            {"text": "The result is 5."},
            {"text": "Divide the total amount by the item count."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "expand",
            "start_segment": 0,
            "end_segment": 2,
            "reason": "Expanded source.",
            "evidence": [
                {
                    "kind": "relationship",
                    "start_segment": 2,
                    "end_segment": 2,
                    "quote": "Multiply the values.",
                }
            ],
        }
        with self.assertRaisesRegex(
            InventoryEvidenceAuditError,
            "quote is not present",
        ):
            parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=2,
            )

    def test_prompt_is_domain_neutral(self):
        item = make_item()
        prompt = build_inventory_evidence_audit_prompt(
            item=item,
            neighborhood_segments=[
                {
                    "segment_id": 0,
                    "text": "Divide the total amount by the item count.",
                }
            ],
            selection_reasons=["missing operand"],
        )
        self.assertIn("bounded transcript", prompt.lower())
        self.assertNotIn("coupon", prompt.lower())
        self.assertNotIn("yield to maturity", prompt.lower())


if __name__ == "__main__":
    unittest.main()
