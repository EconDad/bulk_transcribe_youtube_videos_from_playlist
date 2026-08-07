from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.inventory_evidence_audit import (
    InventoryEvidenceAuditError,
    apply_inventory_audit_decision,
    decision_evidence_records,
    find_deterministic_expansion,
    item_needs_evidence_audit,
    parse_inventory_evidence_audit_response,
)


def make_item(**updates):
    raw = {
        "calculation_id": "CALC_0001",
        "name": "Normalize a measurement",
        "source_mode": "spoken",
        "start_segment": 2,
        "end_segment": 2,
        "variables_mentioned": ["sample payments", "item count"],
        "operations_mentioned": ["division"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The source describes an arithmetic procedure.",
    }
    raw.update(updates)
    return CalculationItem.from_mapping(raw)


class InventoryEvidenceAuditRevisionTests(unittest.TestCase):
    def test_plural_variable_can_match_singular_source(self):
        item = make_item(
            start_segment=0,
            end_segment=0,
            variables_mentioned=["sample payments"],
            operations_mentioned=[],
        )
        needed, _ = item_needs_evidence_audit(
            item=item,
            segments=[{"text": "Each sample payment is recorded."}],
        )
        self.assertFalse(needed)

    def test_deterministic_search_computes_minimal_span(self):
        item = make_item()
        segments = [
            {"text": "There are 4 items."},
            {"text": "The sample payment is 20 units."},
            {"text": "The result is 5 units."},
            {"text": "Divide the sample payment by the item count."},
        ]
        decision = find_deterministic_expansion(
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=3,
        )
        self.assertIsNotNone(decision)
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertEqual(
            (updated.start_segment, updated.end_segment),
            (1, 3),
        )
        needed, reasons = item_needs_evidence_audit(
            item=updated,
            segments=segments,
        )
        self.assertFalse(needed)
        self.assertEqual(reasons, ())

    def test_parser_uses_segment_ids_without_quotes_or_kinds(self):
        item = make_item(
            variables_mentioned=["observed output"],
            operations_mentioned=["multiplication"],
        )
        segments = [
            {"text": "The starting input is 4."},
            {"text": "The observed output is 8."},
            {"text": "This is the final example."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "downgrade_non_symbolic",
            "evidence_segment_ids": [1, 2],
            "reason": "Only an outcome is grounded.",
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=2,
        )
        self.assertEqual(decision.evidence_segment_ids, (1, 2))
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        self.assertFalse(updated.formula_expected)

    def test_expand_is_rejected_when_claims_remain_unsupported(self):
        item = make_item(
            variables_mentioned=["sample payment"],
            operations_mentioned=["multiplication"],
        )
        segments = [
            {"text": "The sample payment is 20."},
            {"text": "The result is 10."},
            {"text": "Another observation follows."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "expand",
            "evidence_segment_ids": [0, 1],
            "reason": "Use the nearby example.",
        }
        with self.assertRaisesRegex(
            InventoryEvidenceAuditError,
            "does not ground current inventory claims",
        ):
            parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=2,
            )

    def test_python_copies_exact_source_text(self):
        item = make_item()
        segments = [
            {"text": "First exact sentence."},
            {"text": "Middle sentence."},
            {"text": "Third exact sentence."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "downgrade_non_symbolic",
            "evidence_segment_ids": [0, 2],
            "reason": "Result-only event.",
        }
        decision = parse_inventory_evidence_audit_response(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=2,
        )
        self.assertEqual(
            decision_evidence_records(
                decision=decision,
                segments=segments,
            ),
            (
                {"segment_id": 0, "source_text": "First exact sentence."},
                {"segment_id": 2, "source_text": "Third exact sentence."},
            ),
        )


if __name__ == "__main__":
    unittest.main()
