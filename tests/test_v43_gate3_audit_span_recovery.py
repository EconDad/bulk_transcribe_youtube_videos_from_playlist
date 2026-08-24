from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.gate3_audit_recovery import (
    parse_inventory_evidence_audit_response_with_gate3_repair,
)
from research_v43.inventory_evidence_audit import AuditAction


class Gate3AuditSpanRecoveryTests(unittest.TestCase):
    def test_widens_result_only_selection_to_ground_named_operands(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Sum listed assets",
                "source_mode": "spoken",
                "start_segment": 8,
                "end_segment": 9,
                "variables_mentioned": [
                    "cash",
                    "stand",
                    "machine",
                    "supplies",
                    "land",
                ],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states a summed total.",
            }
        )
        segments = [
            {"text": "The account has cash on hand."},
            {"text": "The stand is another asset."},
            {"text": "The machine is also listed."},
            {"text": "Supplies are on hand."},
            {"text": "Land is the final asset."},
            {"text": "Additional explanation."},
            {"text": "Additional explanation."},
            {"text": "Additional explanation."},
            {"text": "When we sum up those assets,"},
            {"text": "the result is the total."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [8, 9],
            "reason": "The source states the sum after listing its operands.",
            "revised_variables_mentioned": [
                "cash",
                "stand",
                "machine",
                "supplies",
                "land",
            ],
            "revised_operations_mentioned": ["addition"],
        }

        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=9,
        )

        self.assertEqual(decision.action, AuditAction.RECONCILE)
        self.assertEqual(min(decision.evidence_segment_ids), 0)
        self.assertEqual(max(decision.evidence_segment_ids), 9)
        self.assertEqual(
            decision.revised_operations_mentioned,
            ("addition",),
        )

    def test_does_not_widen_when_revised_claim_never_grounds(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Synthetic sum",
                "source_mode": "spoken",
                "start_segment": 1,
                "end_segment": 1,
                "variables_mentioned": ["missing operand", "present operand"],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "Synthetic item.",
            }
        )
        segments = [
            {"text": "The present operand is listed."},
            {"text": "We add the values."},
            {"text": "No other named quantity appears."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [1],
            "reason": "Attempted reconciliation.",
            "revised_variables_mentioned": [
                "missing operand",
                "present operand",
            ],
            "revised_operations_mentioned": ["addition"],
        }

        with self.assertRaises(Exception):
            parse_inventory_evidence_audit_response_with_gate3_repair(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=0,
                neighborhood_end=2,
            )


if __name__ == "__main__":
    unittest.main()
