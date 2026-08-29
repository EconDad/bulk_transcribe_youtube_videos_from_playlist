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

    def test_adjacent_extension_recovers_operand_just_outside_audit_neighborhood(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Sum listed values",
                "source_mode": "spoken",
                "start_segment": 11,
                "end_segment": 12,
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
            {"text": "The cash is introduced here."},
            {"text": "Background."},
            {"text": "Background."},
            {"text": "The stand is listed."},
            {"text": "The machine is listed."},
            {"text": "The supplies are listed."},
            {"text": "The land is listed."},
            {"text": "Background."},
            {"text": "Background."},
            {"text": "Background."},
            {"text": "Background."},
            {"text": "When we sum up those assets,"},
            {"text": "the result is the total."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [11, 12],
            "reason": "The result follows the operand list.",
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
            neighborhood_start=3,
            neighborhood_end=12,
        )

        self.assertEqual(decision.action, AuditAction.RECONCILE)
        self.assertEqual(min(decision.evidence_segment_ids), 0)
        self.assertEqual(max(decision.evidence_segment_ids), 12)
        self.assertIn("adjacent operand recovery", decision.reason)

    def test_adjacent_extension_requires_operation_in_selected_span(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Synthetic calculation",
                "source_mode": "spoken",
                "start_segment": 5,
                "end_segment": 5,
                "variables_mentioned": ["first", "second"],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "Synthetic item.",
            }
        )
        segments = [
            {"text": "The first value appears."},
            {"text": "The second value appears."},
            {"text": "We add those values."},
            {"text": "Background."},
            {"text": "Background."},
            {"text": "The result is stated here."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [5],
            "reason": "Attempted reconciliation.",
            "revised_variables_mentioned": ["first", "second"],
            "revised_operations_mentioned": ["addition"],
        }

        with self.assertRaises(Exception):
            parse_inventory_evidence_audit_response_with_gate3_repair(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=3,
                neighborhood_end=5,
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

    def test_recovers_grounded_original_values_when_model_returns_aliases(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Unit price calculation",
                "source_mode": "spoken",
                "start_segment": 2,
                "end_segment": 2,
                "variables_mentioned": [
                    "total_value",
                    "number_of_units",
                    "$100,000",
                    "10,000",
                ],
                "operations_mentioned": ["division"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states a division.",
            }
        )
        segments = [
            {"text": "The asset is valued at $100,000."},
            {"text": "It is split into 10,000 units."},
            {"text": "Take 100,000 divided by 10,000."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1, 2],
            "reason": "The division is explicit.",
            "revised_variables_mentioned": [
                "total_value",
                "number_of_units",
            ],
            "revised_operations_mentioned": ["division"],
        }

        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=2,
        )

        self.assertEqual(decision.action, AuditAction.RECONCILE)
        self.assertEqual(
            decision.revised_variables_mentioned,
            ("$100,000", "10,000"),
        )

    def test_spoken_numeric_result_is_not_reused_as_addition_operand(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Reported total",
                "source_mode": "spoken",
                "start_segment": 1,
                "end_segment": 2,
                "variables_mentioned": ["1418", "518"],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The inventory treated both values as operands.",
            }
        )
        segments = [
            {"text": "The value rose to one thousand four"},
            {"text": "hundred and eighteen then add the unspecified payments"},
            {"text": "The reported result goes up to five hundred and eighteen."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1, 2],
            "reason": "Attempted numeric reconciliation.",
            "revised_variables_mentioned": ["1418", "518"],
            "revised_operations_mentioned": ["addition"],
        }

        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=2,
        )

        self.assertEqual(decision.action, AuditAction.DOWNGRADE_NON_SYMBOLIC)

    def test_prior_result_may_be_reused_with_a_distinct_grounded_operand(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Chained calculation",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 1,
                "variables_mentioned": ["12", "3"],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source reuses a prior result.",
            }
        )
        segments = [
            {"text": "The first result equals twelve."},
            {"text": "Then add twelve and three to get fifteen."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "reason": "Both inputs and the operation are explicit.",
            "revised_variables_mentioned": ["12", "3"],
            "revised_operations_mentioned": ["addition"],
        }

        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=1,
        )

        self.assertEqual(decision.action, AuditAction.RECONCILE)

    def test_unrelated_times_word_does_not_block_non_symbolic_downgrade(self):
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Reported percentage",
                "source_mode": "spoken",
                "start_segment": 1,
                "end_segment": 1,
                "variables_mentioned": ["518", "1000"],
                "operations_mentioned": ["percentage calculation"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The inventory inferred an operation.",
            }
        )
        segments = [
            {"text": "The initial amount was one thousand."},
            {"text": "The reported gain was five hundred and eighteen."},
            {"text": "A percentage was reported without a procedure."},
            {"text": "A lot of the times people compare the outcomes."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [1, 2],
            "reason": "The model guessed multiplication.",
            "revised_variables_mentioned": ["518", "1000"],
            "revised_operations_mentioned": ["multiplication"],
        }

        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=1,
            neighborhood_end=2,
        )

        self.assertEqual(decision.action, AuditAction.DOWNGRADE_NON_SYMBOLIC)


if __name__ == "__main__":
    unittest.main()
