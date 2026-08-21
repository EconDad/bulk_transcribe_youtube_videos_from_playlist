from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import run_research_v43_resilient as resilient
from research_v43.calculation_inventory import (
    CalculationInventory,
    CalculationItem,
    SourceMode,
)
from research_v43.expression_ast import FormulaCandidate
from research_v43.gate2_recovery import (
    audit_with_gate2_semantic_downgrades,
    find_deterministic_expansion_gate2,
    validate_entailment_response_with_gate2_quote_repair,
)


class Gate2RecoveryTests(unittest.TestCase):
    def _item(
        self,
        *,
        calculation_id="CALC_0001",
        name="Example",
        start=0,
        end=0,
        variables=(),
        operations=(),
    ):
        return CalculationItem(
            calculation_id=calculation_id,
            name=name,
            source_mode=SourceMode.SPOKEN,
            start_segment=start,
            end_segment=end,
            variables_mentioned=tuple(variables),
            operations_mentioned=tuple(operations),
            visual_equation_cue=False,
            formula_expected=True,
            reason="Synthetic source-grounded event.",
        )

    def test_numeric_instance_without_local_operand_operation_binding_is_downgraded(self):
        segments = [
            {"text": "For example, divide 50 by 1000."},
            {"text": "Now let's say the price is 800."},
            {"text": "The result is 6.3 percent."},
        ]
        item = self._item(
            start=0,
            end=2,
            variables=("50", "800"),
            operations=("division",),
        )
        inventory = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(item,),
        )

        audited, records = audit_with_gate2_semantic_downgrades(
            inventory=inventory,
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertTrue(
            any(
                record.get("action")
                == "downgrade_non_symbolic_numeric_instance"
                for record in records
            )
        )

    def test_post_evidence_audit_rechecks_expanded_span(self):
        segments = [
            {"text": "For example, divide 50 by 1000."},
            {"text": "Now let's say the price is 800."},
            {"text": "The result is 6.3 percent."},
        ]
        item = self._item(
            start=2,
            end=2,
            variables=("50", "800"),
            operations=("division",),
        )
        inventory = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(item,),
        )
        expanded = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(replace(item, start_segment=0, end_segment=2),),
        )

        def fake_evidence_audit(*, inventory, segments, **kwargs):
            return expanded, (
                {
                    "calculation_id": "CALC_0001",
                    "action": "expand",
                    "decision_source": "synthetic_test",
                },
            )

        with mock.patch.object(
            resilient,
            "_ORIGINAL_RUN_INVENTORY_EVIDENCE_AUDIT",
            side_effect=fake_evidence_audit,
        ):
            audited, records = (
                resilient._run_inventory_evidence_audit_with_gate2_postcheck(
                    inventory=inventory,
                    segments=segments,
                )
            )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertTrue(
            any(
                record.get("action")
                == "downgrade_non_symbolic_numeric_instance"
                for record in records
            )
        )

    def test_outcome_without_complete_reusable_support_is_downgraded(self):
        segments = [
            {"text": "Assume the earlier amount keeps compounding."},
            {"text": "At the end the reported result is 3400."},
        ]
        item = self._item(
            start=1,
            end=1,
            variables=("25", "5%"),
            operations=("exponentiation", "addition"),
        )
        inventory = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(item,),
        )

        audited, records = audit_with_gate2_semantic_downgrades(
            inventory=inventory,
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertTrue(
            any(
                record.get("action")
                == "downgrade_non_symbolic_outcome_only"
                for record in records
            )
        )

    def test_coherent_numeric_subtraction_example_is_preserved(self):
        segments = [
            {"text": "Let's say you paid 1200 for the item."},
            {"text": "At the end you only get 1000 back."},
            {"text": "So you lose 200."},
        ]
        item = self._item(
            start=0,
            end=2,
            variables=("1200", "1000"),
            operations=("subtraction",),
        )
        inventory = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(item,),
        )

        audited, records = audit_with_gate2_semantic_downgrades(
            inventory=inventory,
            segments=segments,
        )

        self.assertTrue(audited.calculations[0].formula_expected)
        self.assertFalse(
            any(
                record.get("action")
                == "downgrade_non_symbolic_numeric_instance"
                for record in records
            )
        )

    def test_matching_number_without_local_operation_cue_does_not_preserve(self):
        segments = [
            {"text": "For example, divide 50 by 1000."},
            {"text": "Suppose another input is 800."},
            {"text": "The reported result is 6.3 percent."},
        ]
        item = self._item(
            start=0,
            end=2,
            variables=("50", "800"),
            operations=("division",),
        )
        inventory = CalculationInventory(
            schema_version="1.0",
            video_id="video",
            calculations=(item,),
        )

        audited, records = audit_with_gate2_semantic_downgrades(
            inventory=inventory,
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertTrue(
            any(
                record.get("action")
                == "downgrade_non_symbolic_numeric_instance"
                for record in records
            )
        )

    def test_bounded_expansion_can_use_full_audit_neighborhood(self):
        segments = [
            {"text": "The face value is 1000."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "Context."},
            {"text": "He paid a premium and will lose 200."},
        ]
        item = self._item(
            start=6,
            end=6,
            variables=("premium paid", "face value"),
            operations=("subtraction",),
        )

        decision = find_deterministic_expansion_gate2(
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=6,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.evidence_segment_ids[0], 0)
        self.assertEqual(decision.evidence_segment_ids[-1], 6)

    def test_quote_repair_uses_only_existing_cited_ranges(self):
        segments = [
            {"text": "The loss is 200."},
            {"text": "That 200 comes off the coupon payments for a difference of 898."},
            {"text": "He will make 898."},
        ]
        item = self._item(
            start=0,
            end=2,
            variables=("coupon payments", "loss"),
            operations=("subtraction",),
        )
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "result",
                "name": "Result",
                "ascii": "result = coupon_payments - loss",
                "latex": "result = coupon\\_payments - loss",
                "derivation_type": "derived",
                "variables": [
                    {"symbol": "result", "meaning": "result", "unit": ""},
                    {"symbol": "coupon_payments", "meaning": "coupon payments", "unit": ""},
                    {"symbol": "loss", "meaning": "loss", "unit": ""},
                ],
                "derivation_steps": ["The loss comes off the coupon payments."],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 2,
                        "relationship": "subtraction",
                    }
                ],
            }
        )
        payload = {
            "calculation_id": "CALC_0001",
            "formula_id": "result",
            "nodes": [
                {
                    "node_id": "NODE_0001",
                    "expression": "coupon_payments - loss",
                    "operation": "subtraction",
                    "status": "derived",
                    "evidence": [
                        {
                            "start_segment": 0,
                            "end_segment": 2,
                            "quote": "Subtract the loss from coupon payments to get 898.",
                        }
                    ],
                    "identifier_groundings": [
                        {
                            "identifier": "coupon_payments",
                            "start_segment": 1,
                            "end_segment": 1,
                            "quote": "coupon payments",
                        },
                        {
                            "identifier": "loss",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "loss is 200",
                        },
                        {
                            "identifier": "result",
                            "start_segment": 2,
                            "end_segment": 2,
                            "quote": "He will make 89,8.",
                        },
                    ],
                    "depends_on_node_ids": [],
                    "derivation_step": "The loss comes off the coupon payments.",
                }
            ],
        }

        report = validate_entailment_response_with_gate2_quote_repair(
            payload,
            item=item,
            candidate=candidate,
            segments=segments,
        )

        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
