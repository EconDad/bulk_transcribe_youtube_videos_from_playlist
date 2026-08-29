from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationInventory
from research_v43.expression_ast import FormulaCandidate
from research_v43.semantic_recovery import (
    audit_visual_equation_cues_with_semantic_downgrades,
    parse_formula_extraction_response_with_variable_completion,
    validate_entailment_response_with_grounding_hull_repair,
)


def inventory_with(item):
    return CalculationInventory.from_mapping(
        {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [item],
        }
    )


class SemanticRecoveryTests(unittest.TestCase):
    def test_comparison_only_event_is_downgraded_non_symbolic(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Rate comparison",
            "source_mode": "spoken",
            "start_segment": 0,
            "end_segment": 1,
            "variables_mentioned": ["income growth", "interest rate"],
            "operations_mentioned": ["comparison"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "The source compares two rates.",
        }
        segments = [
            {"text": "Income growth must be higher than the interest rate."},
            {"text": "Income needs to grow faster."},
        ]

        audited, records = audit_visual_equation_cues_with_semantic_downgrades(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertEqual(
            records[-1]["action"],
            "downgrade_non_symbolic_comparison",
        )

    def test_worked_numeric_example_without_operation_is_downgraded(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Worked spending example",
            "source_mode": "spoken",
            "start_segment": 3,
            "end_segment": 3,
            "variables_mentioned": ["income", "borrowed_amount"],
            "operations_mentioned": ["addition"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "Inventory inferred addition from numeric values.",
        }
        segments = [
            {"text": "Let me give you an example."},
            {"text": "Suppose you earn $100,000 a year and have no debt."},
            {"text": "You can borrow $10,000 on a credit card."},
            {"text": "So you can spend $110,000 even though you earn $100,000."},
        ]

        audited, records = audit_visual_equation_cues_with_semantic_downgrades(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertEqual(
            records[-1]["action"],
            "downgrade_non_symbolic_numeric_example",
        )

    def test_explicit_reusable_addition_is_not_downgraded(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Total resources",
            "source_mode": "spoken",
            "start_segment": 1,
            "end_segment": 1,
            "variables_mentioned": ["income", "borrowed amount"],
            "operations_mentioned": ["addition"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "The source explicitly states addition.",
        }
        segments = [
            {"text": "For example, consider household resources."},
            {"text": "Add income and borrowed amount to get total resources."},
        ]

        audited, records = audit_visual_equation_cues_with_semantic_downgrades(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertTrue(audited.calculations[0].formula_expected)
        self.assertFalse(
            any(
                str(record.get("action", "")).startswith(
                    "downgrade_non_symbolic"
                )
                for record in records
            )
        )

    def test_missing_result_variable_metadata_is_completed(self):
        item = inventory_with(
            {
                "calculation_id": "CALC_0001",
                "name": "Spending ratio",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": ["spending", "money spent"],
                "operations_mentioned": ["division"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states division.",
            }
        ).calculations[0]
        payload = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "The source states a reusable ratio.",
            "candidates": [
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "spending_ratio",
                    "name": "Spending ratio",
                    "ascii": "ratio = spending / money_spent",
                    "latex": r"r = \frac{s}{m}",
                    "derivation_type": "stated",
                    "variables": [
                        {
                            "symbol": "spending",
                            "meaning": "spending",
                            "unit": "",
                        },
                        {
                            "symbol": "money_spent",
                            "meaning": "money spent",
                            "unit": "",
                        },
                    ],
                    "derivation_steps": ["Divide spending by money spent."],
                    "source_claims": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "relationship": "Divide spending by money spent.",
                        }
                    ],
                }
            ],
        }

        parsed = parse_formula_extraction_response_with_variable_completion(
            payload,
            item=item,
        )

        symbols = {
            variable["symbol"]
            for variable in parsed.candidates[0].variables
        }
        self.assertEqual(symbols, {"ratio", "spending", "money_spent"})
        self.assertEqual(
            parsed.candidates[0].ascii,
            "ratio = spending / money_spent",
        )

    def test_mixed_case_identifiers_are_canonicalized_before_validation(self):
        item = inventory_with(
            {
                "calculation_id": "CALC_0001",
                "name": "Bond value",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": ["present value", "coupon payment"],
                "operations_mentioned": ["addition"],
                "visual_equation_cue": True,
                "formula_expected": True,
                "reason": "The source states a reusable equation.",
            }
        ).calculations[0]
        payload = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "The source states a reusable equation.",
            "candidates": [
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "bond_value",
                    "name": "Bond value",
                    "ascii": "PV = Coupon + Face_Value",
                    "latex": r"PV = C + F",
                    "derivation_type": "stated",
                    "variables": [
                        {"symbol": "PV", "meaning": "present value", "unit": "$"},
                        {"symbol": "Coupon", "meaning": "coupon payment", "unit": "$"},
                        {"symbol": "Face_Value", "meaning": "face value", "unit": "$"},
                    ],
                    "derivation_steps": ["Add coupon payment and face value."],
                    "source_claims": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "relationship": "Present value equals coupon plus face value.",
                        }
                    ],
                }
            ],
        }

        parsed = parse_formula_extraction_response_with_variable_completion(
            payload,
            item=item,
        )

        candidate = parsed.candidates[0]
        self.assertEqual(candidate.ascii, "pv = coupon + face_value")
        self.assertEqual(
            {variable["symbol"] for variable in candidate.variables},
            {"pv", "coupon", "face_value"},
        )
        self.assertEqual(payload["candidates"][0]["ascii"], "PV = Coupon + Face_Value")

    def test_numeric_leading_identifier_is_prefixed_before_validation(self):
        item = inventory_with(
            {
                "calculation_id": "CALC_0001",
                "name": "Scaled value",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": ["scaled value", "123 units"],
                "operations_mentioned": ["multiplication"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states a reusable multiplication.",
            }
        ).calculations[0]
        payload = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "The source states a reusable equation.",
            "candidates": [
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "scaled_value",
                    "name": "Scaled value",
                    "ascii": "scaled_value = 123_units * multiplier",
                    "latex": r"s = 123u \times m",
                    "derivation_type": "stated",
                    "variables": [
                        {
                            "symbol": "scaled_value",
                            "meaning": "scaled value",
                            "unit": "units",
                        },
                        {
                            "symbol": "123_units",
                            "meaning": "123 units",
                            "unit": "units",
                        },
                        {
                            "symbol": "multiplier",
                            "meaning": "multiplier",
                            "unit": "",
                        },
                    ],
                    "derivation_steps": [
                        "Multiply 123 units by the multiplier."
                    ],
                    "source_claims": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "relationship": (
                                "Scaled value equals 123 units times the multiplier."
                            ),
                        }
                    ],
                }
            ],
        }

        parsed = parse_formula_extraction_response_with_variable_completion(
            payload,
            item=item,
        )

        candidate = parsed.candidates[0]
        self.assertEqual(
            candidate.ascii,
            "scaled_value = value_123_units * multiplier",
        )
        self.assertEqual(
            {variable["symbol"] for variable in candidate.variables},
            {"scaled_value", "value_123_units", "multiplier"},
        )
        self.assertEqual(
            payload["candidates"][0]["ascii"],
            "scaled_value = 123_units * multiplier",
        )

    def test_entailment_operation_evidence_can_use_grounding_hull(self):
        item = inventory_with(
            {
                "calculation_id": "CALC_0001",
                "name": "Price calculation",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 2,
                "variables_mentioned": ["spending", "quantity sold"],
                "operations_mentioned": ["division"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states division across adjacent segments.",
            }
        ).calculations[0]
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "price",
                "name": "Price",
                "ascii": "price = spending / quantity_sold",
                "latex": r"p=\frac{s}{q}",
                "derivation_type": "stated",
                "variables": [
                    {"symbol": "price", "meaning": "price", "unit": ""},
                    {"symbol": "spending", "meaning": "spending", "unit": ""},
                    {
                        "symbol": "quantity_sold",
                        "meaning": "quantity sold",
                        "unit": "",
                    },
                ],
                "derivation_steps": [
                    "Divide spending by quantity sold to get price."
                ],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 2,
                        "relationship": "Divide spending by quantity sold for price.",
                    }
                ],
            }
        )
        node = candidate.parsed.operations[0]
        segments = [
            {"text": "If you divide the amount of spending"},
            {"text": "across the transaction"},
            {"text": "by the quantity sold, you get the price."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "formula_id": "price",
            "nodes": [
                {
                    "node_id": node.node_id,
                    "expression": node.expression,
                    "operation": node.operation,
                    "status": "derived",
                    "evidence": [
                        {
                            "start_segment": 2,
                            "end_segment": 2,
                            "quote": "by the quantity sold, you get the price.",
                        }
                    ],
                    "identifier_groundings": [
                        {
                            "identifier": "spending",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "amount of spending",
                        },
                        {
                            "identifier": "quantity_sold",
                            "start_segment": 2,
                            "end_segment": 2,
                            "quote": "quantity sold",
                        },
                        {
                            "identifier": "price",
                            "start_segment": 2,
                            "end_segment": 2,
                            "quote": "price",
                        },
                    ],
                    "depends_on_node_ids": [],
                    "derivation_step": (
                        "The source-stated division spans adjacent segments."
                    ),
                }
            ],
        }

        report = validate_entailment_response_with_grounding_hull_repair(
            payload,
            item=item,
            candidate=candidate,
            segments=segments,
        )

        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
