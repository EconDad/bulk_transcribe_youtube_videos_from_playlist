from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.entailment import (
    build_entailment_repair_prompt,
    EntailmentValidationError,
    build_entailment_prompt,
    validate_entailment_response,
)
from research_v43.expression_ast import FormulaCandidate


def make_item():
    return CalculationItem.from_mapping(
        {
            "calculation_id": "CALC_0001",
            "name": "Normalize a measurement",
            "source_mode": "spoken",
            "start_segment": 0,
            "end_segment": 1,
            "variables_mentioned": ["first", "second", "count"],
            "operations_mentioned": ["addition", "division"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "The speaker states two operations.",
        }
    )


def make_candidate(ascii_formula="result = total_value / item_count"):
    if ascii_formula == "result = total_value / item_count":
        variables = [
            {"symbol": "result", "meaning": "result", "unit": ""},
            {
                "symbol": "total_value",
                "meaning": "total value",
                "unit": "units",
            },
            {
                "symbol": "item_count",
                "meaning": "item count",
                "unit": "items",
            },
        ]
    else:
        variables = [
            {"symbol": "result", "meaning": "result", "unit": ""},
            {
                "symbol": "first_value",
                "meaning": "first value",
                "unit": "units",
            },
            {
                "symbol": "second_value",
                "meaning": "second value",
                "unit": "units",
            },
            {
                "symbol": "item_count",
                "meaning": "item count",
                "unit": "items",
            },
        ]
    return FormulaCandidate.from_mapping(
        {
            "calculation_id": "CALC_0001",
            "formula_id": "normalized_measurement",
            "name": "Normalized measurement",
            "ascii": ascii_formula,
            "latex": "x",
            "derivation_type": "stated",
            "variables": variables,
            "derivation_steps": ["Apply the stated operations."],
            "source_claims": [
                {
                    "start_segment": 0,
                    "end_segment": 1,
                    "relationship": "arithmetic",
                }
            ],
        }
    )


def grounding(identifier, segment, quote):
    return {
        "identifier": identifier,
        "start_segment": segment,
        "end_segment": segment,
        "quote": quote,
    }


class EntailmentTests(unittest.TestCase):
    def test_repair_prompt_requires_literal_replacement_quotes(self):
        prompt = build_entailment_repair_prompt(
            item=make_item(),
            candidate=make_candidate(),
            segments=[
                {"text": "Add the first and second values."},
                {"text": "Divide that total by the count."},
            ],
            invalid_payload={"nodes": []},
            validation_issues=[
                "NODE_0001 grounding quote for first is not present in cited segments"
            ],
        )

        self.assertIn("mandatory edit", prompt)
        self.assertIn("Never repeat a quote", prompt)
        self.assertIn("literal, verbatim substring", prompt)

    def test_direct_single_node_entailment_passes(self):
        segments = [
            {"text": "Divide the total value by the item count."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "Divide the total value by the item count."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "total value"),
                            grounding("item_count", 0, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertTrue(report.passed, report.issues)

    def test_multi_operator_nodes_can_use_dependencies(self):
        segments = [
            {"text": "Add the first value and the second value."},
            {
                "text": (
                    "Then divide that total by the item count to get "
                    "the result."
                )
            },
        ]
        candidate = make_candidate(
            "result = (first_value + second_value) / item_count"
        )
        first, second = candidate.parsed.operations
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": first.node_id,
                        "expression": first.expression,
                        "operation": first.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "Add the first value and the second value."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("first_value", 0, "first value"),
                            grounding("second_value", 0, "second value"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    },
                    {
                        "node_id": second.node_id,
                        "expression": second.expression,
                        "operation": second.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 1,
                                "end_segment": 1,
                                "quote": (
                                    "Then divide that total by the item count "
                                    "to get the result."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("item_count", 1, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [first.node_id],
                        "derivation_step": "",
                    },
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertTrue(report.passed, report.issues)

    def test_missing_operation_cue_fails(self):
        segments = [
            {"text": "The total value and item count are shown."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "The total value and item count are shown."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "total value"),
                            grounding("item_count", 0, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("lacks a cue" in issue for issue in report.issues)
        )

    def test_percent_of_is_a_multiplication_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertTrue(
            _has_operation_cue(
                "multiplication",
                "Equity is three and a half percent of market price.",
            )
        )
        self.assertTrue(
            _has_operation_cue(
                "multiplication",
                "The amount is 10% of the total.",
            )
        )

    def test_unrelated_percent_language_is_not_a_multiplication_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertFalse(
            _has_operation_cue(
                "multiplication",
                "The percentage increased during the year.",
            )
        )
        self.assertFalse(
            _has_operation_cue(
                "multiplication",
                "The investment produced a 10 percent return.",
            )
        )

    def test_amount_removal_language_is_a_subtraction_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertTrue(
            _has_operation_cue(
                "subtraction",
                "We had $20 go straight out of the $100 right off the top.",
            )
        )
        self.assertTrue(
            _has_operation_cue(
                "subtraction",
                "We take $10 out for taxes and that leaves $20.",
            )
        )

    def test_unquantified_removal_language_is_not_a_subtraction_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertFalse(
            _has_operation_cue("subtraction", "We went out for dinner.")
        )
        self.assertFalse(
            _has_operation_cue("subtraction", "Please take out the trash.")
        )

    def test_amount_from_base_percentage_is_a_division_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertTrue(
            _has_operation_cue(
                "division",
                "Fifty dollars from the thousand dollar par value is five percent.",
            )
        )

    def test_non_numeric_from_phrase_is_not_a_division_cue(self):
        from research_v43.entailment import _has_operation_cue

        self.assertFalse(
            _has_operation_cue(
                "division",
                "The report from the committee is complete.",
            )
        )

    def test_non_exact_operation_quote_fails(self):
        segments = [
            {"text": "Divide the total value by the item count."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": "Divide a different value.",
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "total value"),
                            grounding("item_count", 0, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("quote is not present" in issue for issue in report.issues)
        )

    def test_identifier_grounding_can_use_source_paraphrase(self):
        sentence = (
            "The offered amount is $30, which is $10 less than what the "
            "analyst thought the asset was worth."
        )
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Difference example",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": ["$30", "$10"],
                "operations_mentioned": ["subtraction"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states a less-than relationship.",
            }
        )
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "amount_difference",
                "name": "Amount difference",
                "ascii": "difference = reference_amount - offered_amount",
                "latex": "d=r-o",
                "derivation_type": "stated",
                "variables": [
                    {
                        "symbol": "difference",
                        "meaning": "difference between amounts",
                        "unit": "USD",
                    },
                    {
                        "symbol": "reference_amount",
                        "meaning": "analyst's estimated worth",
                        "unit": "USD",
                    },
                    {
                        "symbol": "offered_amount",
                        "meaning": "offered amount",
                        "unit": "USD",
                    },
                ],
                "derivation_steps": [
                    "Rearrange the directly stated less-than relationship."
                ],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 0,
                        "relationship": (
                            "offered_amount = reference_amount - difference"
                        ),
                    }
                ],
            }
        )
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "amount_difference",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "derived",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": sentence,
                            }
                        ],
                        "identifier_groundings": [
                            grounding(
                                "reference_amount",
                                0,
                                "what the analyst thought the asset was worth",
                            ),
                            grounding("offered_amount", 0, "$30"),
                            grounding("difference", 0, "$10 less"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": (
                            "Rearrange offered_amount = reference_amount "
                            "- difference to solve for difference."
                        ),
                    }
                ],
            },
            item=item,
            candidate=candidate,
            segments=[{"text": sentence}],
        )
        self.assertTrue(report.passed, report.issues)

    def test_missing_identifier_grounding_fails(self):
        segments = [
            {"text": "Divide the total value by the item count."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "Divide the total value by the item count."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "total value"),
                            grounding("item_count", 0, "item count"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("result" in issue for issue in report.issues)
        )

    def test_non_exact_identifier_grounding_quote_fails(self):
        segments = [
            {"text": "Divide the total value by the item count."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "entailed",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "Divide the total value by the item count."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "different total"),
                            grounding("item_count", 0, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": "",
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any(
                "grounding quote for total_value" in issue
                for issue in report.issues
            )
        )

    def test_single_derived_node_can_use_direct_evidence(self):
        segments = [
            {"text": "Divide the total value by the item count."},
            {"text": "That gives the result."},
        ]
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "normalized_measurement",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "derived",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": (
                                    "Divide the total value by the item count."
                                ),
                            }
                        ],
                        "identifier_groundings": [
                            grounding("total_value", 0, "total value"),
                            grounding("item_count", 0, "item count"),
                            grounding("result", 1, "result"),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": (
                            "Normalize the directly stated relationship "
                            "as a symbolic equation."
                        ),
                    }
                ],
            },
            item=make_item(),
            candidate=candidate,
            segments=segments,
        )
        self.assertTrue(report.passed, report.issues)

    def test_derived_node_requires_dependency_or_evidence(self):
        candidate = make_candidate()
        node = candidate.parsed.operations[0]
        with self.assertRaisesRegex(
            EntailmentValidationError,
            "dependencies or evidence",
        ):
            validate_entailment_response(
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "normalized_measurement",
                    "nodes": [
                        {
                            "node_id": node.node_id,
                            "expression": node.expression,
                            "operation": node.operation,
                            "status": "derived",
                            "evidence": [],
                            "identifier_groundings": [],
                            "depends_on_node_ids": [],
                            "derivation_step": "Derive the expression.",
                        }
                    ],
                },
                item=make_item(),
                candidate=candidate,
                segments=[
                    {
                        "text": (
                            "Divide the total value by the item count."
                        )
                    },
                    {"text": "That gives the result."},
                ],
            )



    def test_prompt_lists_root_result_identifier_explicitly(self):
        sentence = (
            "The observed amount is 7, which is 3 fewer than "
            "the reference amount."
        )
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Difference example",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": [
                    "observed amount",
                    "reference amount",
                    "3",
                ],
                "operations_mentioned": ["subtraction"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": (
                    "The source states a fewer-than relationship."
                ),
            }
        )
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "amount_difference",
                "name": "Amount difference",
                "ascii": (
                    "difference = "
                    "reference_amount - observed_amount"
                ),
                "latex": "d = r - o",
                "derivation_type": "derived",
                "variables": [
                    {
                        "symbol": "difference",
                        "meaning": "difference between the values",
                        "unit": "units",
                    },
                    {
                        "symbol": "reference_amount",
                        "meaning": "reference amount",
                        "unit": "units",
                    },
                    {
                        "symbol": "observed_amount",
                        "meaning": "observed amount",
                        "unit": "units",
                    },
                ],
                "derivation_steps": [
                    (
                        "Rearrange the source-stated relationship "
                        "to solve for the difference."
                    )
                ],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 0,
                        "relationship": (
                            "observed_amount = "
                            "reference_amount - difference"
                        ),
                    }
                ],
            }
        )

        prompt = build_entailment_prompt(
            item=item,
            candidate=candidate,
            segments=[{"text": sentence}],
        )

        self.assertIn(
            '"left_hand_result_identifier": "difference"',
            prompt,
        )
        self.assertIn(
            '"required_identifier_groundings"',
            prompt,
        )
        self.assertIn('"difference"', prompt)
        self.assertIn("spoken result amount", prompt)
        self.assertIn("3 fewer than", prompt)

    def test_result_can_be_grounded_by_fewer_than_phrase(self):
        sentence = (
            "The observed amount is 7, which is 3 fewer than "
            "the reference amount."
        )
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Difference example",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": [
                    "observed amount",
                    "reference amount",
                    "3",
                ],
                "operations_mentioned": ["subtraction"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": (
                    "The source states a fewer-than relationship."
                ),
            }
        )
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "amount_difference",
                "name": "Amount difference",
                "ascii": (
                    "difference = "
                    "reference_amount - observed_amount"
                ),
                "latex": "d = r - o",
                "derivation_type": "derived",
                "variables": [
                    {
                        "symbol": "difference",
                        "meaning": "difference between the values",
                        "unit": "units",
                    },
                    {
                        "symbol": "reference_amount",
                        "meaning": "reference amount",
                        "unit": "units",
                    },
                    {
                        "symbol": "observed_amount",
                        "meaning": "observed amount",
                        "unit": "units",
                    },
                ],
                "derivation_steps": [
                    (
                        "Rearrange observed_amount = "
                        "reference_amount - difference."
                    )
                ],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 0,
                        "relationship": (
                            "observed_amount = "
                            "reference_amount - difference"
                        ),
                    }
                ],
            }
        )
        node = candidate.parsed.operations[0]

        report = validate_entailment_response(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "amount_difference",
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "expression": node.expression,
                        "operation": node.operation,
                        "status": "derived",
                        "evidence": [
                            {
                                "start_segment": 0,
                                "end_segment": 0,
                                "quote": sentence,
                            }
                        ],
                        "identifier_groundings": [
                            grounding(
                                "reference_amount",
                                0,
                                "reference amount",
                            ),
                            grounding(
                                "observed_amount",
                                0,
                                "observed amount",
                            ),
                            grounding(
                                "difference",
                                0,
                                "3 fewer than",
                            ),
                        ],
                        "depends_on_node_ids": [],
                        "derivation_step": (
                            "Rearrange observed_amount = "
                            "reference_amount - difference "
                            "to solve for difference."
                        ),
                    }
                ],
            },
            item=item,
            candidate=candidate,
            segments=[{"text": sentence}],
        )

        self.assertTrue(report.passed, report.issues)

if __name__ == "__main__":
    unittest.main()
