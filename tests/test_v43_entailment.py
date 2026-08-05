from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.entailment import (
    EntailmentValidationError,
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


class EntailmentTests(unittest.TestCase):
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
            {"text": "Then divide that total by the item count."},
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
                                    "Then divide that total by the item count."
                                ),
                            }
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

    def test_non_exact_quote_fails(self):
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
                            "depends_on_node_ids": [],
                            "derivation_step": (
                                "Derive the expression."
                            ),
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

if __name__ == "__main__":
    unittest.main()
