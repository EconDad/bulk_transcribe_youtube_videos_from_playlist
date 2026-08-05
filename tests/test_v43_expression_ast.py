from __future__ import annotations

import unittest

from research_v43.expression_ast import (
    ExpressionValidationError,
    FormulaCandidate,
    parse_formula,
)


class ExpressionAstTests(unittest.TestCase):
    def test_single_operation_formula(self):
        parsed = parse_formula(
            "result = first_value / second_value"
        )
        self.assertEqual(
            parsed.canonical_ascii,
            "result = first_value / second_value",
        )
        self.assertEqual(
            [item.operation for item in parsed.operations],
            ["division"],
        )

    def test_multi_operator_nested_formula(self):
        parsed = parse_formula(
            "result = (income + adjustment) / "
            "((opening_value + closing_value) / 2)"
        )
        self.assertEqual(
            {item.operation for item in parsed.operations},
            {"addition", "division"},
        )
        self.assertEqual(
            parsed.right_identifiers,
            {
                "income",
                "adjustment",
                "opening_value",
                "closing_value",
            },
        )

    def test_exponent_and_unary_minus(self):
        parsed = parse_formula(
            "present_value = future_value / "
            "(1 + rate)^(-periods)"
        )
        operations = {
            item.operation
            for item in parsed.operations
        }
        self.assertIn("exponentiation", operations)
        self.assertIn("unary_minus", operations)

    def test_whitelisted_functions(self):
        parsed = parse_formula(
            "dispersion = sqrt(sum(first_error^2, second_error^2))"
        )
        operations = {
            item.operation
            for item in parsed.operations
        }
        self.assertIn("function:sqrt", operations)
        self.assertIn("function:sum", operations)

    def test_rejects_arbitrary_function(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "Function is not allowed",
        ):
            parse_formula("result = dangerous(value)")

    def test_rejects_attribute_access(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "Only direct calls",
        ):
            parse_formula("result = object.method(value)")

    def test_rejects_indexing(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "Unsupported syntax",
        ):
            parse_formula("result = values[0]")

    def test_rejects_multiple_assignments(self):
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "exactly one",
        ):
            parse_formula("first = second = third")

    def test_formula_candidate_requires_all_variable_definitions(self):
        candidate = {
            "calculation_id": "CALC_0001",
            "formula_id": "average_value",
            "name": "Average value",
            "ascii": "average_value = total_value / item_count",
            "latex": r"\bar{x}=\frac{T}{n}",
            "derivation_type": "derived",
            "variables": [
                {
                    "symbol": "average_value",
                    "meaning": "Average value",
                    "unit": "units",
                },
                {
                    "symbol": "total_value",
                    "meaning": "Sum of values",
                    "unit": "units",
                },
            ],
            "derivation_steps": [
                "Divide the total by the number of items."
            ],
            "source_claims": [
                {
                    "start_segment": 2,
                    "end_segment": 3,
                    "relationship": "division",
                }
            ],
        }
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "Variable definitions do not match",
        ):
            FormulaCandidate.from_mapping(candidate)

    def test_stated_visual_requires_visual_source(self):
        candidate = {
            "calculation_id": "CALC_0001",
            "formula_id": "displayed_relation",
            "name": "Displayed relation",
            "ascii": "output_value = first_value + second_value",
            "latex": "y=a+b",
            "derivation_type": "stated_visual",
            "variables": [
                {
                    "symbol": "output_value",
                    "meaning": "Output",
                    "unit": "",
                },
                {
                    "symbol": "first_value",
                    "meaning": "First input",
                    "unit": "",
                },
                {
                    "symbol": "second_value",
                    "meaning": "Second input",
                    "unit": "",
                },
            ],
            "derivation_steps": [
                "Add the two displayed inputs."
            ],
            "source_claims": [
                {
                    "start_segment": 4,
                    "end_segment": 4,
                    "relationship": "visual equation cue",
                }
            ],
        }
        with self.assertRaisesRegex(
            ExpressionValidationError,
            "require visual_source",
        ):
            FormulaCandidate.from_mapping(candidate)


if __name__ == "__main__":
    unittest.main()
