from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.formula_extraction import (
    FormulaExtractionError,
    normalize_formula_candidate_variables,
    parse_formula_extraction_response,
)


def make_item(**changes):
    payload = {
        "calculation_id": "CALC_0001",
        "name": "Normalize a measurement",
        "source_mode": "spoken",
        "start_segment": 1,
        "end_segment": 2,
        "variables_mentioned": ["result", "amounts"],
        "operations_mentioned": ["division"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The speaker states a division.",
    }
    payload.update(changes)
    return CalculationItem.from_mapping(payload)


def base_candidate():
    return {
        "calculation_id": "CALC_0001",
        "formula_id": "normalized_measurement",
        "name": "Normalized measurement",
        "ascii": "normalized_measurement = numerator / denominator",
        "latex": r"m=\frac{n}{d}",
        "derivation_type": "stated",
        "variables": [
            {
                "symbol": "normalized_measurement",
                "meaning": "Normalized result",
                "unit": "",
            },
            {
                "symbol": "numerator",
                "meaning": "Top quantity",
                "unit": "",
            },
            {
                "symbol": "denominator",
                "meaning": "Bottom quantity",
                "unit": "",
            },
        ],
        "derivation_steps": ["Divide the numerator by the denominator."],
        "source_claims": [
            {
                "start_segment": 1,
                "end_segment": 2,
                "relationship": "division",
            }
        ],
    }


def response(candidate):
    return {
        "calculation_id": "CALC_0001",
        "disposition": "candidates_proposed",
        "reason": "Source-grounded relationship.",
        "candidates": [candidate],
    }


class ExtractionNormalizationTests(unittest.TestCase):
    def test_numeric_literal_definitions_are_removed(self):
        candidate = base_candidate()
        candidate["ascii"] = "normalized_measurement = 50 / 1200"
        candidate["variables"] = [
            {
                "symbol": "normalized_measurement",
                "meaning": "Normalized result",
                "unit": "%",
            },
            {
                "symbol": "annual_input",
                "meaning": "Semantic description of 50",
                "unit": "units",
            },
            {
                "symbol": "market_input",
                "meaning": "Semantic description of 1200",
                "unit": "units",
            },
        ]

        result = parse_formula_extraction_response(
            response(candidate),
            item=make_item(),
        )

        self.assertEqual(
            [item["symbol"] for item in result.candidates[0].variables],
            ["normalized_measurement"],
        )
        self.assertEqual(
            result.candidates[0].ascii,
            "normalized_measurement = 50 / 1200",
        )

    def test_invalid_literal_like_symbols_are_removed_when_not_identifiers(self):
        candidate = base_candidate()
        candidate["ascii"] = "normalized_measurement = 50 / 800"
        candidate["variables"] = [
            {
                "symbol": "normalized_measurement",
                "meaning": "Normalized result",
                "unit": "%",
            },
            {
                "symbol": "$50",
                "meaning": "Literal numerator",
                "unit": "",
            },
            {
                "symbol": "800 dollars",
                "meaning": "Literal denominator",
                "unit": "",
            },
        ]

        normalized = normalize_formula_candidate_variables(candidate)
        self.assertEqual(
            [item["symbol"] for item in normalized["variables"]],
            ["normalized_measurement"],
        )

    def test_required_rhs_identifiers_are_preserved(self):
        candidate = base_candidate()
        result = parse_formula_extraction_response(
            response(candidate),
            item=make_item(),
        )
        self.assertEqual(
            {item["symbol"] for item in result.candidates[0].variables},
            {"normalized_measurement", "numerator", "denominator"},
        )

    def test_identical_duplicate_required_definition_is_deduplicated(self):
        candidate = base_candidate()
        candidate["variables"].append(
            {
                "symbol": " numerator ",
                "meaning": "Top quantity",
                "unit": "",
            }
        )

        result = parse_formula_extraction_response(
            response(candidate),
            item=make_item(),
        )

        symbols = [
            item["symbol"] for item in result.candidates[0].variables
        ]
        self.assertEqual(symbols.count("numerator"), 1)

    def test_conflicting_duplicate_required_definition_is_rejected(self):
        candidate = base_candidate()
        candidate["variables"].append(
            {
                "symbol": "numerator",
                "meaning": "Different meaning",
                "unit": "",
            }
        )

        with self.assertRaisesRegex(
            FormulaExtractionError,
            "Conflicting duplicate variable definition",
        ):
            parse_formula_extraction_response(
                response(candidate),
                item=make_item(),
            )

    def test_missing_required_identifier_is_not_synthesized(self):
        candidate = base_candidate()
        candidate["variables"] = [
            variable
            for variable in candidate["variables"]
            if variable["symbol"] != "denominator"
        ]
        candidate["variables"].append(
            {
                "symbol": "semantic_alias",
                "meaning": "Alias not present in expression",
                "unit": "",
            }
        )

        with self.assertRaisesRegex(
            FormulaExtractionError,
            "missing=.*denominator",
        ):
            parse_formula_extraction_response(
                response(candidate),
                item=make_item(),
            )


if __name__ == "__main__":
    unittest.main()
