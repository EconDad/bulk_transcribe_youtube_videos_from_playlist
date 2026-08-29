from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.formula_extraction import (
    FormulaExtractionError,
    build_formula_extraction_prompt,
    build_formula_extraction_repair_prompt,
    parse_formula_extraction_response,
)


def make_item(**changes):
    payload = {
        "calculation_id": "CALC_0001",
        "name": "Normalize a measurement",
        "source_mode": "spoken",
        "start_segment": 1,
        "end_segment": 2,
        "variables_mentioned": ["total", "count"],
        "operations_mentioned": ["division"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The speaker states a division.",
    }
    payload.update(changes)
    return CalculationItem.from_mapping(payload)


def candidate_payload():
    return {
        "calculation_id": "CALC_0001",
        "formula_id": "normalized_measurement",
        "name": "Normalized measurement",
        "ascii": "normalized_measurement = total_value / item_count",
        "latex": r"m=\frac{T}{n}",
        "derivation_type": "stated",
        "variables": [
            {
                "symbol": "normalized_measurement",
                "meaning": "Normalized measurement",
                "unit": "units per item",
            },
            {
                "symbol": "total_value",
                "meaning": "Total value",
                "unit": "units",
            },
            {
                "symbol": "item_count",
                "meaning": "Item count",
                "unit": "items",
            },
        ],
        "derivation_steps": [
            "Divide the total value by the item count."
        ],
        "source_claims": [
            {
                "start_segment": 1,
                "end_segment": 2,
                "relationship": "division",
            }
        ],
    }


class FormulaExtractionTests(unittest.TestCase):
    def test_accepts_valid_candidate_response(self):
        result = parse_formula_extraction_response(
            {
                "calculation_id": "CALC_0001",
                "disposition": "candidates_proposed",
                "reason": "The source states the relationship.",
                "candidates": [candidate_payload()],
            },
            item=make_item(),
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].parsed.operations[0].operation,
            "division",
        )

    def test_rejects_claim_outside_inventory_window(self):
        candidate = candidate_payload()
        candidate["source_claims"][0]["start_segment"] = 0
        with self.assertRaisesRegex(
            FormulaExtractionError,
            "outside its inventory item",
        ):
            parse_formula_extraction_response(
                {
                    "calculation_id": "CALC_0001",
                    "disposition": "candidates_proposed",
                    "reason": "Proposed.",
                    "candidates": [candidate],
                },
                item=make_item(),
            )

    def test_visual_review_requires_visual_cue(self):
        with self.assertRaisesRegex(
            FormulaExtractionError,
            "requires a visual equation cue",
        ):
            parse_formula_extraction_response(
                {
                    "calculation_id": "CALC_0001",
                    "disposition": "visual_review_required",
                    "reason": "Equation is on screen.",
                    "candidates": [],
                },
                item=make_item(),
            )

    def test_prompt_contains_no_subject_specific_formula(self):
        prompt = build_formula_extraction_prompt(
            item=make_item(),
            segments=[
                {"text": "Context."},
                {"text": "Take the total."},
                {"text": "Divide it by the count."},
            ],
        )
        self.assertIn("Do not inject a textbook formula", prompt)
        self.assertNotIn("coupon", prompt.lower())
        self.assertNotIn("yield to maturity", prompt.lower())

    def test_repair_prompt_forbids_numeric_leading_identifiers(self):
        prompt = build_formula_extraction_repair_prompt(
            item=make_item(),
            segments=[
                {"text": "Context."},
                {"text": "The source starts with 204 units."},
                {"text": "Ten times that gives the result."},
            ],
            invalid_payload={
                "calculation_id": "CALC_0001",
                "disposition": "candidates_proposed",
                "reason": "Proposed.",
                "candidates": [
                    {
                        "ascii": "result = 123_units * 10",
                        "variables": [{"symbol": "123_units"}],
                    }
                ],
            },
            validation_error=(
                "variables[0].symbol must be one snake_case identifier"
            ),
        )

        self.assertIn("Identifiers must begin with a lowercase letter", prompt)
        self.assertIn("numeric literal", prompt)
        self.assertIn("never emit a symbol such as 123_units", prompt)


if __name__ == "__main__":
    unittest.main()
