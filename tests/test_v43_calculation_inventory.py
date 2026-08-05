from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import (
    CalculationInventory,
    InventoryValidationError,
    build_inventory_prompt,
    parse_inventory_response,
)


def inventory_payload():
    return {
        "schema_version": "1.0",
        "video_id": "video-123",
        "calculations": [
            {
                "calculation_id": "CALC_0001",
                "name": "Normalize a measurement",
                "source_mode": "spoken",
                "start_segment": 2,
                "end_segment": 4,
                "variables_mentioned": [
                    "total measurement",
                    "number of observations",
                ],
                "operations_mentioned": ["division"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": (
                    "The speaker explicitly says to divide the total "
                    "by the number of observations."
                ),
            },
            {
                "calculation_id": "CALC_0002",
                "name": "Displayed equation",
                "source_mode": "visual_cue",
                "start_segment": 8,
                "end_segment": 8,
                "variables_mentioned": [],
                "operations_mentioned": [],
                "visual_equation_cue": True,
                "formula_expected": True,
                "reason": "The speaker announces an equation on screen.",
            },
        ],
    }


class CalculationInventoryTests(unittest.TestCase):
    def test_valid_inventory(self):
        inventory = CalculationInventory.from_mapping(
            inventory_payload()
        )
        self.assertEqual(len(inventory.calculations), 2)

    def test_response_video_id_and_range(self):
        parsed = parse_inventory_response(
            json.dumps(inventory_payload()),
            expected_video_id="video-123",
            maximum_segment=10,
        )
        self.assertEqual(parsed.video_id, "video-123")

    def test_rejects_out_of_order_inventory(self):
        payload = inventory_payload()
        payload["calculations"].reverse()
        with self.assertRaisesRegex(
            InventoryValidationError,
            "transcript progression",
        ):
            CalculationInventory.from_mapping(payload)

    def test_rejects_visual_mode_without_visual_cue(self):
        payload = inventory_payload()
        payload["calculations"][1]["visual_equation_cue"] = False
        with self.assertRaisesRegex(
            InventoryValidationError,
            "requires visual_equation_cue",
        ):
            CalculationInventory.from_mapping(payload)

    def test_rejects_segment_outside_source(self):
        with self.assertRaisesRegex(
            InventoryValidationError,
            "exceeds the source",
        ):
            parse_inventory_response(
                json.dumps(inventory_payload()),
                expected_video_id="video-123",
                maximum_segment=7,
            )

    def test_prompt_is_domain_neutral(self):
        prompt = build_inventory_prompt(
            video_id="video-123",
            segments=[
                {
                    "segment_id": 4,
                    "text": "Divide the total by the count.",
                }
            ],
        )
        self.assertIn("calculation event", prompt)
        self.assertIn("Do not invent textbook formulas", prompt)
        self.assertNotIn("coupon", prompt.lower())
        self.assertNotIn("yield to maturity", prompt.lower())


if __name__ == "__main__":
    unittest.main()
