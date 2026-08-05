from __future__ import annotations

import unittest

from research_v43.calculation_inventory import (
    CalculationInventory,
)
from research_v43.coverage import (
    CoverageState,
    CoverageValidationError,
    reconcile_coverage,
)


def make_inventory(
    *,
    first_expected: bool = True,
    second_expected: bool = False,
):
    return CalculationInventory.from_mapping(
        {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                {
                    "calculation_id": "CALC_0001",
                    "name": "First calculation",
                    "source_mode": "spoken",
                    "start_segment": 1,
                    "end_segment": 2,
                    "variables_mentioned": ["total", "count"],
                    "operations_mentioned": ["division"],
                    "visual_equation_cue": False,
                    "formula_expected": first_expected,
                    "reason": "The source describes a division.",
                },
                {
                    "calculation_id": "CALC_0002",
                    "name": "Procedural estimate",
                    "source_mode": "spoken",
                    "start_segment": 4,
                    "end_segment": 5,
                    "variables_mentioned": [],
                    "operations_mentioned": [],
                    "visual_equation_cue": False,
                    "formula_expected": second_expected,
                    "reason": (
                        "The source describes a procedure without a "
                        "symbolic relationship."
                    ),
                },
            ],
        }
    )


class CoverageTests(unittest.TestCase):
    def test_complete_coverage_passes(self):
        report = reconcile_coverage(
            inventory=make_inventory(),
            resolutions=[
                {
                    "calculation_id": "CALC_0001",
                    "state": "formula_retained",
                    "formula_ids": ["normalized_measurement"],
                    "reason": "Validated and retained.",
                },
                {
                    "calculation_id": "CALC_0002",
                    "state": "non_symbolic_calculation",
                    "formula_ids": [],
                    "reason": "No reusable symbolic relationship.",
                },
            ],
            formulas=[
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "normalized_measurement",
                }
            ],
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.unresolved, 0)
        self.assertEqual(report.formulas_retained, 1)

    def test_partial_formula_coverage_fails(self):
        report = reconcile_coverage(
            inventory=make_inventory(
                first_expected=True,
                second_expected=True,
            ),
            resolutions=[
                {
                    "calculation_id": "CALC_0001",
                    "state": "formula_retained",
                    "formula_ids": ["normalized_measurement"],
                    "reason": "Validated and retained.",
                },
                {
                    "calculation_id": "CALC_0002",
                    "state": "formula_rejected",
                    "formula_ids": [],
                    "reason": "Entailment failed.",
                },
            ],
            formulas=[
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "normalized_measurement",
                }
            ],
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.unresolved, 0)
        self.assertTrue(
            any("expected a formula" in issue for issue in report.issues)
        )

    def test_missing_resolution_fails(self):
        report = reconcile_coverage(
            inventory=make_inventory(),
            resolutions=[
                {
                    "calculation_id": "CALC_0001",
                    "state": "formula_retained",
                    "formula_ids": ["normalized_measurement"],
                    "reason": "Validated.",
                }
            ],
            formulas=[
                {
                    "calculation_id": "CALC_0001",
                    "formula_id": "normalized_measurement",
                }
            ],
        )
        self.assertFalse(report.passed)
        self.assertIn(
            "CALC_0002 has no coverage resolution",
            report.issues,
        )

    def test_visual_review_blocks_expected_formula(self):
        report = reconcile_coverage(
            inventory=make_inventory(),
            resolutions=[
                {
                    "calculation_id": "CALC_0001",
                    "state": "visual_review_required",
                    "formula_ids": [],
                    "reason": "The equation image needs verification.",
                },
                {
                    "calculation_id": "CALC_0002",
                    "state": "non_symbolic_calculation",
                    "formula_ids": [],
                    "reason": "No formula expected.",
                },
            ],
            formulas=[],
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.visual_review_required, 1)

    def test_rejects_formula_for_wrong_calculation(self):
        report = reconcile_coverage(
            inventory=make_inventory(),
            resolutions=[
                {
                    "calculation_id": "CALC_0001",
                    "state": "formula_retained",
                    "formula_ids": ["formula_one"],
                    "reason": "Retained.",
                },
                {
                    "calculation_id": "CALC_0002",
                    "state": "non_symbolic_calculation",
                    "formula_ids": [],
                    "reason": "No formula expected.",
                },
            ],
            formulas=[
                {
                    "calculation_id": "CALC_0002",
                    "formula_id": "formula_one",
                }
            ],
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("not CALC_0001" in issue for issue in report.issues)
        )

    def test_rejects_duplicate_resolution(self):
        duplicate = {
            "calculation_id": "CALC_0001",
            "state": CoverageState.FORMULA_RETAINED.value,
            "formula_ids": ["formula_one"],
            "reason": "Retained.",
        }
        with self.assertRaisesRegex(
            CoverageValidationError,
            "Duplicate coverage resolution",
        ):
            reconcile_coverage(
                inventory=make_inventory(),
                resolutions=[duplicate, duplicate],
                formulas=[
                    {
                        "calculation_id": "CALC_0001",
                        "formula_id": "formula_one",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
