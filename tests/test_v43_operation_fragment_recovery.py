from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationInventory
from research_v43.operation_fragment_recovery import (
    audit_with_incomplete_operation_fragment_downgrade,
)


def inventory_with(item):
    return CalculationInventory.from_mapping(
        {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [item],
        }
    )


class OperationFragmentRecoveryTests(unittest.TestCase):
    def test_incomplete_division_fragment_is_downgraded(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Spending-to-Money Ratio",
            "source_mode": "spoken",
            "start_segment": 0,
            "end_segment": 0,
            "variables_mentioned": [
                "amount of spending",
                "amount of money spent",
            ],
            "operations_mentioned": ["division"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "Inventory inferred a ratio result.",
        }
        segments = [
            {
                "text": (
                    "If you divide the amount of spending by the amount "
                    "of money spent,"
                )
            }
        ]

        audited, records = audit_with_incomplete_operation_fragment_downgrade(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertFalse(audited.calculations[0].formula_expected)
        self.assertEqual(
            records[-1]["action"],
            "downgrade_non_symbolic_incomplete_operation_fragment",
        )

    def test_named_result_keeps_formula_expected(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Price Calculation",
            "source_mode": "spoken",
            "start_segment": 0,
            "end_segment": 0,
            "variables_mentioned": ["spending", "quantity sold"],
            "operations_mentioned": ["division"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "The source states a reusable division result.",
        }
        segments = [
            {
                "text": (
                    "Divide spending by the quantity sold, and you get price."
                )
            }
        ]

        audited, records = audit_with_incomplete_operation_fragment_downgrade(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertTrue(audited.calculations[0].formula_expected)
        self.assertFalse(
            any(
                record.get("action")
                == "downgrade_non_symbolic_incomplete_operation_fragment"
                for record in records
            )
        )

    def test_complete_clause_without_result_cue_is_not_downgraded(self):
        item = {
            "calculation_id": "CALC_0001",
            "name": "Normalized Measure",
            "source_mode": "spoken",
            "start_segment": 0,
            "end_segment": 0,
            "variables_mentioned": ["total", "count"],
            "operations_mentioned": ["division"],
            "visual_equation_cue": False,
            "formula_expected": True,
            "reason": "Inventory found division.",
        }
        segments = [
            {"text": "Divide the total by the count."}
        ]

        audited, _ = audit_with_incomplete_operation_fragment_downgrade(
            inventory=inventory_with(item),
            segments=segments,
        )

        self.assertTrue(audited.calculations[0].formula_expected)


if __name__ == "__main__":
    unittest.main()
