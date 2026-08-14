import unittest

from research_v43.coverage import (
    CoverageValidationError,
    _formula_key,
    _index_formulas,
)


class CalculationScopedFormulaIdentityTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "CALC_0001": object(),
            "CALC_0002": object(),
            "CALC_0019": object(),
        }

    def test_same_formula_id_allowed_across_calculations(self):
        formulas = [
            {"calculation_id": "CALC_0001", "formula_id": "ratio"},
            {"calculation_id": "CALC_0002", "formula_id": "ratio"},
            {"calculation_id": "CALC_0019", "formula_id": "ratio"},
        ]
        indexed = _index_formulas(formulas, self.inventory)
        self.assertEqual(
            set(indexed),
            {
                ("CALC_0001", "ratio"),
                ("CALC_0002", "ratio"),
                ("CALC_0019", "ratio"),
            },
        )

    def test_same_formula_id_rejected_within_one_calculation(self):
        formulas = [
            {"calculation_id": "CALC_0001", "formula_id": "ratio"},
            {"calculation_id": "CALC_0001", "formula_id": "ratio"},
        ]
        with self.assertRaisesRegex(
            CoverageValidationError,
            "Duplicate formula_id within calculation CALC_0001: ratio",
        ):
            _index_formulas(formulas, self.inventory)

    def test_formula_key_is_calculation_scoped(self):
        self.assertNotEqual(
            _formula_key("CALC_0001", "ratio"),
            _formula_key("CALC_0002", "ratio"),
        )

    def test_unknown_calculation_still_rejected(self):
        with self.assertRaisesRegex(
            CoverageValidationError,
            "unknown calculation CALC_9999",
        ):
            _index_formulas(
                [{"calculation_id": "CALC_9999", "formula_id": "ratio"}],
                self.inventory,
            )


if __name__ == "__main__":
    unittest.main()
