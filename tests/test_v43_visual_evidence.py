from __future__ import annotations

import unittest

from research_v43.expression_ast import parse_formula
from research_v43.visual_evidence import (
    normalize_visual_formula_ascii,
    normalize_visual_symbol,
    select_visual_consensus,
)


class VisualEvidenceTests(unittest.TestCase):
    def test_visual_normalization_is_domain_neutral_and_parser_safe(self):
        raw = (
            "B_0 = C/2 * [1 - (1 + YTM/2)^(-2t)] "
            "/ (YTM/2) + F / (1 + YTM/2)^(2t)"
        )
        parsed = parse_formula(normalize_visual_formula_ascii(raw))
        self.assertEqual(parsed.left_symbol, "b_0")
        self.assertEqual(parsed.identifiers, frozenset({"b_0", "c", "ytm", "t", "f"}))
        self.assertIn("2 * t", parsed.canonical_ascii)
        self.assertIn("-2 * t", parsed.canonical_ascii)
        operations = {item.operation for item in parsed.operations}
        self.assertTrue({"division", "addition", "subtraction", "exponentiation"} <= operations)

    def test_unicode_subscript_symbol_normalization(self):
        self.assertEqual(normalize_visual_symbol("B₀"), "b_0")

    def test_consensus_requires_clean_strict_majority(self):
        records = [
            {
                "status": "ok",
                "parsed_canonical_ascii": expression,
                "result": {"equation_present": True, "uncertain_tokens": []},
            }
            for expression in ("x = a / b", "x = a / b", "x = a / b", "x = a * b")
        ]
        consensus = select_visual_consensus(records, min_consensus=3)
        self.assertTrue(consensus["passed"])
        self.assertEqual(consensus["winner_count"], 3)
        self.assertEqual(consensus["winner_canonical_ascii"], "x = a / b")

    def test_uncertain_frame_does_not_vote(self):
        records = [
            {
                "status": "ok",
                "parsed_canonical_ascii": "x = a / b",
                "result": {"equation_present": True, "uncertain_tokens": uncertain},
            }
            for uncertain in ([], ["operator"], [])
        ]
        consensus = select_visual_consensus(records, min_consensus=3)
        self.assertFalse(consensus["passed"])
        self.assertEqual(consensus["eligible_frames"], 2)

    def test_tied_ast_groups_fail_closed(self):
        records = [
            {
                "status": "ok",
                "parsed_canonical_ascii": expression,
                "result": {"equation_present": True, "uncertain_tokens": []},
            }
            for expression in ("x = a / b", "x = a / b", "x = a * b", "x = a * b")
        ]
        self.assertFalse(select_visual_consensus(records, min_consensus=2)["passed"])


if __name__ == "__main__":
    unittest.main()
