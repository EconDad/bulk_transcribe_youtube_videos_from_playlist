from __future__ import annotations

import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.entailment import _has_operation_cue
from research_v43.inventory_evidence_audit import (
    AuditAction,
    _parse_audit_action,
    _variable_appears,
    find_deterministic_expansion,
)


def make_item(**updates):
    raw = {
        "calculation_id": "CALC_0001",
        "name": "Arithmetic example",
        "source_mode": "spoken",
        "start_segment": 4,
        "end_segment": 4,
        "variables_mentioned": ["$1,200", "$1,000"],
        "operations_mentioned": ["subtraction"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The source describes an arithmetic relationship.",
    }
    raw.update(updates)
    return CalculationItem.from_mapping(raw)


class MatcherHardeningTests(unittest.TestCase):
    def test_numeric_literals_do_not_suffix_match(self):
        self.assertFalse(_variable_appears("$1,200", "$200"))
        self.assertTrue(_variable_appears("$1,200", "He paid $1,200."))
        self.assertTrue(_variable_appears("$1,000", "He gets $1,000 back."))
        self.assertFalse(_variable_appears("$1,000", "The loss is $200."))
        self.assertFalse(_variable_appears("$50", "$1,050"))
        self.assertTrue(_variable_appears("5%", "The rate is 5%."))
        self.assertTrue(
            _variable_appears("60 coupons", "He receives 60 coupons.")
        )

    def test_operation_cues_require_word_boundaries(self):
        self.assertFalse(
            _has_operation_cue(
                "subtraction",
                "regardless of how long he holds it",
            )
        )
        self.assertTrue(
            _has_operation_cue(
                "subtraction",
                "the difference between the two amounts",
            )
        )
        self.assertTrue(
            _has_operation_cue("subtraction", "he is going to lose $200")
        )
        self.assertTrue(
            _has_operation_cue("subtraction", "that is a $200 profit")
        )
        self.assertTrue(
            _has_operation_cue("division", "we are dividing it by the price")
        )
        self.assertFalse(
            _has_operation_cue("division", "the percentage increased")
        )
        self.assertFalse(
            _has_operation_cue("division", "performance improved")
        )

    def test_deterministic_search_refuses_far_evidence(self):
        item = make_item(
            start_segment=5,
            end_segment=5,
            variables_mentioned=["$50", "$800"],
            operations_mentioned=["division"],
        )
        segments = [
            {"text": "Noise."},
            {"text": "Divide $50 by a prior price."},
            {"text": "Noise."},
            {"text": "Noise."},
            {"text": "The new price is $800."},
            {"text": "The resulting percentage is 6.3%."},
        ]
        decision = find_deterministic_expansion(
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=5,
            max_auto_distance=3,
        )
        self.assertIsNone(decision)

    def test_deterministic_search_accepts_local_exact_evidence(self):
        item = make_item()
        segments = [
            {"text": "Noise."},
            {"text": "Noise."},
            {"text": "He paid $1,200."},
            {"text": "Noise."},
            {"text": "He will lose $200."},
            {"text": "He gets $1,000 back."},
        ]
        decision = find_deterministic_expansion(
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=5,
            max_auto_distance=3,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.evidence_segment_ids, (2, 4, 5))

    def test_unambiguous_action_typos_are_normalized(self):
        self.assertEqual(
            _parse_audit_action("dowgrade_non_symbolic"),
            AuditAction.DOWNGRADE_NON_SYMBOLIC,
        )
        self.assertEqual(
            _parse_audit_action("downdgrade_non_symbolic"),
            AuditAction.DOWNGRADE_NON_SYMBOLIC,
        )
        self.assertEqual(_parse_audit_action("expand"), AuditAction.EXPAND)


if __name__ == "__main__":
    unittest.main()
