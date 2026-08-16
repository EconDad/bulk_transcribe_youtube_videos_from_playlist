from __future__ import annotations

import unittest

from research_v43.finalization import FinalizationError
from research_v43.narrative_localization import localize_narrative_extraction


class NarrativeLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": float(i), "end": float(i + 1), "text": text}
            for i, text in enumerate(
                [
                    "The introduction explains why patient investing matters.",
                    "The speaker contrasts market price with estimated business value.",
                    "He says investors should understand the business before buying.",
                    "A temporary market decline can create an opportunity.",
                    "The example uses a 10 percent discount to illustrate the gap.",
                    "The speaker says the discount is only an example.",
                    "He returns to the importance of discipline.",
                    "Patience helps investors wait for an attractive price.",
                    "The conclusion connects discipline and patience.",
                    "The lesson ends with a reminder to stay consistent.",
                ]
            )
        ]

    def test_broad_range_is_localized_to_supported_window(self):
        payload = {
            "evidence": [
                {
                    "topic": "Discount example",
                    "text": "The example uses a 10 percent discount.",
                    "explanation": "The figure illustrates the gap between price and estimated value.",
                    "start_segment": 0,
                    "end_segment": 9,
                }
            ]
        }
        repairs: list[str] = []
        result = localize_narrative_extraction(
            payload,
            segments=self.segments,
            minimum_segment=0,
            maximum_segment=9,
            on_repair=repairs.append,
        )
        self.assertEqual(len(result), 1)
        self.assertLessEqual(
            result[0]["end_segment"] - result[0]["start_segment"] + 1,
            6,
        )
        self.assertLessEqual(result[0]["start_segment"], 4)
        self.assertGreaterEqual(result[0]["end_segment"], 4)
        self.assertTrue(repairs)

    def test_narrow_range_is_preserved(self):
        payload = {
            "evidence": [
                {
                    "topic": "Patience",
                    "text": "Patience helps investors wait for an attractive price.",
                    "explanation": "The speaker connects patience to buying discipline.",
                    "start_segment": 7,
                    "end_segment": 8,
                }
            ]
        }
        result = localize_narrative_extraction(
            payload,
            segments=self.segments,
            minimum_segment=0,
            maximum_segment=9,
        )
        self.assertEqual(result[0]["start_segment"], 7)
        self.assertEqual(result[0]["end_segment"], 8)

    def test_localization_preserves_numeric_grounding(self):
        payload = {
            "evidence": [
                {
                    "topic": "Discount example",
                    "text": "The example uses a 20 percent discount.",
                    "explanation": "The number illustrates the valuation gap.",
                    "start_segment": 0,
                    "end_segment": 9,
                }
            ]
        }
        with self.assertRaises(FinalizationError):
            localize_narrative_extraction(
                payload,
                segments=self.segments,
                minimum_segment=0,
                maximum_segment=9,
            )

    def test_localization_never_expands_model_range(self):
        payload = {
            "evidence": [
                {
                    "topic": "Patience",
                    "text": "Patience helps investors wait for an attractive price.",
                    "explanation": "This point appears near the conclusion.",
                    "start_segment": 0,
                    "end_segment": 6,
                }
            ]
        }
        with self.assertRaises(FinalizationError):
            localize_narrative_extraction(
                payload,
                segments=self.segments,
                minimum_segment=0,
                maximum_segment=9,
            )


if __name__ == "__main__":
    unittest.main()
