from __future__ import annotations

import unittest

from research_v43.finalization import FinalizationError, NarrativeEvidence
from research_v43.synthesis_recovery import recover_synthesis


class SynthesisRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"text": "Opening framework for the lesson."},
            {"text": "The first idea focuses on business quality."},
            {"text": "The second idea focuses on price."},
            {"text": "The example uses a 10 percent discount."},
            {"text": "Discipline matters during the process."},
            {"text": "Patience closes the lesson."},
        ]
        self.evidence = (
            NarrativeEvidence("N0001", "Opening", "Opening framework.", "The lesson begins here.", 0, 0),
            NarrativeEvidence("N0002", "Business", "Business quality matters.", "This is the first idea.", 1, 1),
            NarrativeEvidence("N0003", "Price", "Price matters.", "This is the second idea.", 2, 2),
            NarrativeEvidence("N0004", "Discount", "The example uses a 10 percent discount.", "The example quantifies the discount.", 3, 3),
            NarrativeEvidence("N0005", "Discipline", "Discipline matters.", "The process requires discipline.", 4, 4),
            NarrativeEvidence("N0006", "Patience", "Patience closes the lesson.", "The lesson concludes here.", 5, 5),
        )

    def _base_payload(self):
        return {
            "executive_summary": (
                "The lesson opens with a framework. Business quality matters. "
                "Price matters. Patience closes the lesson."
            ),
            "executive_summary_evidence_ids": ["N0001", "N0002", "N0003", "N0006"],
            "key_takeaways": [
                {"text": "Business quality matters.", "evidence_ids": ["N0002"]},
                {"text": "Price matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience closes the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Opening", "summary": "The lesson opens with a framework.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "Price matters.", "evidence_ids": ["N0003"]},
                {"heading": "Conclusion", "summary": "Patience closes the lesson.", "evidence_ids": ["N0006"]},
            ],
        }

    def test_prunes_takeaway_evidence_without_losing_numeric_grounding(self):
        payload = self._base_payload()
        payload["key_takeaways"][2]["evidence_ids"] = ["N0001", "N0003", "N0004"]
        repairs = []

        result = recover_synthesis(
            payload,
            evidence=self.evidence,
            segments=self.segments,
            on_repair=repairs.append,
        )

        self.assertEqual(result["key_takeaways"][2]["evidence_ids"], ["N0004"])
        self.assertTrue(any("pruned evidence IDs" in message for message in repairs))

    def test_reorders_sections_by_transcript_progression(self):
        payload = self._base_payload()
        payload["sections"] = [
            payload["sections"][2],
            payload["sections"][0],
            payload["sections"][1],
        ]
        repairs = []

        result = recover_synthesis(
            payload,
            evidence=self.evidence,
            segments=self.segments,
            on_repair=repairs.append,
        )

        self.assertEqual(
            [section["heading"] for section in result["sections"]],
            ["Opening", "Price", "Conclusion"],
        )
        self.assertTrue(any("reordered by transcript progression" in message for message in repairs))

    def test_cannot_prune_if_numeric_grounding_would_be_lost(self):
        payload = self._base_payload()
        payload["key_takeaways"][2] = {
            "text": "The example uses a 10 percent discount and 20 percent premium.",
            "evidence_ids": ["N0001", "N0003", "N0004"],
        }

        with self.assertRaises(FinalizationError):
            recover_synthesis(
                payload,
                evidence=self.evidence,
                segments=self.segments,
            )

    def test_unknown_evidence_still_fails_closed(self):
        payload = self._base_payload()
        payload["key_takeaways"][0]["evidence_ids"] = ["N9999", "N0002", "N0003"]

        with self.assertRaises(FinalizationError):
            recover_synthesis(
                payload,
                evidence=self.evidence,
                segments=self.segments,
            )


if __name__ == "__main__":
    unittest.main()
