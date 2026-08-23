from __future__ import annotations

import unittest

from research_v43.finalization import NarrativeEvidence
from research_v43.finalization_recovery import (
    build_citations_and_research_with_formula_claim_splitting,
)


class FinalizationRecoveryTests(unittest.TestCase):
    def _segments(self):
        return [
            {
                "start": float(index),
                "end": float(index + 1),
                "text": f"Segment {index} source text.",
            }
            for index in range(20)
        ]

    def _narrative(self):
        return {
            "executive_summary": "A. B. C. D.",
            "executive_summary_evidence_ids": ["N0001"],
            "key_takeaways": [
                {"text": "A grounded point.", "evidence_ids": ["N0001"]},
                {"text": "Another grounded point.", "evidence_ids": ["N0001"]},
                {"text": "A third grounded point.", "evidence_ids": ["N0001"]},
                {"text": "A fourth grounded point.", "evidence_ids": ["N0001"]},
            ],
            "sections": [
                {"heading": "One", "summary": "A grounded point.", "evidence_ids": ["N0001"]},
                {"heading": "Two", "summary": "Another grounded point.", "evidence_ids": ["N0001"]},
                {"heading": "Three", "summary": "A third grounded point.", "evidence_ids": ["N0001"]},
            ],
        }

    def test_long_formula_source_claim_is_split_only_for_citations(self):
        evidence = (
            NarrativeEvidence(
                "N0001",
                "Topic",
                "A grounded point.",
                "The source supports it.",
                0,
                0,
            ),
        )
        original_claim = {
            "start_segment": 4,
            "end_segment": 11,
            "relationship": "subtraction",
        }
        formulas = [
            {
                "calculation_id": "CALC_0001",
                "formula_id": "result",
                "name": "Result",
                "ascii": "result = left - right",
                "latex": "result = left - right",
                "derivation_type": "derived",
                "variables": [],
                "derivation_steps": ["Subtract right from left."],
                "source_claims": [original_claim],
            }
        ]

        source_map, research, formulas_payload = (
            build_citations_and_research_with_formula_claim_splitting(
                narrative=self._narrative(),
                evidence=evidence,
                formulas=formulas,
                segments=self._segments(),
            )
        )

        final_formula = formulas_payload["formulas"][0]
        self.assertEqual(final_formula["source_claims"], [original_claim])
        self.assertEqual(research["formulas"][0]["source_claims"], [original_claim])

        citation_ids = final_formula["citation_ids"]
        self.assertEqual(len(citation_ids), 2)

        citations = {
            item["citation_id"]: item
            for item in source_map["citations"]
        }
        formula_ranges = [
            (
                citations[citation_id]["start_segment"],
                citations[citation_id]["end_segment"],
            )
            for citation_id in citation_ids
        ]
        self.assertEqual(formula_ranges, [(4, 9), (10, 11)])
        self.assertTrue(
            all(citations[citation_id]["segment_count"] <= 6 for citation_id in citation_ids)
        )

    def test_short_formula_source_claim_is_unchanged(self):
        evidence = (
            NarrativeEvidence(
                "N0001",
                "Topic",
                "A grounded point.",
                "The source supports it.",
                0,
                0,
            ),
        )
        original_claim = {
            "start_segment": 4,
            "end_segment": 6,
            "relationship": "division",
        }
        formulas = [
            {
                "calculation_id": "CALC_0001",
                "formula_id": "ratio",
                "name": "Ratio",
                "ascii": "ratio = left / right",
                "latex": "ratio = left / right",
                "derivation_type": "stated",
                "variables": [],
                "derivation_steps": ["Divide left by right."],
                "source_claims": [original_claim],
            }
        ]

        source_map, _, formulas_payload = (
            build_citations_and_research_with_formula_claim_splitting(
                narrative=self._narrative(),
                evidence=evidence,
                formulas=formulas,
                segments=self._segments(),
            )
        )

        final_formula = formulas_payload["formulas"][0]
        self.assertEqual(final_formula["source_claims"], [original_claim])
        self.assertEqual(len(final_formula["citation_ids"]), 1)
        citation = next(
            item
            for item in source_map["citations"]
            if item["citation_id"] == final_formula["citation_ids"][0]
        )
        self.assertEqual((citation["start_segment"], citation["end_segment"]), (4, 6))


if __name__ == "__main__":
    unittest.main()
