from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.finalization import (
    FINAL_PROMPT_VERSION,
    FinalizationError,
    NarrativeEvidence,
    build_citations_and_research,
    build_narrative_chunks,
    merge_narrative_evidence,
    parse_narrative_extraction,
    validate_synthesis,
    verify_final_package,
    write_final_package,
)


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": float(i), "end": float(i + 1), "text": text}
            for i, text in enumerate(
                [
                    "The lesson begins by comparing two approaches.",
                    "The first approach focuses on understanding the business.",
                    "The second emphasizes the price paid for that business.",
                    "A 10 percent discount is used in this example.",
                    "The speaker then explains why discipline matters.",
                    "The conclusion returns to patience and consistency.",
                ]
            )
        ]

    def test_narrative_chunking_overlaps(self):
        segments = [{"text": str(i)} for i in range(20)]
        self.assertEqual(
            build_narrative_chunks(
                segments,
                chunk_segments=10,
                overlap_segments=2,
            ),
            ((0, 9), (8, 17), (16, 19)),
        )

    def test_extraction_rejects_unsupported_number(self):
        payload = {
            "evidence": [
                {
                    "topic": "Discount",
                    "text": "The example uses a 20 percent discount.",
                    "explanation": "The figure illustrates the speaker's point.",
                    "start_segment": 3,
                    "end_segment": 3,
                }
            ]
        }
        with self.assertRaises(FinalizationError):
            parse_narrative_extraction(
                payload,
                segments=self.segments,
                minimum_segment=0,
                maximum_segment=5,
            )

    def test_extraction_rejects_machine_identifier(self):
        payload = {
            "evidence": [
                {
                    "topic": "Business",
                    "text": "The speaker discusses current_yield in the lesson.",
                    "explanation": "This is an internal identifier.",
                    "start_segment": 1,
                    "end_segment": 1,
                }
            ]
        }
        with self.assertRaises(FinalizationError):
            parse_narrative_extraction(
                payload,
                segments=self.segments,
                minimum_segment=0,
                maximum_segment=5,
            )

    def test_merge_orders_and_deduplicates(self):
        items = [
            {
                "topic": "Later",
                "text": "Patience matters.",
                "explanation": "The conclusion emphasizes patience.",
                "start_segment": 5,
                "end_segment": 5,
            },
            {
                "topic": "Earlier",
                "text": "Understanding the business matters.",
                "explanation": "The speaker introduces this first.",
                "start_segment": 1,
                "end_segment": 1,
            },
            {
                "topic": "Earlier",
                "text": "Understanding the business matters.",
                "explanation": "Duplicate wording is ignored.",
                "start_segment": 1,
                "end_segment": 1,
            },
        ]
        merged = merge_narrative_evidence(items)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].evidence_id, "N0001")
        self.assertEqual(merged[0].start_segment, 1)
        self.assertEqual(merged[1].start_segment, 5)

    def _evidence(self):
        return (
            NarrativeEvidence("N0001", "Opening", "The lesson compares approaches.", "It establishes the framework.", 0, 0),
            NarrativeEvidence("N0002", "Business", "Understanding the business matters.", "The first approach emphasizes the business.", 1, 1),
            NarrativeEvidence("N0003", "Price", "The price paid also matters.", "The second approach emphasizes price.", 2, 2),
            NarrativeEvidence("N0004", "Discount", "The example uses a 10 percent discount.", "The number illustrates the example.", 3, 3),
            NarrativeEvidence("N0005", "Discipline", "Discipline matters.", "The speaker connects discipline to the process.", 4, 4),
            NarrativeEvidence("N0006", "Conclusion", "Patience and consistency close the lesson.", "The conclusion returns to those habits.", 5, 5),
        )

    def test_synthesis_rejects_out_of_order_sections(self):
        payload = {
            "executive_summary": (
                "The lesson compares two approaches. Understanding the business matters. "
                "The price paid also matters. Patience and consistency close the lesson."
            ),
            "executive_summary_evidence_ids": ["N0001", "N0002", "N0003", "N0006"],
            "key_takeaways": [
                {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
                {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
            ],
        }
        with self.assertRaises(FinalizationError):
            validate_synthesis(
                payload,
                evidence=self._evidence(),
                segments=self.segments,
            )

    def test_synthesis_accepts_ordered_grounded_payload(self):
        payload = {
            "executive_summary": (
                "The lesson compares two approaches. Understanding the business matters. "
                "The price paid also matters. Patience and consistency close the lesson."
            ),
            "executive_summary_evidence_ids": ["N0001", "N0002", "N0003", "N0006"],
            "key_takeaways": [
                {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
        }
        result = validate_synthesis(
            payload,
            evidence=self._evidence(),
            segments=self.segments,
        )
        self.assertEqual(len(result["sections"]), 3)

    def test_formula_source_claims_become_citations(self):
        narrative = {
            "executive_summary": "A. B. C. D.",
            "executive_summary_evidence_ids": ["N0001"],
            "key_takeaways": [
                {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
        }
        formulas = [
            {
                "calculation_id": "CALC_0001",
                "formula_id": "ratio",
                "name": "Ratio",
                "ascii": "ratio = numerator / denominator",
                "latex": "r=n/d",
                "derivation_type": "stated",
                "variables": [],
                "derivation_steps": ["The speaker states the relationship."],
                "source_claims": [
                    {"start_segment": 1, "end_segment": 2, "relationship": "division"}
                ],
            }
        ]
        source_map, research, formulas_payload = build_citations_and_research(
            narrative=narrative,
            evidence=self._evidence(),
            formulas=formulas,
            segments=self.segments,
        )
        self.assertTrue(formulas_payload["formulas"][0]["citation_ids"])
        self.assertTrue(research["formulas"][0]["citation_ids"])
        self.assertGreaterEqual(len(source_map["citations"]), 4)

    def test_write_and_verify_final_package(self):
        narrative = {
            "executive_summary": "A. B. C. D.",
            "executive_summary_evidence_ids": ["N0001"],
            "key_takeaways": [
                {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
        }
        source_map, research, formulas_payload = build_citations_and_research(
            narrative=narrative,
            evidence=self._evidence(),
            formulas=[],
            segments=self.segments,
        )
        diagnostic = {
            "calculation_inventory.json": {"schema_version": "1.0", "calculations": []},
            "formula_entailment.json": {"schema_version": "1.0", "reports": []},
            "formula_coverage.json": {"schema_version": "1.0", "passed": True, "unresolved": 0},
            "rejected_formulas.json": {"schema_version": "1.0", "rejected_formulas": []},
            "visual_evidence.json": {"schema_version": "1.0", "records": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = write_final_package(
                output_root=directory,
                video_id="video123",
                title="Synthetic",
                source_url="https://example.invalid/video123",
                source_package_sha256="a" * 64,
                research=research,
                source_map=source_map,
                formulas_payload=formulas_payload,
                diagnostic_payloads=diagnostic,
                analysis_details={"model": "fake"},
            )
            self.assertTrue((result.package_dir / "_READY").is_file())
            self.assertEqual(
                verify_final_package(
                    result.package_dir,
                    source_package_sha256="a" * 64,
                    prompt_version=FINAL_PROMPT_VERSION,
                ),
                [],
            )

    def test_tampered_package_hash_is_detected(self):
        narrative = {
            "executive_summary": "A. B. C. D.",
            "executive_summary_evidence_ids": ["N0001"],
            "key_takeaways": [
                {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"text": "The example uses a 10 percent discount.", "evidence_ids": ["N0004"]},
                {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
            "sections": [
                {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
                {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0006"]},
            ],
        }
        source_map, research, formulas_payload = build_citations_and_research(
            narrative=narrative,
            evidence=self._evidence(),
            formulas=[],
            segments=self.segments,
        )
        diagnostic = {
            "calculation_inventory.json": {"schema_version": "1.0", "calculations": []},
            "formula_entailment.json": {"schema_version": "1.0", "reports": []},
            "formula_coverage.json": {"schema_version": "1.0", "passed": True, "unresolved": 0},
            "rejected_formulas.json": {"schema_version": "1.0", "rejected_formulas": []},
            "visual_evidence.json": {"schema_version": "1.0", "records": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = write_final_package(
                output_root=directory,
                video_id="video123",
                title="Synthetic",
                source_url="https://example.invalid/video123",
                source_package_sha256="a" * 64,
                research=research,
                source_map=source_map,
                formulas_payload=formulas_payload,
                diagnostic_payloads=diagnostic,
                analysis_details={"model": "fake"},
            )
            (result.package_dir / "research.md").write_text("tampered\n", encoding="utf-8")
            issues = verify_final_package(
                result.package_dir,
                source_package_sha256="a" * 64,
            )
            self.assertTrue(any("hash mismatch" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main()
