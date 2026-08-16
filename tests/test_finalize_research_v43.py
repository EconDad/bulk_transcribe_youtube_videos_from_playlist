from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finalize_research_v43 import finalize_video
from research_v43.artifacts import write_diagnostic_package
from research_v43.model_client import JsonModelResponse, ModelInvocation


class FakeClient:
    model = "fake-qwen"

    def __init__(self):
        self.calls = 0

    def complete_json(self, *, stage, **kwargs):
        self.calls += 1
        if stage.startswith("narrative_evidence"):
            if "S0-S5" not in stage:
                raise AssertionError(stage)
            payload = {
                "evidence": [
                    {
                        "topic": "Opening",
                        "text": "The lesson compares two approaches.",
                        "explanation": "The comparison establishes the lesson structure.",
                        "start_segment": 0,
                        "end_segment": 0,
                    },
                    {
                        "topic": "Business",
                        "text": "Understanding the business matters.",
                        "explanation": "The first approach emphasizes understanding the business.",
                        "start_segment": 1,
                        "end_segment": 1,
                    },
                    {
                        "topic": "Price",
                        "text": "The price paid also matters.",
                        "explanation": "The second approach emphasizes the price paid.",
                        "start_segment": 2,
                        "end_segment": 2,
                    },
                    {
                        "topic": "Discipline",
                        "text": "Discipline matters to the process.",
                        "explanation": "The speaker connects discipline to the process.",
                        "start_segment": 4,
                        "end_segment": 4,
                    },
                    {
                        "topic": "Conclusion",
                        "text": "Patience and consistency close the lesson.",
                        "explanation": "The conclusion returns to patience and consistency.",
                        "start_segment": 5,
                        "end_segment": 5,
                    },
                ]
            }
        elif stage.startswith("narrative_synthesis"):
            payload = {
                "executive_summary": (
                    "The lesson compares two approaches. Understanding the business matters. "
                    "The price paid also matters. Patience and consistency close the lesson."
                ),
                "executive_summary_evidence_ids": ["N0001", "N0002", "N0003", "N0005"],
                "key_takeaways": [
                    {"text": "Understanding the business matters.", "evidence_ids": ["N0002"]},
                    {"text": "The price paid also matters.", "evidence_ids": ["N0003"]},
                    {"text": "Discipline matters to the process.", "evidence_ids": ["N0004"]},
                    {"text": "Patience and consistency close the lesson.", "evidence_ids": ["N0005"]},
                ],
                "sections": [
                    {"heading": "Opening", "summary": "The lesson compares two approaches.", "evidence_ids": ["N0001"]},
                    {"heading": "Price", "summary": "The price paid also matters.", "evidence_ids": ["N0003"]},
                    {"heading": "Conclusion", "summary": "Patience and consistency close the lesson.", "evidence_ids": ["N0005"]},
                ],
            }
        else:
            raise AssertionError(stage)
        return JsonModelResponse(
            payload=payload,
            invocation=ModelInvocation(
                model=self.model,
                think=False,
                num_ctx=8192,
                prompt_chars=100,
                response_chars=100,
                stage=stage,
            ),
        )


def write_raw_package(root: Path, video_id: str) -> str:
    package = root / video_id
    package.mkdir(parents=True)
    segments = [
        {"start": float(i), "end": float(i + 1), "text": text, "avg_logprob": -0.1}
        for i, text in enumerate(
            [
                "The lesson begins by comparing two approaches.",
                "The first approach focuses on understanding the business.",
                "The second emphasizes the price paid for that business.",
                "A simple example follows without a reusable formula.",
                "The speaker then explains why discipline matters to the process.",
                "The conclusion returns to patience and consistency.",
            ]
        )
    ]
    (package / "transcript.json").write_text(json.dumps(segments), encoding="utf-8")
    (package / "transcript.txt").write_text("\n".join(item["text"] for item in segments) + "\n", encoding="utf-8")
    metadata = {
        "video_id": video_id,
        "title": "Synthetic conceptual lesson",
        "canonical_url": f"https://youtube.com/watch?v={video_id}",
        "segment_count": len(segments),
    }
    (package / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (package / "quality.json").write_text(
        json.dumps({"video_id": video_id, "requires_retranscription": False}),
        encoding="utf-8",
    )
    # Transcript source loader treats this as the immutable upstream package identity.
    package_sha = hashlib.sha256((package / "transcript.json").read_bytes()).hexdigest()
    (package / "_READY").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "status": "analysis_ready",
                "package_sha256": package_sha,
            }
        ),
        encoding="utf-8",
    )
    return package_sha


class FinalizeResearchV43Tests(unittest.TestCase):
    def test_finalize_conceptual_video_without_invented_formulas(self):
        video_id = "abc123XYZ01"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "Raw Transcripts"
            diagnostic_root = root / "Diagnostics"
            processed_root = root / "Processed Research"
            manifest_path = root / "manifests" / "research.jsonl"
            source_sha = write_raw_package(raw_root, video_id)

            write_diagnostic_package(
                output_root=diagnostic_root,
                video_id=video_id,
                source_package_sha256=source_sha,
                prompt_version="phase4-qwen3-v4.3-stage-e.3",
                payloads={
                    "calculation_inventory.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "calculations": [],
                    },
                    "formulas.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "formulas": [],
                    },
                    "formula_entailment.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "reports": [],
                    },
                    "formula_coverage.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "passed": True,
                        "identified_calculations": 0,
                        "formulas_retained": 0,
                        "non_symbolic_calculations": 0,
                        "insufficient_source_detail": 0,
                        "visual_review_required": 0,
                        "formula_rejected": 0,
                        "unresolved": 0,
                        "issues": [],
                        "resolutions": [],
                    },
                    "rejected_formulas.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "rejected_formulas": [],
                    },
                    "model_invocations.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "invocations": [],
                    },
                    "visual_evidence.json": {
                        "schema_version": "1.0",
                        "video_id": video_id,
                        "records": [],
                    },
                },
            )

            result = finalize_video(
                video_id=video_id,
                raw_root=raw_root,
                diagnostic_root=diagnostic_root,
                processed_root=processed_root,
                manifest_path=manifest_path,
                client=FakeClient(),
                chunk_segments=8,
                overlap_segments=1,
            )

            research = json.loads((result / "research.json").read_text(encoding="utf-8"))
            formulas = json.loads((result / "formulas.json").read_text(encoding="utf-8"))
            ready = json.loads((result / "_READY").read_text(encoding="utf-8"))
            manifest = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][0]

            self.assertEqual(research["formulas"], [])
            self.assertEqual(formulas["formulas"], [])
            self.assertEqual(ready["status"], "research_ready")
            self.assertEqual(manifest["status"], "research_ready")
            self.assertEqual(manifest["prompt_version"], "phase4-qwen3-v4.3-stage-f.1")


if __name__ == "__main__":
    unittest.main()
