import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from youtube_research_analysis import (
    ResearchInputPackageWriter,
    ResearchManifestStore,
    ResearchPackageWriter,
    TranscriptSourcePackage,
    discover_transcript_sources,
    plan_research_sources,
)
from youtube_research_io import (
    TranscriptPackageWriter,
    TranscriptQuality,
    VideoMetadata,
)

VIDEO_ID = "abc123DEF45"


class ResearchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw_root = self.root / "Raw Transcripts"
        self.processed_root = self.root / "Processed Research"
        self.manifest_path = self.root / "manifests" / "research.jsonl"
        self._write_source("Original transcript text")

    def tearDown(self):
        self.temp.cleanup()

    def _write_source(self, first_text):
        metadata = VideoMetadata(
            video_id=VIDEO_ID,
            title="Test Research Video",
            source_url=f"https://youtu.be/{VIDEO_ID}",
            channel="Test Channel",
            duration_seconds=30,
            transcription_backend="faster-whisper-large-v3",
        )
        quality = TranscriptQuality(
            quality_status="usable",
            selected_format="txt",
            metrics={"segment_count": 3, "word_count": 12},
        )
        segments = [
            {"start": 0, "end": 10, "text": first_text, "avg_logprob": -0.1},
            {"start": 10, "end": 20, "text": "Revenue minus expenses equals profit.", "avg_logprob": -0.2},
            {"start": 20, "end": 30, "text": "Higher productivity supports long-run growth.", "avg_logprob": -0.15},
        ]
        TranscriptPackageWriter(self.raw_root).write(
            metadata=metadata,
            transcript_text=" ".join(item["text"] for item in segments),
            segments=segments,
            quality=quality,
        )

    @staticmethod
    def _research_payload():
        return {
            "executive_summary": "The lesson connects basic profitability and productivity.",
            "executive_summary_citation_ids": ["C1", "C2"],
            "key_takeaways": [
                {"text": "Profit is presented as revenue less expenses.", "citation_ids": ["C1"]},
                {"text": "Productivity matters for long-run growth.", "citation_ids": ["C2"]},
            ],
            "sections": [
                {"heading": "Profitability", "summary": "The transcript introduces a basic profit identity.", "citation_ids": ["C1"]},
                {"heading": "Productivity", "summary": "The final segment links productivity to growth.", "citation_ids": ["C2"]},
            ],
            "formulas": [
                {
                    "formula_id": "profit_identity",
                    "name": "Profit identity",
                    "ascii": "profit = revenue - expenses",
                    "latex": r"\text{Profit} = \text{Revenue} - \text{Expenses}",
                    "derivation_type": "stated",
                    "variables": [
                        {"symbol": "profit", "meaning": "Residual earnings", "unit": "currency"},
                        {"symbol": "revenue", "meaning": "Total sales", "unit": "currency"},
                        {"symbol": "expenses", "meaning": "Total costs", "unit": "currency"},
                    ],
                    "derivation_steps": ["Start with revenue.", "Subtract expenses."],
                    "citation_ids": ["C1"],
                }
            ],
            "caveats": ["The transcript gives a simplified accounting identity."],
        }

    @staticmethod
    def _citations():
        return [
            {"citation_id": "C1", "start_segment": 1, "end_segment": 1},
            {"citation_id": "C2", "start_segment": 2, "end_segment": 2},
        ]

    def test_source_loader_validates_ready_package(self):
        source = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        self.assertEqual(source.video_id, VIDEO_ID)
        self.assertEqual(len(source.segments), 3)
        self.assertEqual(len(source.package_sha256), 64)
        self.assertEqual(source.metadata["title"], "Test Research Video")

    def test_manifest_requeues_when_source_hash_changes(self):
        source = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        manifest = ResearchManifestStore(self.manifest_path)
        self.assertEqual(manifest.queue(source)["status"], "analysis_queued")
        analyzing = manifest.transition(video_id=VIDEO_ID, new_status="analyzing")
        self.assertEqual(analyzing["attempts"]["analysis"], 1)
        manifest.transition(video_id=VIDEO_ID, new_status="research_ready", updates={"research_package_sha256": "a" * 64})
        self.assertEqual(manifest.queue(source)["status"], "research_ready")

        self._write_source("Changed transcript text")
        changed = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        self.assertNotEqual(changed.package_sha256, source.package_sha256)
        requeued = manifest.queue(changed)
        self.assertEqual(requeued["status"], "analysis_queued")
        self.assertEqual(requeued["source_package_sha256"], changed.package_sha256)

    def test_writer_emits_atomic_research_package(self):
        source = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        result = ResearchPackageWriter(self.processed_root).write(
            source=source,
            research=self._research_payload(),
            citations=self._citations(),
            analysis_backend="offline-test",
            prompt_version="phase3-v1",
        )
        required = {"_READY", "metadata.json", "research.md", "research.json", "formulas.json", "source_map.json"}
        self.assertEqual({path.name for path in result.directory.iterdir()}, required)
        ready = json.loads((result.directory / "_READY").read_text(encoding="utf-8"))
        research = json.loads((result.directory / "research.json").read_text(encoding="utf-8"))
        source_map = json.loads((result.directory / "source_map.json").read_text(encoding="utf-8"))
        markdown = (result.directory / "research.md").read_text(encoding="utf-8")
        self.assertEqual(ready["status"], "research_ready")
        self.assertEqual(ready["source_package_sha256"], source.package_sha256)
        self.assertEqual(research["formulas"][0]["formula_id"], "profit_identity")
        self.assertEqual(source_map["citations"][0]["start_seconds"], 10.0)
        self.assertIn("## Executive Summary", markdown)
        self.assertIn("[C1]", markdown)

    def test_writer_rejects_unknown_citation(self):
        source = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        research = self._research_payload()
        research["key_takeaways"][0]["citation_ids"] = ["UNKNOWN"]
        with self.assertRaisesRegex(ValueError, "unknown citations"):
            ResearchPackageWriter(self.processed_root).write(
                source=source,
                research=research,
                citations=self._citations(),
                analysis_backend="offline-test",
                prompt_version="phase3-v1",
            )




class ResearchPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw_root = self.root / "Raw Transcripts"
        self.processed_root = self.root / "Processed Research"
        self.inputs_root = self.root / "Research Inputs"
        self.manifest = ResearchManifestStore(
            self.root / "manifests" / "research.jsonl"
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_source(self, video_id, title, text):
        TranscriptPackageWriter(self.raw_root).write(
            metadata=VideoMetadata(
                video_id=video_id,
                title=title,
                source_url=f"https://youtu.be/{video_id}",
                channel="Test Channel",
                duration_seconds=10,
                transcription_backend="test",
            ),
            transcript_text=text,
            segments=[
                {
                    "start": 0,
                    "end": 10,
                    "text": text,
                    "avg_logprob": -0.1,
                }
            ],
            quality=TranscriptQuality(
                quality_status="usable",
                metrics={
                    "segment_count": 1,
                    "word_count": len(text.split()),
                },
            ),
        )

    def test_planner_filters_completed_active_and_failed(self):
        ids = {
            "new": "new111AAA22",
            "ready": "rdy111AAA22",
            "active": "act111AAA22",
            "failed": "bad111AAA22",
        }

        for name, video_id in ids.items():
            self._write_source(video_id, name, f"{name} text")

        sources = discover_transcript_sources(self.raw_root)
        by_id = {source.video_id: source for source in sources}

        ready_source = by_id[ids["ready"]]
        self.manifest.queue(ready_source)
        self.manifest.transition(
            video_id=ready_source.video_id,
            new_status="analyzing",
        )
        self.manifest.transition(
            video_id=ready_source.video_id,
            new_status="research_ready",
            updates={
                "source_package_sha256": (
                    ready_source.package_sha256
                )
            },
        )

        ready_dir = self.processed_root / ready_source.video_id
        ready_dir.mkdir(parents=True)
        (ready_dir / "_READY").write_text(
            json.dumps(
                {
                    "status": "research_ready",
                    "source_package_sha256": (
                        ready_source.package_sha256
                    ),
                }
            ),
            encoding="utf-8",
        )

        active_source = by_id[ids["active"]]
        self.manifest.queue(active_source)
        self.manifest.transition(
            video_id=active_source.video_id,
            new_status="analyzing",
        )

        failed_source = by_id[ids["failed"]]
        self.manifest.queue(failed_source)
        self.manifest.transition(
            video_id=failed_source.video_id,
            new_status="analysis_failed",
            error="test failure",
        )

        plan = plan_research_sources(
            sources,
            self.manifest,
            self.processed_root,
        )

        self.assertEqual(
            plan["selected_video_ids"],
            [ids["failed"], ids["new"]],
        )
        self.assertEqual(
            plan["completed_video_ids"],
            [ids["ready"]],
        )
        self.assertEqual(
            plan["in_progress_video_ids"],
            [ids["active"]],
        )
        self.assertEqual(
            plan["retry_video_ids"],
            [ids["failed"]],
        )

    def test_planner_reselects_changed_source(self):
        video_id = "chg111AAA22"
        self._write_source(video_id, "Changed", "First text")
        source = TranscriptSourcePackage.load(
            self.raw_root,
            video_id,
        )

        self.manifest.queue(source)
        self.manifest.transition(
            video_id=video_id,
            new_status="analyzing",
        )
        self.manifest.transition(
            video_id=video_id,
            new_status="research_ready",
        )

        ready_dir = self.processed_root / video_id
        ready_dir.mkdir(parents=True)
        (ready_dir / "_READY").write_text(
            json.dumps(
                {
                    "status": "research_ready",
                    "source_package_sha256": (
                        source.package_sha256
                    ),
                }
            ),
            encoding="utf-8",
        )

        self._write_source(video_id, "Changed", "Second text")
        changed = TranscriptSourcePackage.load(
            self.raw_root,
            video_id,
        )

        plan = plan_research_sources(
            [changed],
            self.manifest,
            self.processed_root,
        )

        self.assertEqual(
            plan["selected_video_ids"],
            [video_id],
        )
        self.assertEqual(
            plan["changed_source_video_ids"],
            [video_id],
        )

    def test_input_writer_emits_numbered_segments(self):
        video_id = "inp111AAA22"
        self._write_source(
            video_id,
            "Input Package",
            "Revenue minus expenses equals profit.",
        )

        source = TranscriptSourcePackage.load(
            self.raw_root,
            video_id,
        )

        result = ResearchInputPackageWriter(
            self.inputs_root
        ).write(source)

        required = {
            "_READY",
            "analysis_input.json",
            "analysis_prompt.md",
            "metadata.json",
        }

        self.assertEqual(
            {path.name for path in result.directory.iterdir()},
            required,
        )

        payload = json.loads(
            (
                result.directory / "analysis_input.json"
            ).read_text(encoding="utf-8")
        )
        prompt = (
            result.directory / "analysis_prompt.md"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            payload["segments"][0]["segment_id"],
            "S0000",
        )
        self.assertIn("[S0000 |", prompt)
        self.assertEqual(
            payload["source_package_sha256"],
            source.package_sha256,
        )


if __name__ == "__main__":
    unittest.main()
