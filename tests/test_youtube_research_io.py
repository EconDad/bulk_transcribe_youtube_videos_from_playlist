import json
import tempfile
import unittest
from pathlib import Path

from youtube_research_io import ManifestStore, TranscriptPackageWriter, TranscriptQuality, VideoMetadata

VIDEO_ID = "As1a2VgbdWg"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class YoutubeResearchIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "Raw Transcripts"
        self.manifest_path = self.root / "manifests" / "videos.jsonl"
        self.metadata = VideoMetadata(
            video_id=VIDEO_ID,
            title="Warren Buffett: How To Analyze a BALANCE SHEET",
            source_url=URL,
            channel="Brian Feroldi",
            duration_seconds=480,
            language="en",
            transcription_backend="faster-whisper-large-v3",
        )
        self.segments = [
            {"start": 0, "end": 2, "text": "Warren Buffett has been analyzing businesses.", "avg_logprob": -0.1},
            {"start": 2, "end": 5, "text": "Assets equal liabilities plus shareholders equity.", "avg_logprob": -0.2},
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_package_and_state_sequence(self):
        manifest = ManifestStore(self.manifest_path)
        manifest.queue(self.metadata)
        manifest.transition(video_id=VIDEO_ID, new_status="transcribing")
        result = TranscriptPackageWriter(self.raw).write(
            metadata=self.metadata,
            transcript_text="One. Two.",
            segments=self.segments,
            quality=TranscriptQuality(quality_status="usable", metrics={"segment_count": 2}),
        )
        manifest.transition(
            video_id=VIDEO_ID,
            new_status="analysis_ready",
            updates={"transcript_directory": str(result.directory), "package_sha256": result.package_sha256},
        )
        self.assertEqual(
            {"transcript.txt", "transcript.csv", "transcript.json", "metadata.json", "quality.json", "_READY"},
            {path.name for path in result.directory.iterdir()},
        )
        record = manifest.get(VIDEO_ID)
        self.assertEqual("analysis_ready", record["status"])
        self.assertEqual(["queued", "transcribing", "analysis_ready"], [e["status"] for e in record["status_history"]])
        rows = json.loads((result.directory / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(-0.1, rows[0]["avg_logprob"])

    def test_failure_transition(self):
        manifest = ManifestStore(self.manifest_path)
        manifest.queue(self.metadata)
        manifest.transition(video_id=VIDEO_ID, new_status="transcribing")
        manifest.transition(video_id=VIDEO_ID, new_status="transcription_failed", error="RuntimeError: synthetic")
        self.assertEqual("transcription_failed", manifest.get(VIDEO_ID)["status"])
        self.assertFalse((self.raw / VIDEO_ID / "_READY").exists())

    def test_rerun_is_allowed_and_counted(self):
        manifest = ManifestStore(self.manifest_path)
        manifest.queue(self.metadata)
        manifest.transition(video_id=VIDEO_ID, new_status="transcribing")
        manifest.transition(video_id=VIDEO_ID, new_status="transcribing")
        self.assertEqual(2, manifest.get(VIDEO_ID)["attempts"]["transcription"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
