import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from youtube_research_io import plan_playlist_videos


@dataclass
class FakeVideo:
    video_id: str
    title: str = "Test video"


class FakeManifest:
    def __init__(self, statuses=None):
        self.statuses = statuses or {}

    def get(self, video_id):
        status = self.statuses.get(video_id)
        return {"video_id": video_id, "status": status} if status else None


class PlaylistPlannerTests(unittest.TestCase):
    def test_stable_order_deduplication_and_manifest_filtering(self):
        manifest = FakeManifest(
            {
                "ready": "analysis_ready",
                "active-queued": "queued",
                "active-transcribing": "transcribing",
                "failed": "transcription_failed",
            }
        )

        videos = [
            FakeVideo("ready"),
            FakeVideo("new-1"),
            FakeVideo("new-1", title="Duplicate occurrence"),
            FakeVideo("active-queued"),
            FakeVideo("failed"),
            FakeVideo("active-transcribing"),
            FakeVideo("new-2"),
        ]

        plan = plan_playlist_videos(videos, manifest)

        self.assertEqual(plan["input_count"], 7)
        self.assertEqual(plan["unique_count"], 6)
        self.assertEqual(
            plan["selected_video_ids"],
            ["new-1", "failed", "new-2"],
        )
        self.assertEqual(plan["duplicate_video_ids"], ["new-1"])
        self.assertEqual(plan["completed_video_ids"], ["ready"])
        self.assertEqual(
            plan["in_progress_video_ids"],
            ["active-queued", "active-transcribing"],
        )
        self.assertEqual(plan["retry_video_ids"], ["failed"])
        self.assertEqual(
            [video.video_id for video in plan["videos"]],
            ["new-1", "failed", "new-2"],
        )

    def test_generator_input_preserves_first_occurrence_order(self):
        videos = (
            FakeVideo(video_id)
            for video_id in ["video-2", "video-1", "video-2", "video-3"]
        )

        plan = plan_playlist_videos(videos, FakeManifest())

        self.assertEqual(
            plan["selected_video_ids"],
            ["video-2", "video-1", "video-3"],
        )
        self.assertEqual(plan["duplicate_video_ids"], ["video-2"])

    def test_missing_video_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no usable video_id"):
            plan_playlist_videos(
                [FakeVideo("valid"), FakeVideo("")],
                FakeManifest(),
            )



class PlaylistExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_videos_create_worker_calls(self):
        from youtube_research_io import execute_playlist_plan

        manifest = FakeManifest(
            {
                "ready": "analysis_ready",
                "queued": "queued",
                "transcribing": "transcribing",
                "failed": "transcription_failed",
            }
        )

        videos = [
            FakeVideo("ready"),
            FakeVideo("new-1"),
            FakeVideo("new-1"),
            FakeVideo("queued"),
            FakeVideo("failed"),
            FakeVideo("transcribing"),
            FakeVideo("new-2"),
        ]

        plan = plan_playlist_videos(videos, manifest)
        worker_calls = []

        async def worker(video):
            worker_calls.append(video.video_id)

        await execute_playlist_plan(plan, worker)

        self.assertEqual(
            worker_calls,
            ["new-1", "failed", "new-2"],
        )
        self.assertNotIn("ready", worker_calls)
        self.assertNotIn("queued", worker_calls)
        self.assertNotIn("transcribing", worker_calls)



class SemaphoreExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_with_semaphore_limits_peak_concurrency(self):
        import asyncio

        from youtube_research_io import execute_with_semaphore

        semaphore = asyncio.Semaphore(1)
        active = 0
        peak_active = 0

        async def operation(value):
            nonlocal active, peak_active

            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return value

        results = await asyncio.gather(
            *[
                execute_with_semaphore(
                    semaphore,
                    operation,
                    value,
                )
                for value in range(4)
            ]
        )

        self.assertEqual(results, [0, 1, 2, 3])
        self.assertEqual(peak_active, 1)


if __name__ == "__main__":
    unittest.main()
