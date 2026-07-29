from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.0"

ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"queued"},
    "queued": {"transcribing", "transcription_failed"},
    "transcribing": {"transcribing", "analysis_ready", "transcription_failed"},
    "analysis_ready": {"transcribing"},
    "transcription_failed": {"queued", "transcribing"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_video_id(video_id: str) -> None:
    if len(video_id) != 11 or not all(char.isalnum() or char in "_-" for char in video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id!r}")


def canonical_youtube_url(video_id: str) -> str:
    validate_video_id(video_id)
    return f"https://www.youtube.com/watch?v={video_id}"


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    source_url: str
    channel: str | None = None
    duration_seconds: float | None = None
    language: str | None = None
    transcription_backend: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> dict[str, Any]:
        validate_video_id(self.video_id)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "video_id": self.video_id,
            "canonical_url": canonical_youtube_url(self.video_id),
            "source_url": self.source_url,
            "title": self.title,
            "channel": self.channel,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "transcription_backend": self.transcription_backend,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True)
class TranscriptQuality:
    quality_status: str = "unreviewed"
    selected_format: str = "txt"
    known_issues: Sequence[str] = ()
    requires_retranscription: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self, video_id: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            "quality_status": self.quality_status,
            "selected_format": self.selected_format,
            "known_issues": list(self.known_issues),
            "requires_retranscription": self.requires_retranscription,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PackageResult:
    video_id: str
    directory: Path
    checksums: Mapping[str, str]
    package_sha256: str
    completed_at: str


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class ManifestStore:
    """Atomic JSONL manifest keyed by YouTube video ID."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {self.path}:{line_number}: {exc}") from exc
        return records

    def _write_unlocked(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def get(self, video_id: str) -> dict[str, Any] | None:
        validate_video_id(video_id)
        with _exclusive_lock(self.lock_path):
            return next((r for r in self._read_unlocked() if r.get("video_id") == video_id), None)

    def transition(
        self,
        *,
        video_id: str,
        new_status: str,
        title: str | None = None,
        url: str | None = None,
        updates: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        validate_video_id(video_id)
        now = utc_now()
        with _exclusive_lock(self.lock_path):
            records = self._read_unlocked()
            index = next((i for i, record in enumerate(records) if record.get("video_id") == video_id), None)
            current = records[index] if index is not None else None
            current_status = current.get("status") if current else None
            if new_status not in ALLOWED_TRANSITIONS.get(current_status, set()):
                raise ValueError(f"Invalid manifest transition for {video_id}: {current_status!r} -> {new_status!r}")

            record = dict(current or {})
            record.setdefault("schema_version", SCHEMA_VERSION)
            record["video_id"] = video_id
            record["url"] = url or record.get("url") or canonical_youtube_url(video_id)
            if title is not None:
                record["title"] = title
            record["status"] = new_status
            record.setdefault("created_at", now)
            record["updated_at"] = now
            history = list(record.get("status_history", []))
            event: dict[str, Any] = {"status": new_status, "at": now}
            if error:
                event["error"] = error
            history.append(event)
            record["status_history"] = history
            attempts = dict(record.get("attempts", {}))
            if new_status == "transcribing":
                attempts["transcription"] = int(attempts.get("transcription", 0)) + 1
            record["attempts"] = attempts
            record["last_error"] = error if error else None
            record.update(dict(updates or {}))

            if index is None:
                records.append(record)
            else:
                records[index] = record
            self._write_unlocked(records)
            return record

    def queue(self, metadata: VideoMetadata) -> dict[str, Any]:
        existing = self.get(metadata.video_id)
        if existing is None or existing.get("status") == "transcription_failed":
            return self.transition(
                video_id=metadata.video_id,
                new_status="queued",
                title=metadata.title,
                url=metadata.source_url,
            )
        return existing


class TranscriptPackageWriter:
    """Atomically writes Raw Transcripts/<video_id>/ and emits _READY last."""

    def __init__(self, raw_transcripts_root: str | Path):
        self.raw_root = Path(raw_transcripts_root)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _package_digest(checksums: Mapping[str, str]) -> str:
        digest = hashlib.sha256()
        for name, checksum in sorted(checksums.items()):
            digest.update(f"{name}\0{checksum}\n".encode("utf-8"))
        return digest.hexdigest()

    def write(
        self,
        *,
        metadata: VideoMetadata,
        transcript_text: str,
        segments: Iterable[Mapping[str, Any]],
        quality: TranscriptQuality,
    ) -> PackageResult:
        validate_video_id(metadata.video_id)
        normalized_segments = [
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "text": str(segment.get("text", "")),
                "avg_logprob": float(segment.get("avg_logprob", 0.0)),
            }
            for segment in segments
        ]
        if not normalized_segments:
            raise ValueError("Transcript contains no segments")

        self.raw_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.raw_root / metadata.video_id
        staging_dir = self.raw_root / f".{metadata.video_id}.{uuid.uuid4().hex}.staging"
        backup_dir = self.raw_root / f".{metadata.video_id}.{uuid.uuid4().hex}.backup"
        staging_dir.mkdir()
        completed_at = utc_now()

        try:
            (staging_dir / "transcript.txt").write_text(
                transcript_text.strip() + "\n", encoding="utf-8", newline="\n"
            )
            with (staging_dir / "transcript.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["start", "end", "text", "avg_logprob"])
                writer.writeheader()
                writer.writerows(normalized_segments)
            (staging_dir / "transcript.json").write_text(
                json.dumps(normalized_segments, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (staging_dir / "quality.json").write_text(
                json.dumps(quality.to_mapping(metadata.video_id), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            artifact_names = ["transcript.txt", "transcript.csv", "transcript.json", "quality.json"]
            checksums = {name: self._sha256(staging_dir / name) for name in artifact_names}
            metadata_payload = metadata.normalized()
            metadata_payload.update(
                {
                    "available_formats": ["txt", "csv", "json"],
                    "selected_format": quality.selected_format,
                    "segment_count": len(normalized_segments),
                    "completed_at": completed_at,
                    "artifact_sha256": checksums,
                }
            )
            (staging_dir / "metadata.json").write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            checksums["metadata.json"] = self._sha256(staging_dir / "metadata.json")
            package_sha256 = self._package_digest(checksums)
            (staging_dir / "_READY").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "video_id": metadata.video_id,
                        "status": "analysis_ready",
                        "completed_at": completed_at,
                        "package_sha256": package_sha256,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

            if final_dir.exists():
                os.replace(final_dir, backup_dir)
            os.replace(staging_dir, final_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            return PackageResult(metadata.video_id, final_dir, checksums, package_sha256, completed_at)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if backup_dir.exists() and not final_dir.exists():
                os.replace(backup_dir, final_dir)
            raise


def plan_playlist_videos(
    videos: Any,
    manifest: ManifestStore,
) -> dict[str, Any]:
    """Select playlist videos for processing in stable playlist order.

    Deduplication is strictly by YouTube video ID. Completed and currently
    active manifest entries are skipped, while failed entries are retried.
    """
    selected_videos: list[Any] = []
    selected_video_ids: list[str] = []
    duplicate_video_ids: list[str] = []
    completed_video_ids: list[str] = []
    in_progress_video_ids: list[str] = []
    retry_video_ids: list[str] = []

    seen_video_ids: set[str] = set()
    input_count = 0

    for video in videos:
        input_count += 1

        video_id = str(getattr(video, "video_id", "") or "").strip()
        if not video_id:
            raise ValueError(
                f"Playlist entry {input_count} has no usable video_id"
            )

        if video_id in seen_video_ids:
            duplicate_video_ids.append(video_id)
            continue

        seen_video_ids.add(video_id)
        existing = manifest.get(video_id)
        status = existing.get("status") if existing else None

        if status == "analysis_ready":
            completed_video_ids.append(video_id)
            continue

        if status in {"queued", "transcribing"}:
            in_progress_video_ids.append(video_id)
            continue

        if status == "transcription_failed":
            retry_video_ids.append(video_id)

        selected_videos.append(video)
        selected_video_ids.append(video_id)

    return {
        "videos": selected_videos,
        "input_count": input_count,
        "unique_count": len(seen_video_ids),
        "selected_video_ids": selected_video_ids,
        "duplicate_video_ids": duplicate_video_ids,
        "completed_video_ids": completed_video_ids,
        "in_progress_video_ids": in_progress_video_ids,
        "retry_video_ids": retry_video_ids,
    }


async def execute_playlist_plan(
    plan: dict[str, Any],
    worker: Any,
) -> None:
    """Run the worker only for videos selected by the playlist plan."""
    import asyncio

    tasks = [worker(video) for video in plan["videos"]]
    if tasks:
        await asyncio.gather(*tasks)
