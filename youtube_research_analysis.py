from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from youtube_research_io import (
    PackageResult,
    _exclusive_lock,
    canonical_youtube_url,
    utc_now,
    validate_video_id,
)

RESEARCH_SCHEMA_VERSION = "1.0"
RESEARCH_ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"analysis_queued"},
    "analysis_queued": {"analysis_queued", "analyzing", "analysis_failed"},
    "analyzing": {"analyzing", "research_ready", "analysis_failed"},
    "research_ready": {"analysis_queued", "analyzing"},
    "analysis_failed": {"analysis_queued", "analyzing"},
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_digest(checksums: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, checksum in sorted(checksums.items()):
        digest.update(f"{name}\0{checksum}\n".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TranscriptSourcePackage:
    video_id: str
    directory: Path
    package_sha256: str
    metadata: Mapping[str, Any]
    quality: Mapping[str, Any]
    transcript_text: str
    segments: tuple[Mapping[str, Any], ...]

    @classmethod
    def load(cls, raw_root: str | Path, video_id: str) -> "TranscriptSourcePackage":
        validate_video_id(video_id)
        directory = Path(raw_root) / video_id
        ready = _read_json(directory / "_READY")
        metadata = _read_json(directory / "metadata.json")
        quality = _read_json(directory / "quality.json")
        raw_segments = _read_json(directory / "transcript.json")

        if ready.get("video_id") != video_id or ready.get("status") != "analysis_ready":
            raise ValueError(f"Transcript package {video_id} is not analysis_ready")
        if metadata.get("video_id") != video_id or quality.get("video_id") != video_id:
            raise ValueError(f"Source video_id mismatch for {video_id}")
        if quality.get("requires_retranscription"):
            raise ValueError(f"Transcript package {video_id} requires retranscription")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"Transcript package {video_id} has no segments")

        segments = tuple(
            {
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", item.get("start", 0.0))),
                "text": str(item.get("text", "")),
                "avg_logprob": float(item.get("avg_logprob", 0.0)),
            }
            for item in raw_segments
        )
        if any(item["end"] < item["start"] for item in segments):
            raise ValueError(f"Transcript package {video_id} has invalid timestamps")
        expected_count = metadata.get("segment_count")
        if expected_count is not None and int(expected_count) != len(segments):
            raise ValueError(f"Segment count mismatch for {video_id}")

        transcript_text = (directory / "transcript.txt").read_text(encoding="utf-8")
        if not transcript_text.strip():
            raise ValueError(f"Transcript text is empty for {video_id}")
        package_sha256 = str(ready.get("package_sha256") or "")
        if len(package_sha256) != 64:
            raise ValueError(f"Invalid source package SHA for {video_id}")

        return cls(video_id, directory, package_sha256, metadata, quality, transcript_text, segments)


class ResearchManifestStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
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
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
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
            return next((item for item in self._read_unlocked() if item.get("video_id") == video_id), None)

    def transition(self, *, video_id: str, new_status: str, title: str | None = None,
                   url: str | None = None, updates: Mapping[str, Any] | None = None,
                   error: str | None = None) -> dict[str, Any]:
        validate_video_id(video_id)
        now = utc_now()
        with _exclusive_lock(self.lock_path):
            records = self._read_unlocked()
            index = next((i for i, item in enumerate(records) if item.get("video_id") == video_id), None)
            current = records[index] if index is not None else None
            current_status = current.get("status") if current else None
            if new_status not in RESEARCH_ALLOWED_TRANSITIONS.get(current_status, set()):
                raise ValueError(f"Invalid research transition for {video_id}: {current_status!r} -> {new_status!r}")

            record = dict(current or {})
            record.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
            record["video_id"] = video_id
            record["url"] = url or record.get("url") or canonical_youtube_url(video_id)
            if title is not None:
                record["title"] = title
            record["status"] = new_status
            record.setdefault("created_at", now)
            record["updated_at"] = now
            history = list(record.get("status_history", []))
            event = {"status": new_status, "at": now}
            if error:
                event["error"] = error
            history.append(event)
            record["status_history"] = history
            attempts = dict(record.get("attempts", {}))
            if new_status == "analyzing":
                attempts["analysis"] = int(attempts.get("analysis", 0)) + 1
            record["attempts"] = attempts
            record["last_error"] = error if error else None
            record.update(dict(updates or {}))
            if index is None:
                records.append(record)
            else:
                records[index] = record
            self._write_unlocked(records)
            return record

    def queue(self, source: TranscriptSourcePackage) -> dict[str, Any]:
        existing = self.get(source.video_id)
        source_changed = existing is not None and existing.get("source_package_sha256") != source.package_sha256
        if existing is None or existing.get("status") == "analysis_failed" or source_changed:
            return self.transition(
                video_id=source.video_id,
                new_status="analysis_queued",
                title=str(source.metadata.get("title") or ""),
                url=str(source.metadata.get("source_url") or canonical_youtube_url(source.video_id)),
                updates={"source_package_sha256": source.package_sha256, "source_directory": str(source.directory)},
            )
        return existing


class ResearchPackageWriter:
    REQUIRED_KEYS = {"executive_summary", "executive_summary_citation_ids", "key_takeaways", "sections", "formulas", "caveats"}

    def __init__(self, processed_root: str | Path):
        self.processed_root = Path(processed_root)

    def _source_map(self, source: TranscriptSourcePackage, citations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        records, seen = [], set()
        for item in citations:
            citation_id = str(item.get("citation_id") or "").strip()
            if not citation_id or citation_id in seen:
                raise ValueError(f"Invalid or duplicate citation_id: {citation_id!r}")
            seen.add(citation_id)
            start_index = int(item.get("start_segment", -1))
            end_index = int(item.get("end_segment", start_index))
            if start_index < 0 or end_index < start_index or end_index >= len(source.segments):
                raise ValueError(f"Invalid segment range for {citation_id}: {start_index}-{end_index}")
            records.append({
                "citation_id": citation_id,
                "start_segment": start_index,
                "end_segment": end_index,
                "start_seconds": source.segments[start_index]["start"],
                "end_seconds": source.segments[end_index]["end"],
                "segment_count": end_index - start_index + 1,
            })
        return {"schema_version": RESEARCH_SCHEMA_VERSION, "video_id": source.video_id,
                "source_package_sha256": source.package_sha256, "citations": records}

    @staticmethod
    def _citation_ids(value: Any, valid: set[str], context: str) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{context} requires citation_ids")
        result = [str(item) for item in value]
        unknown = [item for item in result if item not in valid]
        if unknown:
            raise ValueError(f"{context} references unknown citations: {unknown}")
        return result

    def _normalize(self, research: Mapping[str, Any], valid: set[str]) -> dict[str, Any]:
        missing = self.REQUIRED_KEYS - set(research)
        if missing:
            raise ValueError(f"Research payload missing keys: {sorted(missing)}")
        summary = str(research["executive_summary"]).strip()
        if not summary:
            raise ValueError("executive_summary cannot be empty")
        result = {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "executive_summary": summary,
            "executive_summary_citation_ids": self._citation_ids(research["executive_summary_citation_ids"], valid, "executive_summary"),
            "key_takeaways": [], "sections": [], "formulas": [],
            "caveats": [str(item).strip() for item in research["caveats"] if str(item).strip()],
        }
        for i, item in enumerate(research["key_takeaways"]):
            text = str(item.get("text") or "").strip()
            if not text:
                raise ValueError(f"key_takeaways[{i}] text cannot be empty")
            result["key_takeaways"].append({"text": text, "citation_ids": self._citation_ids(item.get("citation_ids"), valid, f"key_takeaways[{i}]")})
        for i, item in enumerate(research["sections"]):
            heading, summary_text = str(item.get("heading") or "").strip(), str(item.get("summary") or "").strip()
            if not heading or not summary_text:
                raise ValueError(f"sections[{i}] requires heading and summary")
            result["sections"].append({"heading": heading, "summary": summary_text,
                                       "citation_ids": self._citation_ids(item.get("citation_ids"), valid, f"sections[{i}]")})
        formula_ids = set()
        for i, item in enumerate(research["formulas"]):
            formula_id = str(item.get("formula_id") or "").strip()
            derivation_type = str(item.get("derivation_type") or "").strip()
            if not formula_id or formula_id in formula_ids:
                raise ValueError(f"Invalid or duplicate formula_id: {formula_id!r}")
            if derivation_type not in {"stated", "derived"}:
                raise ValueError(f"formulas[{i}] derivation_type must be stated or derived")
            formula_ids.add(formula_id)
            result["formulas"].append({
                "formula_id": formula_id, "name": str(item.get("name") or "").strip(),
                "ascii": str(item.get("ascii") or "").strip(), "latex": str(item.get("latex") or "").strip(),
                "derivation_type": derivation_type, "variables": list(item.get("variables") or []),
                "derivation_steps": [str(step) for step in item.get("derivation_steps") or []],
                "citation_ids": self._citation_ids(item.get("citation_ids"), valid, f"formulas[{i}]")})
        return result

    @staticmethod
    def _labels(ids: Sequence[str]) -> str:
        return " ".join(f"[{item}]" for item in ids)

    def _markdown(self, source: TranscriptSourcePackage, research: Mapping[str, Any], source_map: Mapping[str, Any]) -> str:
        title = str(source.metadata.get("title") or source.video_id)
        lines = [f"# {title}", "", f"- **Video ID:** `{source.video_id}`",
                 f"- **Source:** {source.metadata.get('canonical_url') or canonical_youtube_url(source.video_id)}",
                 f"- **Transcript package SHA-256:** `{source.package_sha256}`", "", "## Executive Summary", "",
                 research["executive_summary"] + " " + self._labels(research["executive_summary_citation_ids"]), "", "## Key Takeaways", ""]
        lines += [f"- {item['text']} {self._labels(item['citation_ids'])}" for item in research["key_takeaways"]]
        lines += ["", "## Sections", ""]
        for item in research["sections"]:
            lines += [f"### {item['heading']}", "", item["summary"] + " " + self._labels(item["citation_ids"]), ""]
        lines += ["## Formulas", ""]
        if not research["formulas"]:
            lines += ["No formulas were supported by this transcript.", ""]
        for item in research["formulas"]:
            steps = list(item.get("derivation_steps") or [])
            meaning = ""
            if steps and steps[0].startswith("Meaning:"):
                meaning = steps.pop(0).split(":", 1)[1].strip()

            lines += [
                f"### {item['name'] or item['formula_id']}",
                "",
                f"- **Formula ID:** `{item['formula_id']}`",
                f"- **Type:** {item['derivation_type']}",
            ]
            if meaning:
                lines.append(f"- **Meaning:** {meaning}")
            lines += [
                f"- **Machine-readable:** `{item['ascii']}`",
                f"- **LaTeX:** `${item['latex']}$`",
                f"- **Sources:** {self._labels(item['citation_ids'])}",
            ]

            variables = list(item.get("variables") or [])
            if variables:
                lines += ["- **Variables:**", ""]
                for variable in variables:
                    symbol = str(variable.get("symbol") or "")
                    variable_meaning = str(variable.get("meaning") or "")
                    unit = str(variable.get("unit") or "")
                    unit_suffix = f" — unit: {unit}" if unit else ""
                    lines.append(
                        f"  - `{symbol}`: {variable_meaning}{unit_suffix}"
                    )

            if steps:
                lines += ["- **Derivation:**", ""]
                for number, step in enumerate(steps, start=1):
                    lines.append(f"  {number}. {step}")

            lines.append("")
        lines += ["## Caveats", ""] + ([f"- {item}" for item in research["caveats"]] or ["- None identified."])
        lines += ["", "## Source Map", ""]
        for item in source_map["citations"]:
            lines.append(f"- **[{item['citation_id']}]** segments {item['start_segment']}-{item['end_segment']}; {item['start_seconds']:.2f}s-{item['end_seconds']:.2f}s")
        return "\n".join(lines).rstrip() + "\n"

    def write(self, *, source: TranscriptSourcePackage, research: Mapping[str, Any],
              citations: Sequence[Mapping[str, Any]], analysis_backend: str,
              prompt_version: str,
              analysis_details: Mapping[str, Any] | None = None) -> PackageResult:
        source_map = self._source_map(source, citations)
        valid_ids = {item["citation_id"] for item in source_map["citations"]}
        normalized = self._normalize(research, valid_ids)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.processed_root / source.video_id
        staging = self.processed_root / f".{source.video_id}.{uuid.uuid4().hex}.staging"
        backup = self.processed_root / f".{source.video_id}.{uuid.uuid4().hex}.backup"
        completed_at = utc_now()
        with _exclusive_lock(self.processed_root / ".locks" / f"{source.video_id}.lock"):
            staging.mkdir()
            try:
                research_payload = {"schema_version": RESEARCH_SCHEMA_VERSION, "video_id": source.video_id,
                                    "title": source.metadata.get("title"), "source_package_sha256": source.package_sha256, **normalized}
                formulas_payload = {"schema_version": RESEARCH_SCHEMA_VERSION, "video_id": source.video_id,
                                    "source_package_sha256": source.package_sha256, "formulas": normalized["formulas"]}
                payloads = {"research.json": research_payload, "formulas.json": formulas_payload, "source_map.json": source_map}
                for name, payload in payloads.items():
                    (staging / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                (staging / "research.md").write_text(self._markdown(source, normalized, source_map), encoding="utf-8", newline="\n")
                artifact_names = ["research.md", "research.json", "formulas.json", "source_map.json"]
                checksums = {name: _sha256(staging / name) for name in artifact_names}
                metadata = {"schema_version": RESEARCH_SCHEMA_VERSION, "video_id": source.video_id,
                            "title": source.metadata.get("title"), "canonical_url": source.metadata.get("canonical_url") or canonical_youtube_url(source.video_id),
                            "source_package_sha256": source.package_sha256, "source_directory": str(source.directory),
                            "analysis_backend": analysis_backend, "prompt_version": prompt_version, "completed_at": completed_at,
                            "artifact_sha256": checksums}
                if analysis_details is not None:
                    metadata["analysis_details"] = dict(analysis_details)
                (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                checksums["metadata.json"] = _sha256(staging / "metadata.json")
                package_sha256 = _package_digest(checksums)
                ready = {"schema_version": RESEARCH_SCHEMA_VERSION, "video_id": source.video_id, "status": "research_ready",
                         "completed_at": completed_at, "source_package_sha256": source.package_sha256, "package_sha256": package_sha256}
                (staging / "_READY").write_text(json.dumps(ready, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                if final_dir.exists():
                    os.replace(final_dir, backup)
                os.replace(staging, final_dir)
                if backup.exists():
                    shutil.rmtree(backup)
                return PackageResult(source.video_id, final_dir, checksums, package_sha256, completed_at)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                if backup.exists() and not final_dir.exists():
                    os.replace(backup, final_dir)
                raise


def discover_transcript_sources(
    raw_transcripts_root: str | Path,
) -> list[TranscriptSourcePackage]:
    """Load every valid _READY transcript package in video-ID order."""
    root = Path(raw_transcripts_root)
    sources: list[TranscriptSourcePackage] = []

    for ready_path in sorted(root.glob("*/_READY")):
        sources.append(
            TranscriptSourcePackage.load(
                root,
                ready_path.parent.name,
            )
        )

    return sources


def plan_research_sources(
    sources: Sequence[TranscriptSourcePackage],
    manifest: ResearchManifestStore,
    processed_research_root: str | Path,
) -> dict[str, Any]:
    """Plan research work without modifying manifests or packages."""
    processed_root = Path(processed_research_root)

    selected_sources: list[TranscriptSourcePackage] = []
    selected_video_ids: list[str] = []
    completed_video_ids: list[str] = []
    in_progress_video_ids: list[str] = []
    retry_video_ids: list[str] = []
    changed_source_video_ids: list[str] = []
    stale_package_video_ids: list[str] = []

    seen_video_ids: set[str] = set()

    for source in sources:
        if source.video_id in seen_video_ids:
            raise ValueError(
                f"Duplicate transcript source: {source.video_id}"
            )
        seen_video_ids.add(source.video_id)

        existing = manifest.get(source.video_id)
        status = existing.get("status") if existing else None
        manifest_source_sha = (
            existing.get("source_package_sha256")
            if existing
            else None
        )
        source_changed = (
            existing is not None
            and manifest_source_sha != source.package_sha256
        )

        ready_path = processed_root / source.video_id / "_READY"
        ready = None

        if ready_path.is_file():
            try:
                ready = json.loads(
                    ready_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                stale_package_video_ids.append(source.video_id)

        ready_matches = bool(
            ready
            and ready.get("status") == "research_ready"
            and ready.get("source_package_sha256")
            == source.package_sha256
        )

        if ready_matches:
            completed_video_ids.append(source.video_id)
            continue

        if source_changed:
            changed_source_video_ids.append(source.video_id)

        if (
            status in {"analysis_queued", "analyzing"}
            and not source_changed
        ):
            in_progress_video_ids.append(source.video_id)
            continue

        if status == "analysis_failed" and not source_changed:
            retry_video_ids.append(source.video_id)

        if status == "research_ready" and not ready_matches:
            stale_package_video_ids.append(source.video_id)

        selected_sources.append(source)
        selected_video_ids.append(source.video_id)

    return {
        "sources": selected_sources,
        "input_count": len(sources),
        "selected_video_ids": selected_video_ids,
        "completed_video_ids": completed_video_ids,
        "in_progress_video_ids": in_progress_video_ids,
        "retry_video_ids": retry_video_ids,
        "changed_source_video_ids": changed_source_video_ids,
        "stale_package_video_ids": sorted(
            set(stale_package_video_ids)
        ),
    }


class ResearchInputPackageWriter:
    """Create deterministic, provider-neutral model input packages."""

    def __init__(self, analysis_inputs_root: str | Path):
        self.inputs_root = Path(analysis_inputs_root)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = round(float(seconds) * 1000)
        minutes, milliseconds = divmod(
            total_milliseconds,
            60_000,
        )
        whole_seconds, milliseconds = divmod(milliseconds, 1000)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return (
                f"{hours:02d}:{minutes:02d}:"
                f"{whole_seconds:02d}.{milliseconds:03d}"
            )

        return (
            f"{minutes:02d}:{whole_seconds:02d}."
            f"{milliseconds:03d}"
        )

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "citations": [
                {
                    "citation_id": "C1",
                    "start_segment": 0,
                    "end_segment": 0,
                }
            ],
            "research": {
                "executive_summary": "string",
                "executive_summary_citation_ids": ["C1"],
                "key_takeaways": [
                    {
                        "text": "string",
                        "citation_ids": ["C1"],
                    }
                ],
                "sections": [
                    {
                        "heading": "string",
                        "summary": "string",
                        "citation_ids": ["C1"],
                    }
                ],
                "formulas": [
                    {
                        "formula_id": "snake_case_identifier",
                        "name": "string",
                        "ascii": "machine-readable expression",
                        "latex": "LaTeX expression",
                        "derivation_type": "stated or derived",
                        "variables": [
                            {
                                "symbol": "string",
                                "meaning": "string",
                                "unit": "string or empty",
                            }
                        ],
                        "derivation_steps": ["string"],
                        "citation_ids": ["C1"],
                    }
                ],
                "caveats": ["string"],
            },
        }

    def _render_prompt(
        self,
        source: TranscriptSourcePackage,
    ) -> str:
        title = str(
            source.metadata.get("title") or source.video_id
        )
        channel = str(
            source.metadata.get("channel") or "Unknown"
        )

        lines = [
            "# Per-Video Research Analysis Input",
            "",
            "Analyze this video independently.",
            "",
            "## Grounding Rules",
            "",
            "- Use only the supplied transcript.",
            "- Do not add outside facts or silently correct the speaker.",
            "- Every substantive claim must cite transcript segments.",
            "- Use zero-based segment indexes.",
            "- Keep citation ranges as narrow as practical.",
            "- Distinguish formulas explicitly stated by the speaker "
            "from formulas derived from transcript-supported statements.",
            "- Do not invent a formula when the transcript does not "
            "support one.",
            "- Return one JSON object matching `output_schema`.",
            "",
            "## Source",
            "",
            f"- Video ID: `{source.video_id}`",
            f"- Title: {title}",
            f"- Channel: {channel}",
            f"- Source package SHA-256: "
            f"`{source.package_sha256}`",
            "",
            "## Transcript Segments",
            "",
        ]

        for index, segment in enumerate(source.segments):
            start = self._format_timestamp(segment["start"])
            end = self._format_timestamp(segment["end"])
            segment_text = str(segment["text"]).strip()

            lines.append(
                f"[S{index:04d} | {start}-{end}] {segment_text}"
            )

        return "\n".join(lines).rstrip() + "\n"

    def write(
        self,
        source: TranscriptSourcePackage,
    ) -> PackageResult:
        validate_video_id(source.video_id)

        self.inputs_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.inputs_root / source.video_id
        staging_dir = self.inputs_root / (
            f".{source.video_id}.{uuid.uuid4().hex}.staging"
        )
        backup_dir = self.inputs_root / (
            f".{source.video_id}.{uuid.uuid4().hex}.backup"
        )
        lock_path = self.inputs_root / ".locks" / (
            f"{source.video_id}.lock"
        )
        completed_at = utc_now()

        with _exclusive_lock(lock_path):
            staging_dir.mkdir()

            try:
                segments = [
                    {
                        "segment_index": index,
                        "segment_id": f"S{index:04d}",
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "avg_logprob": segment["avg_logprob"],
                    }
                    for index, segment in enumerate(
                        source.segments
                    )
                ]

                input_payload = {
                    "schema_version": RESEARCH_SCHEMA_VERSION,
                    "video_id": source.video_id,
                    "title": source.metadata.get("title"),
                    "channel": source.metadata.get("channel"),
                    "canonical_url": (
                        source.metadata.get("canonical_url")
                        or canonical_youtube_url(source.video_id)
                    ),
                    "source_package_sha256": (
                        source.package_sha256
                    ),
                    "quality": dict(source.quality),
                    "segment_count": len(segments),
                    "segments": segments,
                    "output_schema": self.output_schema(),
                }

                (staging_dir / "analysis_input.json").write_text(
                    json.dumps(
                        input_payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

                (staging_dir / "analysis_prompt.md").write_text(
                    self._render_prompt(source),
                    encoding="utf-8",
                    newline="\n",
                )

                artifact_names = [
                    "analysis_input.json",
                    "analysis_prompt.md",
                ]
                checksums = {
                    name: _sha256(staging_dir / name)
                    for name in artifact_names
                }

                metadata = {
                    "schema_version": RESEARCH_SCHEMA_VERSION,
                    "video_id": source.video_id,
                    "title": source.metadata.get("title"),
                    "source_package_sha256": (
                        source.package_sha256
                    ),
                    "source_directory": str(source.directory),
                    "segment_count": len(segments),
                    "completed_at": completed_at,
                    "artifact_sha256": checksums,
                }

                (staging_dir / "metadata.json").write_text(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

                checksums["metadata.json"] = _sha256(
                    staging_dir / "metadata.json"
                )
                package_sha256 = _package_digest(checksums)

                (staging_dir / "_READY").write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                RESEARCH_SCHEMA_VERSION
                            ),
                            "video_id": source.video_id,
                            "status": "analysis_input_ready",
                            "completed_at": completed_at,
                            "source_package_sha256": (
                                source.package_sha256
                            ),
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

                return PackageResult(
                    source.video_id,
                    final_dir,
                    checksums,
                    package_sha256,
                    completed_at,
                )

            except Exception:
                shutil.rmtree(
                    staging_dir,
                    ignore_errors=True,
                )

                if backup_dir.exists() and not final_dir.exists():
                    os.replace(backup_dir, final_dir)

                raise

