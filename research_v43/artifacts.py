"""Atomic diagnostic artifact writer for research pipeline v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


class ArtifactWriteError(RuntimeError):
    """Raised when diagnostic package writing fails."""


_REQUIRED_PAYLOADS = {
    "calculation_inventory.json",
    "formulas.json",
    "formula_entailment.json",
    "formula_coverage.json",
    "rejected_formulas.json",
    "model_invocations.json",
}


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    package_dir: Path
    package_sha256: str
    artifact_sha256: Mapping[str, str]


def write_diagnostic_package(
    *,
    output_root: str | Path,
    video_id: str,
    source_package_sha256: str,
    prompt_version: str,
    payloads: Mapping[str, Any],
) -> ArtifactWriteResult:
    """Write a complete package atomically and write _READY last."""

    if not video_id.strip():
        raise ArtifactWriteError("video_id cannot be empty")
    if not source_package_sha256.strip():
        raise ArtifactWriteError(
            "source_package_sha256 cannot be empty"
        )
    if not prompt_version.strip():
        raise ArtifactWriteError("prompt_version cannot be empty")

    missing = _REQUIRED_PAYLOADS - set(payloads)
    unexpected = set(payloads) - _REQUIRED_PAYLOADS
    if missing or unexpected:
        raise ArtifactWriteError(
            "Diagnostic payload set is invalid; "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / video_id

    temp_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{video_id}.v43.",
            dir=root,
        )
    )
    temp_package = temp_parent / video_id
    temp_package.mkdir()

    backup = root / f".{video_id}.backup"
    if backup.exists():
        shutil.rmtree(backup)

    try:
        artifact_hashes: dict[str, str] = {}
        for name in sorted(_REQUIRED_PAYLOADS):
            path = temp_package / name
            serialized = (
                json.dumps(
                    payloads[name],
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            path.write_text(serialized, encoding="utf-8")
            _fsync_file(path)
            artifact_hashes[name] = _sha256_file(path)

        metadata = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_package_sha256,
            "prompt_version": prompt_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": artifact_hashes,
        }
        metadata_path = temp_package / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(metadata_path)
        artifact_hashes["metadata.json"] = _sha256_file(metadata_path)

        package_sha = _package_digest(artifact_hashes)
        ready = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_package_sha256,
            "prompt_version": prompt_version,
            "package_sha256": package_sha,
            "artifact_sha256": artifact_hashes,
            "ready_at": datetime.now(timezone.utc).isoformat(),
        }
        ready_path = temp_package / "_READY"
        ready_path.write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(ready_path)
        _fsync_dir(temp_package)

        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temp_package, target)
            _fsync_dir(root)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)

        return ArtifactWriteResult(
            package_dir=target,
            package_sha256=package_sha,
            artifact_sha256=dict(artifact_hashes),
        )
    except Exception as exc:
        raise ArtifactWriteError(
            f"Failed to write v4.3 diagnostic package: {exc}"
        ) from exc
    finally:
        if temp_parent.exists():
            shutil.rmtree(temp_parent, ignore_errors=True)


def verify_diagnostic_package(
    package_dir: str | Path,
    *,
    source_package_sha256: str | None = None,
    prompt_version: str | None = None,
) -> list[str]:
    """Return package-integrity issues without raising for normal failures."""

    package = Path(package_dir)
    issues: list[str] = []
    required_files = _REQUIRED_PAYLOADS | {"metadata.json", "_READY"}
    for name in sorted(required_files):
        if not (package / name).is_file():
            issues.append(f"Missing artifact: {name}")
    if issues:
        return issues

    try:
        metadata = json.loads(
            (package / "metadata.json").read_text(encoding="utf-8")
        )
        ready = json.loads(
            (package / "_READY").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Invalid metadata or _READY: {exc}"]

    if source_package_sha256 is not None:
        if metadata.get("source_package_sha256") != source_package_sha256:
            issues.append("metadata source SHA is stale")
        if ready.get("source_package_sha256") != source_package_sha256:
            issues.append("_READY source SHA is stale")
    if prompt_version is not None:
        if metadata.get("prompt_version") != prompt_version:
            issues.append("metadata prompt version is stale")
        if ready.get("prompt_version") != prompt_version:
            issues.append("_READY prompt version is stale")

    metadata_hashes = metadata.get("artifact_sha256")
    if not isinstance(metadata_hashes, Mapping):
        issues.append("metadata artifact hashes are missing")
        return issues
    expected_hashes = ready.get("artifact_sha256")
    if not isinstance(expected_hashes, Mapping):
        issues.append("_READY artifact hashes are missing")
        return issues
    for name, expected in metadata_hashes.items():
        if expected_hashes.get(name) != expected:
            issues.append(
                f"metadata and _READY hashes disagree: {name}"
            )

    actual_hashes: dict[str, str] = {}
    for name, expected in expected_hashes.items():
        path = package / str(name)
        if not path.is_file():
            issues.append(f"Missing hashed artifact: {name}")
            continue
        actual = _sha256_file(path)
        actual_hashes[str(name)] = actual
        if actual != expected:
            issues.append(f"Artifact hash mismatch: {name}")

    actual_package_sha = _package_digest(actual_hashes)
    if ready.get("package_sha256") != actual_package_sha:
        issues.append("Package SHA mismatch")
    return issues


def _package_digest(artifact_hashes: Mapping[str, str]) -> str:
    material = "\n".join(
        f"{name}:{artifact_hashes[name]}"
        for name in sorted(artifact_hashes)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
