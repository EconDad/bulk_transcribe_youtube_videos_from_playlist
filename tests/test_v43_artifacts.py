from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.artifacts import (
    verify_diagnostic_package,
    write_diagnostic_package,
)


def payloads():
    return {
        "calculation_inventory.json": {"calculations": []},
        "formulas.json": {"formulas": []},
        "formula_entailment.json": {"reports": []},
        "formula_coverage.json": {"passed": True},
        "rejected_formulas.json": {"rejected_formulas": []},
        "model_invocations.json": {"invocations": []},
    }


class ArtifactTests(unittest.TestCase):
    def test_writes_and_verifies_atomic_package(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_diagnostic_package(
                output_root=directory,
                video_id="video-123",
                source_package_sha256="source-sha",
                prompt_version="prompt-v1",
                payloads=payloads(),
            )
            self.assertTrue((result.package_dir / "_READY").is_file())
            self.assertEqual(
                verify_diagnostic_package(
                    result.package_dir,
                    source_package_sha256="source-sha",
                    prompt_version="prompt-v1",
                ),
                [],
            )

    def test_detects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_diagnostic_package(
                output_root=directory,
                video_id="video-123",
                source_package_sha256="source-sha",
                prompt_version="prompt-v1",
                payloads=payloads(),
            )
            path = result.package_dir / "formulas.json"
            path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            issues = verify_diagnostic_package(result.package_dir)
            self.assertTrue(
                any("hash mismatch" in issue.lower() for issue in issues)
            )

    def test_detects_stale_source_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_diagnostic_package(
                output_root=directory,
                video_id="video-123",
                source_package_sha256="source-sha",
                prompt_version="prompt-v1",
                payloads=payloads(),
            )
            issues = verify_diagnostic_package(
                result.package_dir,
                source_package_sha256="new-source",
                prompt_version="prompt-v2",
            )
            self.assertIn("metadata source SHA is stale", issues)
            self.assertIn("_READY prompt version is stale", issues)


if __name__ == "__main__":
    unittest.main()
