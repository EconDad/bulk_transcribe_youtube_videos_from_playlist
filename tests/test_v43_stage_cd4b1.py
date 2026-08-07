from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.artifacts import (
    verify_diagnostic_package,
    write_diagnostic_package,
)
from research_v43.calculation_inventory import (
    CalculationInventory,
    audit_visual_equation_cues,
    has_visual_equation_cue,
)
from research_v43.entailment import _has_operation_cue
from research_v43.model_client import JsonModelResponse, ModelInvocation
from run_research_v43 import run_pipeline


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.think = True

    def complete_json(self, *, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("Unexpected model call")
        return JsonModelResponse(
            payload=self.payloads.pop(0),
            invocation=ModelInvocation(
                model="fake",
                think=bool(kwargs.get("think", True)),
                num_ctx=8192,
                prompt_chars=len(system_prompt) + len(user_prompt),
                response_chars=10,
                elapsed_seconds=0.0,
            ),
        )


def visual_inventory_payload():
    return {
        "schema_version": "1.0",
        "video_id": "video-123",
        "calculations": [
            {
                "calculation_id": "CALC_0001",
                "name": "Referenced equation",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": [],
                "operations_mentioned": [],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The speaker refers to an equation.",
            }
        ],
    }


def write_source(root: Path, video_id: str, texts: list[str]) -> None:
    package = root / video_id
    package.mkdir(parents=True)
    (package / "transcript.json").write_text(
        json.dumps(
            [
                {"start": index, "end": index + 1, "text": text}
                for index, text in enumerate(texts)
            ]
        ),
        encoding="utf-8",
    )
    (package / "_READY").write_text(
        json.dumps({"package_sha256": "source-sha"}),
        encoding="utf-8",
    )
    (package / "metadata.json").write_text(
        json.dumps({"title": "Synthetic test"}),
        encoding="utf-8",
    )


def base_payloads():
    return {
        "calculation_inventory.json": {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [],
        },
        "formulas.json": {
            "schema_version": "1.0",
            "video_id": "video-123",
            "formulas": [],
        },
        "formula_entailment.json": {
            "schema_version": "1.0",
            "video_id": "video-123",
            "reports": [],
        },
        "formula_coverage.json": {
            "schema_version": "1.0",
            "video_id": "video-123",
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
            "video_id": "video-123",
            "rejected_formulas": [],
        },
        "model_invocations.json": {
            "schema_version": "1.0",
            "video_id": "video-123",
            "source_title": "Synthetic test",
            "invocations": [],
        },
    }


class StageCD4B1Tests(unittest.TestCase):
    def test_dividing_is_a_division_cue(self):
        self.assertTrue(
            _has_operation_cue(
                "division",
                "we are dividing it by the price",
            )
        )

    def test_visual_cue_phrases_are_detected(self):
        for text in (
            "Here's the equation.",
            "Here is the formula.",
            "The formula is shown here.",
            "Look at this equation.",
            "The equation on the screen gives the relationship.",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_visual_equation_cue(text))

    def test_nonvisual_equation_language_is_not_promoted(self):
        for text in (
            "We will derive an equation later.",
            "An equation describes the relationship.",
            "The formula can be complex.",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_visual_equation_cue(text))

    def test_inventory_audit_promotes_direct_visual_cue(self):
        inventory = CalculationInventory.from_mapping(
            visual_inventory_payload()
        )
        audited, records = audit_visual_equation_cues(
            inventory=inventory,
            segments=[{"text": "Here's the equation."}],
        )
        item = audited.calculations[0]
        self.assertTrue(item.visual_equation_cue)
        self.assertEqual(item.source_mode.value, "visual_cue")
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["action"],
            "promoted_to_visual_review",
        )

    def test_optional_inventory_audit_is_written_and_verified(self):
        payloads = base_payloads()
        payloads["inventory_audit.json"] = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "audit_version": "fixture",
            "records": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            result = write_diagnostic_package(
                output_root=directory,
                video_id="video-123",
                source_package_sha256="source-sha",
                prompt_version="fixture",
                payloads=payloads,
            )
            self.assertTrue(
                (result.package_dir / "inventory_audit.json").is_file()
            )
            self.assertIn(
                "inventory_audit.json",
                result.artifact_sha256,
            )
            self.assertEqual(
                verify_diagnostic_package(result.package_dir),
                [],
            )

    def test_runner_routes_audited_cue_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "Raw Transcripts"
            write_source(
                raw_root,
                "video-123",
                ["Here's the equation."],
            )
            client = FakeClient([visual_inventory_payload()])

            exit_code, package = run_pipeline(
                video_id="video-123",
                client=client,
                raw_root=raw_root,
                output_root=base / "Diagnostics",
                progress_root=base / "Progress",
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(client.calls, 1)

            coverage = json.loads(
                (package / "formula_coverage.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(coverage["visual_review_required"], 1)

            audit = json.loads(
                (package / "inventory_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(audit["records"]), 1)


if __name__ == "__main__":
    unittest.main()
