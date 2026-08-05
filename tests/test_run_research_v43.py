from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.expression_ast import FormulaCandidate
from research_v43.model_client import JsonModelResponse, ModelInvocation
from run_research_v43 import run_pipeline


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, *, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("Unexpected model call")
        return JsonModelResponse(
            payload=self.payloads.pop(0),
            invocation=ModelInvocation(
                model="fake-model",
                think=True,
                num_ctx=8192,
                prompt_chars=len(system_prompt) + len(user_prompt),
                response_chars=10,
            ),
        )


def write_source(root: Path, video_id: str, texts):
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


def candidate_payload():
    return {
        "calculation_id": "CALC_0001",
        "formula_id": "normalized_measurement",
        "name": "Normalized measurement",
        "ascii": "normalized_measurement = total_value / item_count",
        "latex": r"m=\frac{T}{n}",
        "derivation_type": "stated",
        "variables": [
            {
                "symbol": "normalized_measurement",
                "meaning": "normalized measurement",
                "unit": "units per item",
            },
            {
                "symbol": "total_value",
                "meaning": "total value",
                "unit": "units",
            },
            {
                "symbol": "item_count",
                "meaning": "item count",
                "unit": "items",
            },
        ],
        "derivation_steps": [
            "Divide the total value by the item count."
        ],
        "source_claims": [
            {
                "start_segment": 0,
                "end_segment": 0,
                "relationship": "division",
            }
        ],
    }


class RunnerTests(unittest.TestCase):
    def test_complete_spoken_formula_pipeline(self):
        candidate = FormulaCandidate.from_mapping(candidate_payload())
        node = candidate.parsed.operations[0]
        inventory = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                {
                    "calculation_id": "CALC_0001",
                    "name": "Normalize a measurement",
                    "source_mode": "spoken",
                    "start_segment": 0,
                    "end_segment": 0,
                    "variables_mentioned": ["total value", "item count"],
                    "operations_mentioned": ["division"],
                    "visual_equation_cue": False,
                    "formula_expected": True,
                    "reason": "The speaker states a division.",
                }
            ],
        }
        extraction = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "The source states the relationship.",
            "candidates": [candidate_payload()],
        }
        entailment = {
            "calculation_id": "CALC_0001",
            "formula_id": "normalized_measurement",
            "nodes": [
                {
                    "node_id": node.node_id,
                    "expression": node.expression,
                    "operation": node.operation,
                    "status": "entailed",
                    "evidence": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": (
                                "Divide the total value by the item count."
                            ),
                        }
                    ],
                    "depends_on_node_ids": [],
                    "derivation_step": "",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "Raw Transcripts"
            output_root = base / "Diagnostics"
            write_source(
                raw_root,
                "video-123",
                ["Divide the total value by the item count."],
            )
            client = FakeClient([inventory, extraction, entailment])
            exit_code, package = run_pipeline(
                video_id="video-123",
                client=client,
                raw_root=raw_root,
                output_root=output_root,
                progress_root=base / "Progress",
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(client.calls, 3)
            coverage = json.loads(
                (package / "formula_coverage.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(coverage["passed"])
            formulas = json.loads(
                (package / "formulas.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(formulas["formulas"]), 1)

    def test_visual_cue_is_persisted_as_review_required(self):
        inventory = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                {
                    "calculation_id": "CALC_0001",
                    "name": "Displayed equation",
                    "source_mode": "visual_cue",
                    "start_segment": 0,
                    "end_segment": 0,
                    "variables_mentioned": [],
                    "operations_mentioned": [],
                    "visual_equation_cue": True,
                    "formula_expected": True,
                    "reason": "The speaker announces an equation.",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "Raw Transcripts"
            write_source(raw_root, "video-123", ["Here is the equation."])
            client = FakeClient([inventory])
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
            self.assertFalse(coverage["passed"])


    def test_chunked_inventory_resumes_without_model_calls(self):
        empty_first = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [],
        }
        empty_second = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "Raw Transcripts"
            output_root = base / "Diagnostics"
            progress_root = base / "Progress"
            write_source(
                raw_root,
                "video-123",
                [f"Conceptual segment {index}." for index in range(5)],
            )
            first_client = FakeClient([empty_first, empty_second])
            exit_code, _ = run_pipeline(
                video_id="video-123",
                client=first_client,
                raw_root=raw_root,
                output_root=output_root,
                progress_root=progress_root,
                inventory_chunk_segments=3,
                inventory_overlap_segments=1,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(first_client.calls, 2)

            second_client = FakeClient([])
            exit_code, _ = run_pipeline(
                video_id="video-123",
                client=second_client,
                raw_root=raw_root,
                output_root=output_root,
                progress_root=progress_root,
                inventory_chunk_segments=3,
                inventory_overlap_segments=1,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(second_client.calls, 0)


if __name__ == "__main__":
    unittest.main()
