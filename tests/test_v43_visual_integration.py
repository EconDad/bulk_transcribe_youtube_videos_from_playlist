from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.expression_ast import FormulaCandidate
from research_v43.model_client import JsonModelResponse, ModelInvocation
from research_v43.visual_evidence import VisualRecoveryResult
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
                model="fake-model",
                think=True,
                num_ctx=8192,
                prompt_chars=len(system_prompt) + len(user_prompt),
                response_chars=10,
            ),
        )


class FakeVisualRecoverer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def recover(self, **kwargs):
        self.calls += 1
        return self.result


def write_source(root: Path, video_id: str):
    package = root / video_id
    package.mkdir(parents=True)
    (package / "transcript.json").write_text(
        json.dumps([{"start": 100.0, "end": 105.0, "text": "Here is the equation."}]),
        encoding="utf-8",
    )
    (package / "_READY").write_text(
        json.dumps({"package_sha256": "source-sha"}), encoding="utf-8"
    )
    (package / "metadata.json").write_text(
        json.dumps({"title": "Synthetic visual test"}), encoding="utf-8"
    )


class VisualRunnerIntegrationTests(unittest.TestCase):
    def test_visual_consensus_candidate_routes_to_formula_retained(self):
        video_id = "video-visual"
        inventory = {
            "schema_version": "1.0",
            "video_id": video_id,
            "calculations": [
                {
                    "calculation_id": "CALC_0001",
                    "name": "Displayed relationship",
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
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "visual_equation",
                "name": "Displayed relationship",
                "ascii": "result = input_value / divisor",
                "latex": r"r=\frac{x}{d}",
                "derivation_type": "stated_visual",
                "variables": [
                    {"symbol": "result", "meaning": "visible result", "unit": ""},
                    {"symbol": "input_value", "meaning": "visible input", "unit": ""},
                    {"symbol": "divisor", "meaning": "visible divisor", "unit": ""},
                ],
                "derivation_steps": ["Transcribed from agreeing source-video frames."],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 0,
                        "relationship": "The transcript announces the displayed equation.",
                    }
                ],
                "visual_source": {
                    "source_url": "https://example.invalid/video",
                    "source_media_sha256": "abc",
                    "consensus_count": 3,
                },
            }
        )
        visual_result = VisualRecoveryResult(
            calculation_id="CALC_0001",
            state="formula_retained",
            reason="3/3 frames agree.",
            candidate=candidate,
            evidence={
                "schema_version": "1.0",
                "calculation_id": "CALC_0001",
                "status": "formula_retained",
                "reason": "3/3 frames agree.",
                "consensus": {"passed": True, "winner_count": 3},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "Raw Transcripts"
            write_source(raw_root, video_id)
            client = FakeClient([inventory])
            visual = FakeVisualRecoverer(visual_result)
            exit_code, package = run_pipeline(
                video_id=video_id,
                client=client,
                raw_root=raw_root,
                output_root=base / "Diagnostics",
                progress_root=base / "Progress",
                visual_recoverer=visual,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(client.calls, 1)
            self.assertEqual(visual.calls, 1)
            coverage = json.loads((package / "formula_coverage.json").read_text(encoding="utf-8"))
            self.assertTrue(coverage["passed"])
            self.assertEqual(coverage["formulas_retained"], 1)
            self.assertEqual(coverage["visual_review_required"], 0)
            self.assertEqual(coverage["resolutions"][0]["state"], "formula_retained")
            formulas = json.loads((package / "formulas.json").read_text(encoding="utf-8"))
            self.assertEqual(formulas["formulas"][0]["derivation_type"], "stated_visual")
            visual_evidence = json.loads((package / "visual_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(visual_evidence["records"][0]["status"], "formula_retained")


if __name__ == "__main__":
    unittest.main()
