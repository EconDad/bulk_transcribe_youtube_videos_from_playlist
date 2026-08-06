from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_v43.calculation_inventory import CalculationItem
from research_v43.entailment import (
    build_entailment_prompt,
    validate_entailment_response,
)
from research_v43.expression_ast import FormulaCandidate
from research_v43.formula_extraction import build_formula_extraction_prompt
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


def write_source(root: Path, video_id: str, texts: list[str]) -> None:
    package = root / video_id
    package.mkdir(parents=True)
    segments = [
        {
            "segment_id": index,
            "start": float(index),
            "end": float(index + 1),
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    (package / "transcript.json").write_text(
        json.dumps({"segments": segments}),
        encoding="utf-8",
    )
    (package / "metadata.json").write_text(
        json.dumps({"video_id": video_id, "title": "Fixture"}),
        encoding="utf-8",
    )


def candidate_payload():
    return {
        "calculation_id": "CALC_0001",
        "formula_id": "normalized_measurement",
        "name": "Normalized measurement",
        "ascii": "normalized_measurement = total_value / item_count",
        "latex": "m=T/n",
        "derivation_type": "stated",
        "variables": [
            {
                "symbol": "normalized_measurement",
                "meaning": "normalized measurement",
                "unit": "units",
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


def inventory_payload(*, formula_expected=True):
    return {
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
                "formula_expected": formula_expected,
                "reason": "The source states a division.",
            }
        ],
    }


def valid_entailment(candidate):
    node = candidate.parsed.operations[0]
    sentence = (
        "Divide the total value by the item count to get the "
        "normalized measurement."
    )
    return {
        "calculation_id": "CALC_0001",
        "formula_id": candidate.formula_id,
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
                        "quote": sentence,
                    }
                ],
                "identifier_groundings": [
                    {
                        "identifier": "total_value",
                        "start_segment": 0,
                        "end_segment": 0,
                        "quote": "total value",
                    },
                    {
                        "identifier": "item_count",
                        "start_segment": 0,
                        "end_segment": 0,
                        "quote": "item count",
                    },
                    {
                        "identifier": "normalized_measurement",
                        "start_segment": 0,
                        "end_segment": 0,
                        "quote": "normalized measurement",
                    },
                ],
                "depends_on_node_ids": [],
                "derivation_step": "",
            }
        ],
    }


class StageCD4ATests(unittest.TestCase):
    def test_nonvisual_prompt_excludes_visual_disposition(self):
        item = CalculationItem.from_mapping(
            inventory_payload()["calculations"][0]
        )
        prompt = build_formula_extraction_prompt(
            item=item,
            segments=[
                {"text": "Divide the total value by the item count."}
            ],
        )
        schema = prompt.split("RESPONSE SCHEMA:", 1)[1].split(
            "SOURCE SEGMENTS:", 1
        )[0]
        self.assertNotIn("visual_review_required", schema)
        self.assertIn("left of '='", prompt)
        self.assertIn("exactly one variable definition", prompt)

    def test_formula_expected_false_skips_downstream_models(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "Raw Transcripts"
            write_source(raw, "video-123", ["A numerical example."])
            client = FakeClient(
                [inventory_payload(formula_expected=False)]
            )
            exit_code, package = run_pipeline(
                video_id="video-123",
                client=client,
                raw_root=raw,
                output_root=base / "Diagnostics",
                progress_root=base / "Progress",
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(client.calls, 1)
            coverage = json.loads(
                (package / "formula_coverage.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(coverage["non_symbolic_calculations"], 1)

    def test_invalid_extraction_gets_one_repair(self):
        invalid_candidate = candidate_payload()
        invalid_candidate["variables"] = [
            entry
            for entry in invalid_candidate["variables"]
            if entry["symbol"] != "normalized_measurement"
        ]
        invalid = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "Candidate proposed.",
            "candidates": [invalid_candidate],
        }
        repaired = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "Candidate repaired.",
            "candidates": [candidate_payload()],
        }
        candidate = FormulaCandidate.from_mapping(candidate_payload())
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "Raw Transcripts"
            write_source(
                raw,
                "video-123",
                [
                    "Divide the total value by the item count to get "
                    "the normalized measurement."
                ],
            )
            client = FakeClient(
                [
                    inventory_payload(),
                    invalid,
                    repaired,
                    valid_entailment(candidate),
                ]
            )
            exit_code, _ = run_pipeline(
                video_id="video-123",
                client=client,
                raw_root=raw,
                output_root=base / "Diagnostics",
                progress_root=base / "Progress",
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(client.calls, 4)

    def test_altered_ast_expression_gets_one_repair(self):
        extraction = {
            "calculation_id": "CALC_0001",
            "disposition": "candidates_proposed",
            "reason": "Candidate proposed.",
            "candidates": [candidate_payload()],
        }
        candidate = FormulaCandidate.from_mapping(candidate_payload())
        altered = valid_entailment(candidate)
        altered["nodes"][0]["expression"] = (
            "normalized_measurement = total_value / item_count"
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "Raw Transcripts"
            write_source(
                raw,
                "video-123",
                [
                    "Divide the total value by the item count to get "
                    "the normalized measurement."
                ],
            )
            client = FakeClient(
                [
                    inventory_payload(),
                    extraction,
                    altered,
                    valid_entailment(candidate),
                ]
            )
            exit_code, _ = run_pipeline(
                video_id="video-123",
                client=client,
                raw_root=raw,
                output_root=base / "Diagnostics",
                progress_root=base / "Progress",
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(client.calls, 4)

    def test_entailment_prompt_contains_exact_ast_templates(self):
        item = CalculationItem.from_mapping(
            inventory_payload()["calculations"][0]
        )
        candidate = FormulaCandidate.from_mapping(candidate_payload())
        node = candidate.parsed.operations[0]
        prompt = build_entailment_prompt(
            item=item,
            candidate=candidate,
            segments=[
                {
                    "text": (
                        "Divide the total value by the item count to get "
                        "the normalized measurement."
                    )
                }
            ],
        )
        self.assertIn(f'"node_id": "{node.node_id}"', prompt)
        self.assertIn(f'"expression": "{node.expression}"', prompt)
        self.assertIn(
            "never add the formula's left-hand assignment",
            prompt,
        )

    def test_redundant_dependency_grounding_is_allowed(self):
        sentence = (
            "Add one to the rate, then multiply the principal by "
            "that amount to get the result."
        )
        item = CalculationItem.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "name": "Growth calculation",
                "source_mode": "spoken",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": ["rate", "principal", "result"],
                "operations_mentioned": ["addition", "multiplication"],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "The source states two operations.",
            }
        )
        candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "growth",
                "name": "Growth",
                "ascii": "result = principal * (1 + rate)",
                "latex": "r=p(1+i)",
                "derivation_type": "stated",
                "variables": [
                    {"symbol": "result", "meaning": "result", "unit": "units"},
                    {"symbol": "principal", "meaning": "principal", "unit": "units"},
                    {"symbol": "rate", "meaning": "rate", "unit": ""},
                ],
                "derivation_steps": [
                    "Add one to the rate.",
                    "Multiply by the principal.",
                ],
                "source_claims": [
                    {
                        "start_segment": 0,
                        "end_segment": 0,
                        "relationship": "two operations",
                    }
                ],
            }
        )
        first, second = candidate.parsed.operations
        payload = {
            "calculation_id": "CALC_0001",
            "formula_id": "growth",
            "nodes": [
                {
                    "node_id": first.node_id,
                    "expression": first.expression,
                    "operation": first.operation,
                    "status": "entailed",
                    "evidence": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "Add one to the rate",
                        }
                    ],
                    "identifier_groundings": [
                        {
                            "identifier": "rate",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "rate",
                        }
                    ],
                    "depends_on_node_ids": [],
                    "derivation_step": "",
                },
                {
                    "node_id": second.node_id,
                    "expression": second.expression,
                    "operation": second.operation,
                    "status": "entailed",
                    "evidence": [
                        {
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": (
                                "multiply the principal by that amount "
                                "to get the result"
                            ),
                        }
                    ],
                    "identifier_groundings": [
                        {
                            "identifier": "principal",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "principal",
                        },
                        {
                            "identifier": "rate",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "rate",
                        },
                        {
                            "identifier": "result",
                            "start_segment": 0,
                            "end_segment": 0,
                            "quote": "result",
                        },
                    ],
                    "depends_on_node_ids": [first.node_id],
                    "derivation_step": "",
                },
            ],
        }
        report = validate_entailment_response(
            payload,
            item=item,
            candidate=candidate,
            segments=[{"text": sentence}],
        )
        self.assertTrue(report.passed, report.issues)


if __name__ == "__main__":
    unittest.main()
