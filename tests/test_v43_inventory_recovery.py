from __future__ import annotations

import json
import unittest

from research_v43.inventory_recovery import (
    AdaptiveInventoryOllamaClient,
    normalize_inventory_order,
    parse_inventory_response_with_order_repair,
)


def item(calculation_id, start, end):
    return {
        "calculation_id": calculation_id,
        "name": calculation_id,
        "source_mode": "spoken",
        "start_segment": start,
        "end_segment": end,
        "variables_mentioned": [],
        "operations_mentioned": ["addition"],
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "The source states arithmetic.",
    }


class SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post_json(self, *, url, payload, timeout_seconds):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("Unexpected transport call")
        return self.responses.pop(0)


class InventoryRecoveryTests(unittest.TestCase):
    def test_normalizes_only_item_order(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                item("CALC_0002", 9, 9),
                item("CALC_0001", 3, 4),
            ],
        }
        repaired = normalize_inventory_order(payload)
        self.assertEqual(
            [entry["calculation_id"] for entry in repaired["calculations"]],
            ["CALC_0001", "CALC_0002"],
        )
        self.assertEqual(repaired["calculations"][0]["start_segment"], 3)
        self.assertEqual(repaired["calculations"][1]["start_segment"], 9)

    def test_parser_repairs_out_of_order_array(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                item("CALC_0002", 9, 9),
                item("CALC_0001", 3, 4),
            ],
        }
        parsed = parse_inventory_response_with_order_repair(
            json.dumps(payload),
            expected_video_id="video-123",
            maximum_segment=10,
        )
        self.assertEqual(
            [entry.calculation_id for entry in parsed.calculations],
            ["CALC_0001", "CALC_0002"],
        )

    def test_non_order_validation_errors_still_fail(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "wrong-video",
            "calculations": [item("CALC_0001", 3, 4)],
        }
        with self.assertRaises(ValueError):
            parse_inventory_response_with_order_repair(
                json.dumps(payload),
                expected_video_id="video-123",
                maximum_segment=10,
            )

    def test_inventory_length_exhaustion_retries_once_with_larger_budget(self):
        valid_content = json.dumps(
            {
                "schema_version": "1.0",
                "video_id": "video-123",
                "calculations": [],
            }
        )
        transport = SequenceTransport(
            [
                {
                    "message": {"content": "", "thinking": "x" * 100},
                    "done_reason": "length",
                    "eval_count": 1536,
                },
                {
                    "message": {"content": valid_content, "thinking": ""},
                    "done_reason": "stop",
                    "eval_count": 50,
                },
            ]
        )
        client = AdaptiveInventoryOllamaClient(
            model="qwen3:8b",
            think=True,
            num_ctx=8192,
            num_predict=1536,
            transport=transport,
        )
        response = client.complete_json(
            system_prompt="Return JSON.",
            user_prompt="Inventory this transcript.",
            stage="calculation_inventory chunk 1/2 segments 0-39",
            num_predict=1536,
        )
        self.assertEqual(response.payload["video_id"], "video-123")
        self.assertEqual(len(transport.payloads), 2)
        self.assertEqual(
            transport.payloads[0]["options"]["num_predict"],
            1536,
        )
        self.assertEqual(
            transport.payloads[1]["options"]["num_predict"],
            3072,
        )

    def test_non_inventory_length_exhaustion_does_not_retry(self):
        transport = SequenceTransport(
            [
                {
                    "message": {"content": "", "thinking": "x" * 100},
                    "done_reason": "length",
                    "eval_count": 1536,
                }
            ]
        )
        client = AdaptiveInventoryOllamaClient(
            model="qwen3:8b",
            think=True,
            num_ctx=8192,
            num_predict=1536,
            transport=transport,
        )
        with self.assertRaises(Exception):
            client.complete_json(
                system_prompt="Return JSON.",
                user_prompt="Extract a formula.",
                stage="formula_extraction CALC_0001",
                num_predict=1536,
            )
        self.assertEqual(len(transport.payloads), 1)


if __name__ == "__main__":
    unittest.main()
