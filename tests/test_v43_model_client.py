from __future__ import annotations

import json
import unittest

from research_v43.model_client import (
    ModelClientError,
    OllamaJsonClient,
)


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, *, url, payload, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        return self.response


class ModelClientTests(unittest.TestCase):
    def test_sends_json_only_deterministic_request(self):
        transport = FakeTransport(
            {
                "message": {
                    "content": json.dumps({"answer": "ok"})
                }
            }
        )
        client = OllamaJsonClient(
            transport=transport,
            model="test-model",
            think=True,
            num_ctx=4096,
        )

        response = client.complete_json(
            system_prompt="System",
            user_prompt="User",
        )

        self.assertEqual(response.payload, {"answer": "ok"})
        _, payload, _ = transport.calls[0]
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["num_predict"], 1536)
        self.assertEqual(payload["keep_alive"], "30m")

    def test_rejects_non_json_message_content(self):
        client = OllamaJsonClient(
            transport=FakeTransport(
                {"message": {"content": "not-json"}}
            )
        )
        with self.assertRaisesRegex(
            ModelClientError,
            "not valid JSON",
        ):
            client.complete_json(
                system_prompt="System",
                user_prompt="User",
            )


    def test_stage_is_included_in_transport_failure(self):
        class FailingTransport:
            def post_json(self, **kwargs):
                raise ModelClientError("Model HTTP request failed: timed out")

        client = OllamaJsonClient(transport=FailingTransport())
        with self.assertRaisesRegex(
            ModelClientError,
            "inventory chunk 1/3 failed",
        ):
            client.complete_json(
                system_prompt="System",
                user_prompt="User",
                stage="inventory chunk 1/3",
            )

    def test_empty_content_reports_ollama_generation_metadata(self):
        client = OllamaJsonClient(
            transport=FakeTransport(
                {
                    "message": {
                        "content": "",
                        "thinking": "internal reasoning",
                    },
                    "done_reason": "length",
                    "eval_count": 1536,
                }
            )
        )
        with self.assertRaisesRegex(
            ModelClientError,
            r"done_reason=length.*eval_count=1536.*thinking_chars=18",
        ):
            client.complete_json(
                system_prompt="System",
                user_prompt="User",
                stage="formula_entailment CALC_0001/example",
            )


if __name__ == "__main__":
    unittest.main()
