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


if __name__ == "__main__":
    unittest.main()
