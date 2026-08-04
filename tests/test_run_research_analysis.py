import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_research_analysis import (
    OllamaError,
    _normalize_chunk_surface,
    _normalize_formula_semantics,
    _prune_chunk_payload,
    _repair_formula_variables,
    _locate_formula_support,
    _validate_numeric_grounding,
    build_evidence_catalog,
    _repair_chunk_citations,
    _select_support_window,
    _validate_chunk_payload,
    _validate_formula_candidate,
    _validate_narrative_quality,
    _chunk_from_range,
    analyze_video,
    chunk_transcript,
    extract_chunk_evidence,
    safe_chunk_token_budget,
)
from youtube_research_analysis import TranscriptSourcePackage
from youtube_research_io import (
    TranscriptPackageWriter,
    TranscriptQuality,
    VideoMetadata,
)


VIDEO_ID = "qlt111AAA22"


class FakeOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.required_models = []

    def require_model(self, model):
        self.required_models.append(model)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        return {
            "message": {
                "role": "assistant",
                "content": json.dumps(payload),
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }




class ScriptedOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if item == "TRUNCATED":
            return {
                "message": {
                    "role": "assistant",
                    "content": '{"claims":[{"topic":"truncated',
                },
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 100,
                "eval_count": kwargs["num_predict"],
            }
        return {
            "message": {
                "role": "assistant",
                "content": json.dumps(item),
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

class ResearchQualityV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw_root = self.root / "Raw Transcripts"
        self.processed_root = self.root / "Processed Research"
        self.manifest_path = self.root / "manifests" / "research.jsonl"

        TranscriptPackageWriter(self.raw_root).write(
            metadata=VideoMetadata(
                video_id=VIDEO_ID,
                title="Financial Relationships",
                source_url=f"https://youtu.be/{VIDEO_ID}",
                channel="Test Channel",
                duration_seconds=50,
                transcription_backend="test",
            ),
            transcript_text=(
                "Revenue is money entering the business. "
                "Cost of revenue is subtracted from revenue. "
                "The result is gross profit. "
                "Assets minus liabilities gives total equity. "
                "Equity divided by shares gives book value per share."
            ),
            segments=[
                {"start": 0, "end": 10, "text": "Revenue is money entering the business.", "avg_logprob": -0.1},
                {"start": 10, "end": 20, "text": "Cost of revenue is subtracted from revenue.", "avg_logprob": -0.1},
                {"start": 20, "end": 30, "text": "The result is gross profit.", "avg_logprob": -0.1},
                {"start": 30, "end": 40, "text": "Assets minus liabilities gives total equity.", "avg_logprob": -0.1},
                {"start": 40, "end": 50, "text": "Equity divided by shares gives book value per share.", "avg_logprob": -0.1},
            ],
            quality=TranscriptQuality(
                quality_status="usable",
                metrics={"segment_count": 5, "word_count": 28},
            ),
        )
        self.source = TranscriptSourcePackage.load(self.raw_root, VIDEO_ID)
        self.chunk = chunk_transcript(self.source.segments, token_budget=1000)[0]

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def valid_chunk_payload():
        return {
            "claims": [
                {
                    "topic": "Gross profit",
                    "text": "Gross profit remains after cost of revenue is removed from total revenue.",
                    "explanation": "This relationship shows how revenue is reduced by the direct cost of generating it.",
                    "start_segment": 0,
                    "end_segment": 2,
                },
                {
                    "topic": "Book value per share",
                    "text": "Book value per share allocates total equity across the outstanding shares.",
                    "explanation": "It converts a whole-company balance-sheet amount into a per-share measure.",
                    "start_segment": 4,
                    "end_segment": 4,
                },
            ],
            "formulas": [
                {
                    "formula_id": "gross_profit",
                    "name": "Gross profit",
                    "description": "Gross profit equals total revenue after subtracting cost of revenue.",
                    "ascii": "gross_profit = total_revenue - cost_of_revenue",
                    "latex": r"\\text{Gross Profit}=\\text{Total Revenue}-\\text{Cost of Revenue}",
                    "derivation_type": "stated",
                    "variables": [
                        {"symbol": "gross_profit", "meaning": "Profit after direct revenue costs", "unit": "currency"},
                        {"symbol": "total_revenue", "meaning": "Money entering the business", "unit": "currency"},
                        {"symbol": "cost_of_revenue", "meaning": "Direct cost of generating revenue", "unit": "currency"},
                    ],
                    "derivation_steps": ["Subtract cost of revenue from total revenue."],
                    "start_segment": 0,
                    "end_segment": 2,
                }
            ],
            "caveats": [],
        }

    @staticmethod
    def valid_narrative():
        return {
            "executive_summary": (
                "The lesson explains how basic financial relationships connect rather than presenting each term as an isolated value. "
                "It first shows how revenue and direct costs determine gross profit. "
                "It then moves from whole-company balance-sheet value to a per-share measure."
            ),
            "executive_summary_evidence_ids": [
                "E0001",
                "E0002",
                "E0003",
            ],
            "key_takeaways": [
                {
                    "text": "Gross profit is the remainder after direct revenue costs are deducted from revenue.",
                    "evidence_ids": ["E0001"],
                },
                {
                    "text": "Book value per share allocates total equity across the outstanding shares.",
                    "evidence_ids": ["E0002"],
                },
                {
                    "text": "The gross-profit relationship connects total revenue, cost of revenue, and the resulting profit.",
                    "evidence_ids": ["E0003"],
                },
                {
                    "text": "The lesson connects whole-company financial amounts to measures that describe one share.",
                    "evidence_ids": ["E0001", "E0002"],
                },
            ],
            "sections": [
                {
                    "heading": "Revenue and direct costs",
                    "summary": "The lesson begins with revenue entering the business and explains that direct revenue costs must be removed before gross profit is known.",
                    "evidence_ids": ["E0001"],
                },
                {
                    "heading": "Gross-profit relationship",
                    "summary": "The symbolic relationship expresses gross profit as total revenue less the cost of generating that revenue.",
                    "evidence_ids": ["E0003"],
                },
                {
                    "heading": "Per-share value",
                    "summary": "The final concept divides whole-company equity across outstanding shares to produce book value per share.",
                    "evidence_ids": ["E0002"],
                },
            ],
        }

    def test_scalar_value_is_rejected_as_formula(self):
        payload = self.valid_chunk_payload()
        formula = payload["formulas"][0]
        formula["ascii"] = "65"
        formula["latex"] = "65"

        with self.assertRaisesRegex(OllamaError, "exactly one '='"):
            _validate_chunk_payload(
                payload,
                source=self.source,
                chunk=self.chunk,
            )

    def test_chunk_wide_citation_is_rejected(self):
        payload = self.valid_chunk_payload()
        payload["claims"][0]["end_segment"] = 20

        with self.assertRaises(OllamaError):
            _validate_chunk_payload(
                payload,
                source=self.source,
                chunk=self.chunk,
            )

    def test_copied_fragment_is_rejected(self):
        payload = self.valid_chunk_payload()
        payload["claims"][0]["text"] = (
            "Revenue is money entering the business cost of revenue is subtracted from revenue the result is gross profit."
        )

        with self.assertRaisesRegex(OllamaError, "copies"):
            _validate_chunk_payload(
                payload,
                source=self.source,
                chunk=self.chunk,
            )

    def test_mixed_language_is_rejected(self):
        payload = self.valid_chunk_payload()
        payload["claims"][0]["text"] = "Gross profit is really重要的 for the lesson."

        with self.assertRaisesRegex(OllamaError, "CJK"):
            _validate_chunk_payload(
                payload,
                source=self.source,
                chunk=self.chunk,
            )

    def test_local_analysis_writes_symbolic_formula(self):
        client = FakeOllamaClient([
            self.valid_chunk_payload(),
            self.valid_narrative(),
        ])

        directory = analyze_video(
            video_id=VIDEO_ID,
            model="qwen2.5-math:7b",
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            manifest_path=self.manifest_path,
            chunk_token_budget=1000,
            client=client,
        )

        formulas = json.loads(
            (directory / "formulas.json").read_text(encoding="utf-8")
        )["formulas"]

        self.assertEqual(len(formulas), 1)
        self.assertEqual(
            formulas[0]["ascii"],
            "gross_profit = total_revenue - cost_of_revenue",
        )
        self.assertTrue(
            formulas[0]["derivation_steps"][0].startswith("Meaning:")
        )



    def test_chunk_schema_omits_unused_summary(self):
        from run_research_analysis import chunk_output_schema

        schema = chunk_output_schema(
            min_segment=0,
            max_segment=10,
        )

        self.assertNotIn(
            "chunk_summary",
            schema["properties"],
        )
        self.assertNotIn(
            "chunk_summary",
            schema["required"],
        )


    def test_support_window_localizes_broad_claim(self):
        from types import SimpleNamespace

        source = SimpleNamespace(
            segments=[
                {"text": "The introduction explains where the data appears."},
                {"text": "The speaker opens the income statement."},
                {"text": "Total revenue is the money entering the business."},
                {"text": "Cost of revenue is subtracted from total revenue."},
                {"text": "The result after subtraction is gross profit."},
                {"text": "The speaker then changes topics."},
                {"text": "The balance sheet contains assets and liabilities."},
                {"text": "Total equity is assets minus liabilities."},
                {"text": "The company has many shares outstanding."},
                {"text": "Book value divides equity by shares."},
            ]
        )

        start, end, _ = _select_support_window(
            source=source,
            start_segment=0,
            end_segment=9,
            support_text=(
                "Gross profit remains after cost of revenue is "
                "subtracted from total revenue."
            ),
        )

        self.assertLessEqual(end - start + 1, 6)
        self.assertGreaterEqual(start, 2)
        self.assertLessEqual(end, 4)

    def test_broad_claim_range_is_repaired_before_validation(self):
        from types import SimpleNamespace

        source = SimpleNamespace(
            segments=[
                {"text": "The introduction describes the website."},
                {"text": "The company example is introduced."},
                {"text": "The income statement is opened."},
                {"text": "Total revenue is money entering the business."},
                {"text": "Cost of revenue is removed from total revenue."},
                {"text": "Gross profit is the remaining amount."},
                {"text": "The lesson next discusses net income."},
                {"text": "The balance sheet is opened."},
                {"text": "Assets and liabilities are listed."},
                {"text": "Equity is assets minus liabilities."},
            ]
        )
        chunk = SimpleNamespace(
            chunk_index=0,
            start_segment=0,
            end_segment=9,
        )
        payload = {
            "claims": [
                {
                    "topic": "Gross profit",
                    "text": (
                        "Gross profit remains after cost of revenue "
                        "is deducted from total revenue."
                    ),
                    "explanation": (
                        "This relationship links incoming revenue "
                        "to the direct cost of producing it."
                    ),
                    "start_segment": 0,
                    "end_segment": 9,
                }
            ],
            "formulas": [],
            "caveats": [],
        }

        repairs = _repair_chunk_citations(
            payload,
            source=source,
            chunk=chunk,
        )

        repaired = payload["claims"][0]
        self.assertTrue(repairs)
        self.assertLessEqual(
            repaired["end_segment"] - repaired["start_segment"] + 1,
            6,
        )
        self.assertGreaterEqual(repaired["start_segment"], 3)
        self.assertLessEqual(repaired["end_segment"], 5)



    def test_safe_budget_reserves_output_space(self):
        self.assertEqual(
            safe_chunk_token_budget(1350, 4096),
            1024,
        )
        self.assertEqual(
            safe_chunk_token_budget(900, 4096),
            900,
        )

    def test_truncated_recovery_recursively_splits_chunk(self):
        left_payload = {
            "claims": [
                {
                    "topic": "Revenue and direct cost",
                    "text": "Direct revenue costs reduce the money entering the business.",
                    "explanation": "This relationship explains how the lesson moves from revenue toward gross profit.",
                    "start_segment": 0,
                    "end_segment": 1,
                }
            ],
            "formulas": [],
            "caveats": [],
        }
        right_payload = {
            "claims": [
                {
                    "topic": "Equity per share",
                    "text": "Book value per share distributes total equity across outstanding shares.",
                    "explanation": "This converts a whole-company balance-sheet amount into a per-share measure.",
                    "start_segment": 3,
                    "end_segment": 4,
                }
            ],
            "formulas": [],
            "caveats": [],
        }
        client = ScriptedOllamaClient(
            [
                "TRUNCATED",
                "TRUNCATED",
                left_payload,
                right_payload,
            ]
        )

        payload, response = extract_chunk_evidence(
            client=client,
            model="qwen2.5-math:7b",
            source=self.source,
            chunk=self.chunk,
            num_ctx=4096,
        )

        self.assertEqual(len(payload["claims"]), 2)
        self.assertEqual(response["request_count"], 4)
        self.assertEqual(response["split_count"], 1)
        self.assertEqual(len(client.calls), 4)


    def test_intermediate_sentence_mechanics_are_normalized(self):
        payload = self.valid_chunk_payload()
        payload["claims"][0]["text"] = (
            "gross profit remains after direct revenue costs are removed"
        )
        payload["claims"][0]["explanation"] = (
            "this connects revenue and direct costs"
        )
        payload["formulas"][0]["description"] = (
            "gross profit equals revenue less direct costs"
        )
        payload["formulas"][0]["derivation_steps"] = [
            "subtract cost of revenue from total revenue"
        ]
        payload["caveats"] = [
            "the transcript uses simplified terminology"
        ]

        repairs = _normalize_chunk_surface(
            payload,
            chunk_index=0,
        )

        self.assertTrue(repairs)
        self.assertEqual(
            payload["claims"][0]["text"],
            "Gross profit remains after direct revenue costs are removed.",
        )
        self.assertEqual(
            payload["formulas"][0]["derivation_steps"][0],
            "Subtract cost of revenue from total revenue.",
        )
        self.assertEqual(payload["caveats"], [])

        _validate_chunk_payload(
            payload,
            source=self.source,
            chunk=self.chunk,
        )

    def test_final_narrative_sentence_gate_remains_strict(self):
        payload = self.valid_narrative()
        payload["key_takeaways"][0]["text"] = (
            "gross profit is revenue less direct costs"
        )

        from run_research_analysis import Evidence

        evidence = [
            Evidence(
                evidence_id="E0001",
                text="Income statement relationships",
                start_segment=0,
                end_segment=2,
            ),
            Evidence(
                evidence_id="E0002",
                text="Per-share relationships",
                start_segment=3,
                end_segment=4,
            ),
            Evidence(
                evidence_id="E0003",
                text="Gross profit relationship",
                start_segment=0,
                end_segment=2,
            ),
        ]

        with self.assertRaisesRegex(
            OllamaError,
            "must begin as a complete sentence",
        ):
            _validate_narrative_quality(
                payload,
                source=self.source,
                evidence=evidence,
            )


    def test_cjk_claim_and_bad_caveat_are_dropped(self):
        payload = self.valid_chunk_payload()
        payload["claims"].append(
            {
                "topic": "Noise",
                "text": "This output contains重要的 foreign text.",
                "explanation": "It should not enter the evidence catalog.",
                "start_segment": 0,
                "end_segment": 0,
            }
        )
        payload["caveats"] = [
            "The output is not seasonally adjusted重要的."
        ]

        repairs = _normalize_chunk_surface(
            payload,
            chunk_index=0,
        )

        self.assertTrue(repairs)
        self.assertEqual(len(payload["claims"]), 2)
        self.assertEqual(payload["caveats"], [])

    def test_missing_formula_variable_definition_is_repaired(self):
        payload = self.valid_chunk_payload()
        payload["formulas"][0]["variables"] = [
            variable
            for variable in payload["formulas"][0]["variables"]
            if variable["symbol"] != "gross_profit"
        ]

        repairs = _repair_formula_variables(
            payload,
            chunk_index=0,
        )

        symbols = {
            variable["symbol"]
            for variable in payload["formulas"][0]["variables"]
        }
        self.assertIn("gross_profit", symbols)
        self.assertTrue(repairs)

    def test_bad_claim_is_pruned_without_losing_valid_formula(self):
        payload = self.valid_chunk_payload()
        payload["claims"] = [
            {
                "topic": "Copied",
                "text": (
                    "Revenue is money entering the business cost of "
                    "revenue is subtracted from revenue the result is "
                    "gross profit."
                ),
                "explanation": "This is copied too closely.",
                "start_segment": 0,
                "end_segment": 2,
            }
        ]

        repairs = _prune_chunk_payload(
            payload,
            source=self.source,
            chunk=self.chunk,
        )

        self.assertTrue(repairs)
        self.assertEqual(payload["claims"], [])
        self.assertEqual(len(payload["formulas"]), 1)

    def test_formula_evidence_reaches_synthesis_catalog(self):
        payload = self.valid_chunk_payload()
        payload["claims"] = []

        evidence, formulas, caveats = build_evidence_catalog(
            [payload]
        )

        self.assertEqual(len(evidence), 1)
        self.assertNotIn("Formula concept:", evidence[0].text)
        self.assertIn("Gross profit", evidence[0].text)
        self.assertEqual(len(formulas), 1)
        self.assertEqual(caveats, [])

    def test_markdown_explains_formula_and_variables(self):
        client = FakeOllamaClient([
            self.valid_chunk_payload(),
            self.valid_narrative(),
        ])

        directory = analyze_video(
            video_id=VIDEO_ID,
            model="qwen2.5-math:7b",
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            manifest_path=self.manifest_path,
            chunk_token_budget=1000,
            client=client,
        )

        markdown = (
            directory / "research.md"
        ).read_text(encoding="utf-8")

        self.assertIn("**Meaning:**", markdown)
        self.assertIn("**Machine-readable:**", markdown)
        self.assertIn("**Variables:**", markdown)
        self.assertIn("`gross_profit`", markdown)


    def test_formula_is_regrounded_to_entailed_transcript(self):
        payload = self.valid_chunk_payload()
        formula = payload["formulas"][0]
        formula["start_segment"] = 3
        formula["end_segment"] = 3

        start, end, _ = _locate_formula_support(
            source=self.source,
            formula=formula,
        )

        self.assertEqual((start, end), (0, 2))

    def test_unsupported_formula_has_no_support_window(self):
        formula = self.valid_chunk_payload()["formulas"][0]
        formula["ascii"] = (
            "market_capitalization = stock_price * shares_outstanding"
        )
        formula["variables"] = [
            {
                "symbol": "market_capitalization",
                "meaning": "Market capitalization",
                "unit": "currency",
            },
            {
                "symbol": "stock_price",
                "meaning": "Stock price",
                "unit": "currency per share",
            },
            {
                "symbol": "shares_outstanding",
                "meaning": "Shares outstanding",
                "unit": "shares",
            },
        ]

        with self.assertRaisesRegex(
            OllamaError,
            "No transcript window",
        ):
            _locate_formula_support(
                source=self.source,
                formula=formula,
            )

    def test_unsupported_year_is_rejected(self):
        with self.assertRaisesRegex(
            OllamaError,
            "unsupported numeric values",
        ):
            _validate_numeric_grounding(
                "Net income was 13 billion in 2018.",
                "Net income changed to 13 billion.",
                context="claim",
            )

    def test_duplicate_takeaways_are_rejected(self):
        payload = self.valid_narrative()
        payload["key_takeaways"][1]["text"] = (
            payload["key_takeaways"][0]["text"]
        )
        payload["key_takeaways"][1]["evidence_ids"] = ["E0001"]

        from run_research_analysis import Evidence

        evidence = [
            Evidence("E0001", "Revenue and gross profit.", 0, 2),
            Evidence("E0002", "Equity and book value per share.", 3, 4),
            Evidence("E0003", "Gross profit relationship.", 0, 2),
        ]

        with self.assertRaisesRegex(
            OllamaError,
            "duplicates another takeaway",
        ):
            _validate_narrative_quality(
                payload,
                source=self.source,
                evidence=evidence,
            )

    def test_internal_debug_language_is_rejected(self):
        payload = self.valid_narrative()
        payload["sections"][0]["heading"] = "Evidence catalog"

        from run_research_analysis import Evidence

        evidence = [
            Evidence("E0001", "Revenue and gross profit.", 0, 2),
            Evidence("E0002", "Equity and book value per share.", 3, 4),
            Evidence("E0003", "Gross profit relationship.", 0, 2),
        ]

        with self.assertRaises(OllamaError):
            _validate_narrative_quality(
                payload,
                source=self.source,
                evidence=evidence,
            )


    def test_formula_regrounding_may_cross_split_boundary(self):
        payload = self.valid_chunk_payload()
        payload["claims"] = []
        payload["formulas"][0]["start_segment"] = 1
        payload["formulas"][0]["end_segment"] = 1

        split_chunk = _chunk_from_range(
            self.source,
            chunk_index=0,
            start_segment=1,
            end_segment=4,
        )

        _repair_chunk_citations(
            payload,
            source=self.source,
            chunk=split_chunk,
        )
        _repair_formula_variables(
            payload,
            chunk_index=0,
        )
        _prune_chunk_payload(
            payload,
            source=self.source,
            chunk=split_chunk,
        )
        _validate_chunk_payload(
            payload,
            source=self.source,
            chunk=split_chunk,
        )

        self.assertEqual(
            (
                payload["formulas"][0]["start_segment"],
                payload["formulas"][0]["end_segment"],
            ),
            (0, 2),
        )

    def test_empty_compact_chunk_is_skipped_without_split(self):
        empty = {
            "claims": [],
            "formulas": [],
            "caveats": [],
        }
        client = ScriptedOllamaClient([empty, empty])

        payload, response = extract_chunk_evidence(
            client=client,
            model="qwen2.5-math:7b",
            source=self.source,
            chunk=self.chunk,
            num_ctx=4096,
        )

        self.assertEqual(payload, empty)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(response.get("split_count", 0), 0)

    def test_overlapping_variable_mentions_do_not_entail_formula(self):
        formula = self.valid_chunk_payload()["formulas"][0]
        formula["formula_id"] = "revenue"
        formula["name"] = "Revenue"
        formula["description"] = (
            "Revenue equals total revenue after cost of revenue."
        )
        formula["ascii"] = (
            "revenue = total_revenue - cost_of_revenue"
        )
        formula["latex"] = (
            r"\text{Revenue}=\text{Total Revenue}-"
            r"\text{Cost of Revenue}"
        )
        formula["variables"] = [
            {
                "symbol": "revenue",
                "meaning": "Revenue",
                "unit": "currency",
            },
            {
                "symbol": "total_revenue",
                "meaning": "Total revenue",
                "unit": "currency",
            },
            {
                "symbol": "cost_of_revenue",
                "meaning": "Cost of revenue",
                "unit": "currency",
            },
        ]

        with self.assertRaisesRegex(
            OllamaError,
            "No transcript window",
        ):
            _locate_formula_support(
                source=self.source,
                formula=formula,
            )


    def test_generic_alias_is_allowed_without_identifier_collision(self):
        formula = self.valid_chunk_payload()["formulas"][0]

        start, end, _ = _locate_formula_support(
            source=self.source,
            formula=formula,
        )

        self.assertEqual((start, end), (0, 2))


    def test_v4_routes_thinking_modes_by_stage(self):
        client = FakeOllamaClient([
            self.valid_chunk_payload(),
            self.valid_narrative(),
        ])

        directory = analyze_video(
            video_id=VIDEO_ID,
            extraction_model="qwen3:8b",
            narrative_model="qwen3:8b",
            extraction_think=True,
            narrative_think=False,
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            manifest_path=self.manifest_path,
            chunk_token_budget=1000,
            client=client,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["model"], "qwen3:8b")
        self.assertIs(client.calls[0]["think"], True)
        self.assertEqual(client.calls[1]["model"], "qwen3:8b")
        self.assertIs(client.calls[1]["think"], False)
        self.assertEqual(client.required_models, ["qwen3:8b"])

        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        details = metadata["analysis_details"]
        self.assertEqual(details["extraction"]["model"], "qwen3:8b")
        self.assertIs(details["extraction"]["think"], True)
        self.assertEqual(details["narrative"]["model"], "qwen3:8b")
        self.assertIs(details["narrative"]["think"], False)

    def test_v4_supports_separate_models(self):
        client = FakeOllamaClient([
            self.valid_chunk_payload(),
            self.valid_narrative(),
        ])

        directory = analyze_video(
            video_id=VIDEO_ID,
            extraction_model="extract-model",
            narrative_model="narrative-model",
            extraction_think=True,
            narrative_think=False,
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            manifest_path=self.manifest_path,
            chunk_token_budget=1000,
            client=client,
        )

        self.assertEqual(
            client.required_models,
            ["extract-model", "narrative-model"],
        )
        self.assertEqual(client.calls[0]["model"], "extract-model")
        self.assertEqual(client.calls[1]["model"], "narrative-model")

        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "extract=extract-model",
            metadata["analysis_backend"],
        )
        self.assertIn(
            "narrative=narrative-model",
            metadata["analysis_backend"],
        )

    def test_legacy_model_override_routes_both_stages(self):
        client = FakeOllamaClient([
            self.valid_chunk_payload(),
            self.valid_narrative(),
        ])

        analyze_video(
            video_id=VIDEO_ID,
            model="legacy-model",
            raw_root=self.raw_root,
            processed_root=self.processed_root,
            manifest_path=self.manifest_path,
            chunk_token_budget=1000,
            client=client,
        )

        self.assertEqual(client.required_models, ["legacy-model"])
        self.assertEqual(
            [call["model"] for call in client.calls],
            ["legacy-model", "legacy-model"],
        )

    def test_chat_request_sends_think_flag(self):
        class CapturingClient:
            def __init__(self):
                self.payload = None
                self.base_url = "http://127.0.0.1:11434"
                self.timeout_seconds = 1

            def _request(self, path, payload=None):
                self.payload = payload
                return {
                    "message": {
                        "role": "assistant",
                        "thinking": "private reasoning",
                        "content": '{"claims":[],"formulas":[],"caveats":[]}',
                    },
                    "done": True,
                    "done_reason": "stop",
                }

        from run_research_analysis import OllamaClient

        client = CapturingClient()
        OllamaClient.chat(
            client,
            model="qwen3:8b",
            system="system",
            user="user",
            schema={
                "type": "object",
                "properties": {},
            },
            num_ctx=8192,
            num_predict=100,
            think=True,
        )

        self.assertIs(client.payload["think"], True)


    def test_pe_ratio_requires_explicit_division_cue(self):
        from types import SimpleNamespace

        source = SimpleNamespace(
            segments=[
                {
                    "text": (
                        "The lesson repeats earnings per share and "
                        "introduces the price to earnings ratio."
                    )
                },
                {"text": "The stock price is sixty four dollars."},
                {
                    "text": (
                        "Take the price and divide it by EPS, which is "
                        "earnings per share."
                    )
                },
                {"text": "The resulting PE ratio is eighteen point four."},
            ]
        )
        formula = {
            "ascii": "pe_ratio = price / eps",
            "variables": [
                {
                    "symbol": "pe_ratio",
                    "meaning": "Price to earnings ratio",
                },
                {
                    "symbol": "price",
                    "meaning": "Stock price",
                },
                {
                    "symbol": "eps",
                    "meaning": "Earnings per share",
                },
            ],
        }

        start, end, _ = _locate_formula_support(
            source=source,
            formula=formula,
        )

        self.assertLessEqual(start, 2)
        self.assertGreaterEqual(end, 2)
        self.assertFalse((start, end) == (0, 0))

    def test_pe_ratio_rejects_per_without_division(self):
        from types import SimpleNamespace

        source = SimpleNamespace(
            segments=[
                {
                    "text": (
                        "Earnings per share is listed beside the "
                        "price to earnings ratio and the stock price."
                    )
                }
            ]
        )
        formula = {
            "ascii": "pe_ratio = price / eps",
            "variables": [
                {
                    "symbol": "pe_ratio",
                    "meaning": "Price to earnings ratio",
                },
                {
                    "symbol": "price",
                    "meaning": "Stock price",
                },
                {
                    "symbol": "eps",
                    "meaning": "Earnings per share",
                },
            ],
        }

        with self.assertRaisesRegex(
            OllamaError,
            "No transcript window",
        ):
            _locate_formula_support(
                source=source,
                formula=formula,
            )

    def test_per_share_formula_may_use_per_share_language(self):
        from types import SimpleNamespace

        source = SimpleNamespace(
            segments=[
                {
                    "text": (
                        "Net income is broken down to one share using "
                        "the shares outstanding, giving earnings per share."
                    )
                }
            ]
        )
        formula = {
            "ascii": (
                "earnings_per_share = net_income / shares_outstanding"
            ),
            "variables": [
                {
                    "symbol": "earnings_per_share",
                    "meaning": "Earnings per share",
                },
                {
                    "symbol": "net_income",
                    "meaning": "Net income",
                },
                {
                    "symbol": "shares_outstanding",
                    "meaning": "Shares outstanding",
                },
            ],
        }

        self.assertEqual(
            _locate_formula_support(
                source=source,
                formula=formula,
            )[:2],
            (0, 0),
        )

    def test_margin_of_safety_is_renamed_as_absolute_gap(self):
        payload = {
            "claims": [],
            "formulas": [
                {
                    "formula_id": "margin_of_safety",
                    "name": "Margin of safety",
                    "description": "Difference between price and book value.",
                    "ascii": (
                        "margin_of_safety = "
                        "market_price - book_value_per_share"
                    ),
                    "latex": (
                        r"\text{Margin of Safety}="
                        r"\text{Market Price}-\text{Book Value Per Share}"
                    ),
                    "variables": [
                        {
                            "symbol": "margin_of_safety",
                            "meaning": "Margin of safety",
                            "unit": "currency per share",
                        },
                        {
                            "symbol": "market_price",
                            "meaning": "Market price",
                            "unit": "currency per share",
                        },
                        {
                            "symbol": "book_value_per_share",
                            "meaning": "Book value per share",
                            "unit": "currency per share",
                        },
                    ],
                    "derivation_steps": [
                        "Subtract book value per share from market price."
                    ],
                    "start_segment": 0,
                    "end_segment": 0,
                }
            ],
            "caveats": [],
        }

        repairs = _normalize_formula_semantics(
            payload,
            chunk_index=0,
        )
        formula = payload["formulas"][0]

        self.assertTrue(repairs)
        self.assertEqual(
            formula["formula_id"],
            "margin_of_safety_gap",
        )
        self.assertEqual(
            formula["ascii"],
            (
                "margin_of_safety_gap = "
                "market_price - book_value_per_share"
            ),
        )
        self.assertEqual(
            formula["variables"][0]["symbol"],
            "margin_of_safety_gap",
        )


    def test_direct_formula_validation_passes_left_symbol(self):
        payload = self.valid_chunk_payload()
        formula = payload["formulas"][0]

        _validate_formula_candidate(
            formula,
            source=self.source,
            context="direct formula validation",
            minimum_segment=0,
            maximum_segment=len(self.source.segments) - 1,
        )

if __name__ == "__main__":
    unittest.main()
