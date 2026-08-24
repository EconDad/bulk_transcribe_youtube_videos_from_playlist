from __future__ import annotations

import json
import unittest

from research_v43.calculation_inventory import (
    CalculationItem,
    InventoryValidationError,
)
from research_v43.entailment import EntailmentValidationError
from research_v43.expression_ast import FormulaCandidate
from research_v43.finalization import NarrativeEvidence
from research_v43.gate3_audit_recovery import (
    parse_inventory_evidence_audit_response_with_gate3_repair,
)
from research_v43.gate3_recovery import (
    build_synthesis_retry_prompt,
    parse_inventory_response_with_gate3_repair,
    recover_synthesis_with_gate3_sentence_count,
    sentence_count_reader_prose,
    validate_entailment_response_with_gate3_structure_repair,
)
from research_v43.inventory_evidence_audit import AuditAction


def inventory_item(
    *,
    operations=None,
    variables=None,
    start=0,
    end=0,
):
    return {
        "calculation_id": "CALC_0001",
        "name": "Synthetic calculation",
        "source_mode": "spoken",
        "start_segment": start,
        "end_segment": end,
        "variables_mentioned": list(variables or ["first", "second"]),
        "operations_mentioned": list(operations or ["division"]),
        "visual_equation_cue": False,
        "formula_expected": True,
        "reason": "Synthetic source-grounded calculation.",
    }


def parse_item(**kwargs):
    return CalculationItem.from_mapping(inventory_item(**kwargs))


class Gate3InventoryRecoveryTests(unittest.TestCase):
    def test_inventory_drops_blank_string_members_and_canonicalizes_aliases(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                inventory_item(
                    variables=["first", "", "   ", "second"],
                    operations=["subtract"],
                )
            ],
        }
        parsed = parse_inventory_response_with_gate3_repair(
            json.dumps(payload),
            expected_video_id="video-123",
            maximum_segment=4,
        )
        item = parsed.calculations[0]
        self.assertEqual(item.variables_mentioned, ("first", "second"))
        self.assertEqual(item.operations_mentioned, ("subtraction",))

    def test_inventory_keeps_non_string_invalid_member_for_strict_rejection(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [
                inventory_item(
                    variables=["first", None],
                )
            ],
        }
        with self.assertRaises(InventoryValidationError):
            parse_inventory_response_with_gate3_repair(
                json.dumps(payload),
                expected_video_id="video-123",
                maximum_segment=4,
            )

    def test_sum_alias_is_canonicalized_without_formula_injection(self):
        payload = {
            "schema_version": "1.0",
            "video_id": "video-123",
            "calculations": [inventory_item(operations=["sum"])],
        }
        parsed = parse_inventory_response_with_gate3_repair(
            json.dumps(payload),
            expected_video_id="video-123",
            maximum_segment=4,
        )
        self.assertEqual(
            parsed.calculations[0].operations_mentioned,
            ("addition",),
        )


class Gate3AuditRecoveryTests(unittest.TestCase):
    def test_off_the_top_is_generic_subtraction_cue(self):
        item = parse_item(
            operations=["subtraction"],
            variables=["$100", "$20"],
            start=0,
            end=1,
        )
        segments = [
            {"text": "$20 of the $100 went to the employee."},
            {"text": "We had $20 go straight out of the $100 right off the top."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "reason": "The cost comes off the top of the starting amount.",
            "revised_variables_mentioned": ["$100", "$20"],
            "revised_operations_mentioned": ["subtraction"],
        }
        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=1,
        )
        self.assertEqual(decision.action, AuditAction.RECONCILE)

    def test_spoken_number_words_ground_numeric_variable_phrases(self):
        item = parse_item(
            operations=["subtraction"],
            variables=["$100", "$70"],
            start=0,
            end=1,
        )
        segments = [
            {"text": "The revenue was a hundred, but the cost was $70 combined."},
            {"text": "We took the hundred and subtract the 70."},
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0, 1],
            "reason": "The source subtracts cost from revenue.",
            "revised_variables_mentioned": [
                "revenue of $100",
                "cost of $70",
            ],
            "revised_operations_mentioned": ["subtraction"],
        }
        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=1,
        )
        self.assertEqual(decision.action, AuditAction.RECONCILE)

    def test_percent_of_is_generic_multiplication_cue(self):
        item = parse_item(
            operations=["percentage calculation"],
            variables=["equity", "market price"],
        )
        segments = [
            {"text": "The equity is three and a half percent of the market price."}
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0],
            "reason": "The source states a percent-of relationship.",
            "revised_variables_mentioned": ["equity", "market price"],
            "revised_operations_mentioned": ["multiplication"],
        }
        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=0,
        )
        self.assertEqual(decision.action, AuditAction.RECONCILE)
        self.assertEqual(decision.revised_operations_mentioned, ("multiplication",))

    def test_missing_operation_in_full_neighborhood_downgrades_non_symbolic(self):
        item = parse_item(
            operations=["division"],
            variables=["net income", "price"],
        )
        segments = [
            {
                "text": (
                    "The business has $20,000 in net income and a $200,000 "
                    "asking price, giving a 10 percent return."
                )
            }
        ]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0],
            "reason": "The model inferred a division relationship.",
            "revised_variables_mentioned": ["net income", "price"],
            "revised_operations_mentioned": ["division"],
        }
        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=0,
        )
        self.assertEqual(decision.action, AuditAction.DOWNGRADE_NON_SYMBOLIC)

    def test_unknown_assignment_operation_downgrades_non_symbolic(self):
        item = parse_item(
            operations=["assignment"],
            variables=["$100"],
        )
        segments = [{"text": "$100 would be the total revenue."}]
        payload = {
            "calculation_id": "CALC_0001",
            "action": "reconcile",
            "evidence_segment_ids": [0],
            "reason": "The value is assigned a label.",
            "revised_variables_mentioned": ["$100"],
            "revised_operations_mentioned": ["assignment"],
        }
        decision = parse_inventory_evidence_audit_response_with_gate3_repair(
            json.dumps(payload),
            item=item,
            segments=segments,
            neighborhood_start=0,
            neighborhood_end=0,
        )
        self.assertEqual(decision.action, AuditAction.DOWNGRADE_NON_SYMBOLIC)


class Gate3EntailmentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.item = parse_item(
            operations=["subtraction"],
            variables=["total assets", "total liabilities", "total equity"],
        )
        self.candidate = FormulaCandidate.from_mapping(
            {
                "calculation_id": "CALC_0001",
                "formula_id": "equity_difference",
                "name": "Synthetic difference",
                "ascii": "total_equity = total_assets - total_liabilities",
                "latex": "e=a-l",
                "derivation_type": "stated",
                "variables": [
                    {"symbol": "total_equity", "meaning": "result", "unit": ""},
                    {"symbol": "total_assets", "meaning": "first total", "unit": ""},
                    {"symbol": "total_liabilities", "meaning": "second total", "unit": ""},
                ],
                "derivation_steps": ["Apply the stated subtraction."],
                "source_claims": [
                    {"start_segment": 0, "end_segment": 0, "relationship": "subtraction"}
                ],
            }
        )
        node = self.candidate.parsed.operations[0]
        self.payload = {
            "calculation_id": "CALC_0001",
            "formula_id": "equity_difference",
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
                            "quote": "total assets minus total liabilities",
                        }
                    ],
                    "identifier_groundings": [
                        {"identifier": "total_assets", "start_segment": 0, "end_segment": 0, "quote": "total assets"},
                        {"identifier": "total_liabilities", "start_segment": 0, "end_segment": 0, "quote": "total liabilities"},
                        {"identifier": "total_equity", "start_segment": 0, "end,segment": 0, "quote": "total equity"},
                    ],
                    "depends_on_node_ids": [],
                    "derivation_step": "",
                }
            ],
        }
        self.segments = [
            {
                "text": (
                    "The total equity is the total assets minus total liabilities."
                )
            }
        ]

    def test_punctuation_only_grounding_key_typo_is_repaired(self):
        report = validate_entailment_response_with_gate3_structure_repair(
            self.payload,
            item=self.item,
            candidate=self.candidate,
            segments=self.segments,
        )
        self.assertTrue(report.passed, report.issues)

    def test_unknown_extra_grounding_field_is_not_dropped(self):
        payload = json.loads(json.dumps(self.payload))
        payload["nodes"][0]["identifier_groundings"][2]["extra"] = "no"
        with self.assertRaises(EntailmentValidationError):
            validate_entailment_response_with_gate3_structure_repair(
                payload,
                item=self.item,
                candidate=self.candidate,
                segments=self.segments,
            )


class Gate3SynthesisRecoveryTests(unittest.TestCase):
    def test_sentence_counter_does_not_split_internal_initialism_periods(self):
        text = (
            "The U.S. central bank has a defined role. "
            "Policy tools affect financial conditions. "
            "The source describes institutional responsibilities. "
            "The lesson closes with a broader summary."
        )
        self.assertEqual(sentence_count_reader_prose(text), 4)

    def test_gate3_synthesis_accepts_four_sentences_with_initialism(self):
        segments = [
            {"text": "The U.S. central bank has a defined role."},
            {"text": "Policy tools affect financial conditions."},
            {"text": "The source describes institutional responsibilities."},
            {"text": "The lesson closes with a broader summary."},
        ]
        evidence = tuple(
            NarrativeEvidence(
                f"N{index + 1:04d}",
                f"Topic {index + 1}",
                segment["text"],
                segment["text"],
                index,
                index,
            )
            for index, segment in enumerate(segments)
        )
        payload = {
            "executive_summary": " ".join(segment["text"] for segment in segments),
            "executive_summary_evidence_ids": [item.evidence_id for item in evidence],
            "key_takeaways": [
                {"text": evidence[index].text, "evidence_ids": [evidence[index].evidence_id]}
                for index in range(4)
            ],
            "sections": [
                {"heading": "Opening", "summary": evidence[0].text, "evidence_ids": [evidence[0].evidence_id]},
                {"heading": "Middle", "summary": evidence[1].text, "evidence_ids": [evidence[1].evidence_id]},
                {"heading": "Conclusion", "summary": evidence[3].text, "evidence_ids": [evidence[3].evidence_id]},
            ],
        }
        result = recover_synthesis_with_gate3_sentence_count(
            payload,
            evidence=evidence,
            segments=segments,
        )
        self.assertEqual(len(result["key_takeaways"]), 4)

    def test_retry_prompt_contains_validation_error_and_four_sentence_rule(self):
        prompt = build_synthesis_retry_prompt(
            "ORIGINAL",
            "executive_summary must contain exactly four sentences",
        )
        self.assertIn("ORIGINAL", prompt)
        self.assertIn("VALIDATION ERROR", prompt)
        self.assertIn("exactly four", prompt)
        self.assertIn("executive_summary must contain exactly four sentences", prompt)


if __name__ == "__main__":
    unittest.main()
