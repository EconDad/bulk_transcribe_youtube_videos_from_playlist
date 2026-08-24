"""Domain-neutral recoveries exposed by the v4.3 Gate 3 pilot.

The Gate 3 pilot surfaced serialization and grounding failures that should not
require subject-specific formula knowledge.  This module keeps the frozen Gate
2 behavior intact while adding only bounded, mechanically verifiable repairs:

* drop blank strings from inventory string arrays while leaving other invalid
  values for the strict parser to reject;
* canonicalize common arithmetic verb/noun aliases in inventory operations;
* accept a few generic linguistic arithmetic cues that the frozen lexical
  validator does not recognize (for example, ``off the top`` subtraction and
  ``percent of`` multiplication);
* ground digit-valued inventory variables against equivalent spoken English
  number words without changing the variable itself;
* downgrade an audit item only when its own bounded neighborhood cannot ground
  the operation the audit claims;
* normalize punctuation-only key typos in entailment response objects before
  rerunning the unchanged strict Gate 2 validator; and
* make Stage G sentence counting resilient to internal periods in initialisms
  while preserving strict four-sentence validation.

None of these helpers inject formulas, infer domain relationships from titles,
or weaken the downstream expression/entailment validators.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping, Sequence

import research_v43.finalization as finalization_core

from .calculation_inventory import CalculationInventory, CalculationItem
from .entailment import EntailmentValidationError, _has_operation_cue
from .gate2_recovery import (
    validate_entailment_response_with_gate2_quote_repair,
)
from .inventory_evidence_audit import (
    AuditAction,
    InventoryAuditDecision,
    InventoryEvidenceAuditError,
    _numeric_signatures,
    _range_text,
    _semantic_word_tokens,
    _singularize_token,
    parse_inventory_evidence_audit_response,
)
from .inventory_recovery import parse_inventory_response_with_order_repair
from .synthesis_recovery import recover_synthesis


_OPERATION_ALIASES = {
    "add": "addition",
    "added": "addition",
    "adding": "addition",
    "addition": "addition",
    "sum": "addition",
    "summation": "addition",
    "subtract": "subtraction",
    "subtracted": "subtraction",
    "subtracting": "subtraction",
    "subtraction": "subtraction",
    "minus": "subtraction",
    "multiply": "multiplication",
    "multiplied": "multiplication",
    "multiplying": "multiplication",
    "multiplication": "multiplication",
    "divide": "division",
    "divided": "division",
    "dividing": "division",
    "division": "division",
    "exponent": "exponentiation",
    "exponentiation": "exponentiation",
    "power": "exponentiation",
}

_EXTRA_OPERATION_CUES = {
    "subtraction": (
        "off the top",
    ),
    "multiplication": (
        "percent of",
        "percentage of",
    ),
}

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)

_INITIALISM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+(?=\s|$)")


def _canonical_operation(value: str) -> str:
    normalized = re.sub(r"[\s-]+", " ", value.strip().casefold())
    return _OPERATION_ALIASES.get(normalized, value.strip())


def normalize_gate3_inventory_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Repair only blank string members and generic operation aliases."""

    normalized = copy.deepcopy(dict(payload))
    calculations = normalized.get("calculations")
    if not isinstance(calculations, list):
        return normalized

    for item in calculations:
        if not isinstance(item, dict):
            continue
        for field in ("variables_mentioned", "operations_mentioned"):
            values = item.get(field)
            if not isinstance(values, list):
                continue
            filtered: list[Any] = []
            for value in values:
                if isinstance(value, str) and not value.strip():
                    continue
                filtered.append(value)
            if field == "operations_mentioned":
                filtered = [
                    _canonical_operation(value) if isinstance(value, str) else value
                    for value in filtered
                ]
            item[field] = filtered
    return normalized


def parse_inventory_response_with_gate3_repair(
    response_text: str,
    *,
    expected_video_id: str,
    maximum_segment: int,
) -> CalculationInventory:
    """Apply mechanical Gate 3 inventory normalization, then strict parsing."""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return parse_inventory_response_with_order_repair(
            response_text,
            expected_video_id=expected_video_id,
            maximum_segment=maximum_segment,
        )
    if not isinstance(payload, Mapping):
        return parse_inventory_response_with_order_repair(
            response_text,
            expected_video_id=expected_video_id,
            maximum_segment=maximum_segment,
        )

    normalized = normalize_gate3_inventory_payload(payload)
    return parse_inventory_response_with_order_repair(
        json.dumps(normalized),
        expected_video_id=expected_video_id,
        maximum_segment=maximum_segment,
    )


def _below_thousand(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.extend((_ONES[hundreds], "hundred"))
    if remainder:
        if remainder < 20:
            parts.append(_ONES[remainder])
        else:
            tens, ones = divmod(remainder, 10)
            parts.append(_TENS[tens])
            if ones:
                parts.append(_ONES[ones])
    return " ".join(parts)


def _integer_words(value: int) -> str:
    if value == 0:
        return "zero"
    if value < 0:
        return "minus " + _integer_words(-value)

    remainder = value
    parts: list[str] = []
    for scale, name in _SCALES:
        quotient, remainder = divmod(remainder, scale)
        if quotient:
            parts.extend((_integer_words(quotient), name))
    if remainder:
        parts.append(_below_thousand(remainder))
    return " ".join(part for part in parts if part)


def _numeric_signature_supported(
    signature: tuple[str, bool],
    source_text: str,
) -> bool:
    if signature in _numeric_signatures(source_text):
        return True

    raw_number, is_percent = signature
    try:
        numeric = float(raw_number)
    except ValueError:
        return False
    if not numeric.is_integer():
        return False

    integer = int(numeric)
    if abs(integer) > 999_999_999_999_999:
        return False

    words = _integer_words(integer)
    candidates = {words}
    if words.startswith("one hundred"):
        candidates.add(words.removeprefix("one "))
    if words.startswith("one thousand"):
        candidates.add(words.removeprefix("one "))

    normalized_source = re.sub(r"\s+", " ", source_text.casefold())
    for candidate in candidates:
        escaped = re.escape(candidate).replace(r"\ ", r"\s+")
        if is_percent:
            pattern = rf"(?<![a-z]){escaped}\s+(?:percent|percentage)(?![a-z])"
        else:
            pattern = rf"(?<![a-z]){escaped}(?![a-z])"
        if re.search(pattern, normalized_source):
            return True
    return False


def _gate3_variable_appears(variable: str, source_text: str) -> bool:
    from .inventory_evidence_audit import _variable_appears

    if _variable_appears(variable, source_text):
        return True

    signatures = _numeric_signatures(variable)
    if signatures and not all(
        _numeric_signature_supported(signature, source_text)
        for signature in signatures
    ):
        return False

    semantic_tokens = {
        _singularize_token(token)
        for token in _semantic_word_tokens(variable)
        if token not in _STOP_WORDS
    }
    source_tokens = {
        _singularize_token(token)
        for token in _semantic_word_tokens(source_text)
        if token not in _STOP_WORDS
    }
    if semantic_tokens and not semantic_tokens.issubset(source_tokens):
        return False

    return bool(signatures or semantic_tokens)


def _gate3_has_operation_cue(operation: str, source_text: str) -> bool:
    if _has_operation_cue(operation, source_text):
        return True
    normalized = re.sub(r"\s+", " ", source_text.casefold())
    for cue in _EXTRA_OPERATION_CUES.get(operation, ()):
        escaped = re.escape(cue).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", normalized):
            return True
    return False


def _normalized_audit_payload(response_text: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, Mapping):
        return None
    normalized = copy.deepcopy(dict(raw))
    for field in (
        "revised_variables_mentioned",
        "revised_operations_mentioned",
    ):
        values = normalized.get(field)
        if not isinstance(values, list):
            continue
        filtered: list[Any] = []
        for value in values:
            if isinstance(value, str) and not value.strip():
                continue
            filtered.append(value)
        if field == "revised_operations_mentioned":
            filtered = [
                _canonical_operation(value) if isinstance(value, str) else value
                for value in filtered
            ]
        normalized[field] = filtered
    return normalized


def _valid_evidence_ids(
    raw: Mapping[str, Any],
    *,
    neighborhood_start: int,
    neighborhood_end: int,
) -> tuple[int, ...] | None:
    values = raw.get("evidence_segment_ids")
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        return None
    result: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < neighborhood_start
            or value > neighborhood_end
        ):
            return None
        if value not in result:
            result.append(value)
    return tuple(sorted(result))


def parse_inventory_evidence_audit_response_with_gate3_repair(
    response_text: str,
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> InventoryAuditDecision:
    """Resolve only mechanically grounded Gate 3 audit failures.

    A failed reconcile is accepted only when every revised variable and revised
    operation is grounded inside the model-selected contiguous span under the
    extended generic lexical rules.  Otherwise the item is downgraded only when
    the claimed operation is absent from the *entire* bounded neighborhood.
    """

    try:
        return parse_inventory_evidence_audit_response(
            response_text,
            item=item,
            segments=segments,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
    except InventoryEvidenceAuditError as original_error:
        pass

    normalized = _normalized_audit_payload(response_text)
    if normalized is None:
        raise original_error

    normalized_text = json.dumps(normalized)
    try:
        return parse_inventory_evidence_audit_response(
            normalized_text,
            item=item,
            segments=segments,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
    except InventoryEvidenceAuditError:
        pass

    action = str(normalized.get("action") or "").strip().casefold().replace("-", "_")
    if action != "reconcile":
        raise original_error

    evidence_ids = _valid_evidence_ids(
        normalized,
        neighborhood_start=neighborhood_start,
        neighborhood_end=neighborhood_end,
    )
    if evidence_ids is None:
        raise original_error

    raw_variables = normalized.get("revised_variables_mentioned")
    raw_operations = normalized.get("revised_operations_mentioned")
    if (
        isinstance(raw_variables, (str, bytes))
        or not isinstance(raw_variables, Sequence)
        or isinstance(raw_operations, (str, bytes))
        or not isinstance(raw_operations, Sequence)
    ):
        raise original_error
    if not raw_variables or not raw_operations:
        raise original_error
    if not all(isinstance(value, str) and value.strip() for value in raw_variables):
        raise original_error
    if not all(isinstance(value, str) and value.strip() for value in raw_operations):
        raise original_error

    variables = tuple(str(value).strip() for value in raw_variables)
    operations = tuple(_canonical_operation(str(value)) for value in raw_operations)
    selected_start = min(item.start_segment, *evidence_ids)
    selected_end = max(item.end_segment, *evidence_ids)
    selected_text = _range_text(segments, selected_start, selected_end)

    variables_grounded = all(
        _gate3_variable_appears(variable, selected_text)
        for variable in variables
    )
    operations_grounded = all(
        _gate3_has_operation_cue(operation, selected_text)
        for operation in operations
    )

    reason = str(normalized.get("reason") or "").strip()
    if variables_grounded and operations_grounded:
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.RECONCILE,
            evidence_segment_ids=evidence_ids,
            reason=(
                reason
                or "Gate 3 bounded lexical recovery grounded the revised claims."
            ),
            revised_variables_mentioned=variables,
            revised_operations_mentioned=operations,
        )

    neighborhood_text = _range_text(
        segments,
        neighborhood_start,
        neighborhood_end,
    )
    operation_supported_anywhere = any(
        _gate3_has_operation_cue(operation, neighborhood_text)
        for operation in operations
    )
    if not operation_supported_anywhere:
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.DOWNGRADE_NON_SYMBOLIC,
            evidence_segment_ids=evidence_ids,
            reason=(
                "Gate 3 bounded recovery found no source cue for the claimed "
                "arithmetic operation anywhere in the audit neighborhood; "
                "the event is retained as non-symbolic rather than inventing "
                "an operation."
            ),
        )

    raise original_error


def _normalize_schema_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"_+", "_", normalized)


def _normalize_mapping_keys_exact(
    raw: Any,
    *,
    expected: set[str],
) -> tuple[Any, bool]:
    if not isinstance(raw, Mapping):
        return raw, False
    normalized_pairs: list[tuple[str, Any]] = []
    for key, value in raw.items():
        if not isinstance(key, str):
            return raw, False
        normalized_pairs.append((_normalize_schema_key(key), value))
    keys = [key for key, _ in normalized_pairs]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        return raw, False
    repaired = {key: value for key, value in normalized_pairs}
    return repaired, repaired != dict(raw)


def normalize_entailment_schema_keys(payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize punctuation-only schema-key typos without dropping fields."""

    repaired = copy.deepcopy(dict(payload))
    changed = False
    nodes = repaired.get("nodes")
    if not isinstance(nodes, list):
        return repaired, False

    node_fields = {
        "node_id",
        "expression",
        "operation",
        "status",
        "evidence",
        "identifier_groundings",
        "depends_on_node_ids",
        "derivation_step",
    }
    evidence_fields = {"start_segment", "end_segment", "quote"}
    grounding_fields = {"identifier", "start_segment", "end_segment", "quote"}

    normalized_nodes: list[Any] = []
    for raw_node in nodes:
        node, node_changed = _normalize_mapping_keys_exact(
            raw_node,
            expected=node_fields,
        )
        if not isinstance(node, Mapping):
            normalized_nodes.append(raw_node)
            continue
        node = dict(node)
        changed = changed or node_changed

        for field, expected in (
            ("evidence", evidence_fields),
            ("identifier_groundings", grounding_fields),
        ):
            records = node.get(field)
            if not isinstance(records, list):
                continue
            normalized_records: list[Any] = []
            for record in records:
                normalized_record, record_changed = _normalize_mapping_keys_exact(
                    record,
                    expected=expected,
                )
                normalized_records.append(normalized_record)
                changed = changed or record_changed
            node[field] = normalized_records
        normalized_nodes.append(node)

    repaired["nodes"] = normalized_nodes
    return repaired, changed


def validate_entailment_response_with_gate3_structure_repair(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
    candidate: Any,
    segments: Sequence[Mapping[str, Any]],
):
    """Repair punctuation-only schema keys, then reuse strict Gate 2 validation."""

    try:
        return validate_entailment_response_with_gate2_quote_repair(
            payload,
            item=item,
            candidate=candidate,
            segments=segments,
        )
    except EntailmentValidationError as original_error:
        repaired, changed = normalize_entailment_schema_keys(payload)
        if not changed:
            raise
        try:
            return validate_entailment_response_with_gate2_quote_repair(
                repaired,
                item=item,
                candidate=candidate,
                segments=segments,
            )
        except EntailmentValidationError:
            raise original_error


def sentence_count_reader_prose(text: str) -> int:
    """Count sentence punctuation without splitting internal initialism periods."""

    value = str(text).strip()
    if not value:
        return 0

    placeholder = "\ue000"

    def protect(match: re.Match[str]) -> str:
        token = match.group(0)
        # If the initialism ends the entire summary, preserve its final period as
        # the sentence terminator while protecting only the internal periods.
        keep_final = match.end() == len(value)
        pieces = list(token)
        period_indexes = [index for index, char in enumerate(pieces) if char == "."]
        for index in period_indexes[:-1] if keep_final else period_indexes:
            pieces[index] = placeholder
        return "".join(pieces)

    protected = _INITIALISM_RE.sub(protect, value)
    return len(_SENTENCE_RE.findall(protected))


def recover_synthesis_with_gate3_sentence_count(
    payload: Mapping[str, Any],
    *,
    evidence: Sequence[Any],
    segments: Sequence[Mapping[str, Any]],
    on_repair=None,
):
    """Run frozen synthesis recovery with the Gate 3 sentence counter."""

    original = finalization_core._sentence_count
    finalization_core._sentence_count = sentence_count_reader_prose
    try:
        return recover_synthesis(
            payload,
            evidence=evidence,
            segments=segments,
            on_repair=on_repair,
        )
    finally:
        finalization_core._sentence_count = original


def build_synthesis_retry_prompt(
    original_prompt: str,
    validation_error: Exception | str,
) -> str:
    """Give subsequent synthesis attempts the deterministic failure reason."""

    return (
        original_prompt
        + "\n\nREPAIR REQUEST:\n"
        + "The previous synthesis response failed deterministic validation. "
        + "Return a complete replacement object using the same validated "
        + "evidence only. Do not add outside facts or new evidence IDs. Correct "
        + "the listed validation defect while preserving all other schema "
        + "requirements. The executive_summary must contain exactly four "
        + "reader-facing sentences. Avoid treating periods inside abbreviations "
        + "or initialisms as sentence boundaries.\n\n"
        + "VALIDATION ERROR:\n"
        + str(validation_error)
    )
