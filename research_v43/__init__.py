"""Generalized research-pipeline primitives for v4.3."""

from .artifacts import (
    ArtifactWriteError,
    ArtifactWriteResult,
    verify_diagnostic_package,
    write_diagnostic_package,
)
from .calculation_inventory import (
    CalculationInventory,
    CalculationItem,
    InventoryValidationError,
    SourceMode,
    build_inventory_prompt,
    parse_inventory_response,
)
from .coverage import (
    CoverageReport,
    CoverageResolution,
    CoverageState,
    CoverageValidationError,
    reconcile_coverage,
)
from .entailment import (
    EntailmentValidationError,
    FormulaEntailmentReport,
    NodeEntailment,
    NodeStatus,
    build_entailment_prompt,
    validate_entailment_response,
)
from .expression_ast import (
    DerivationType,
    ExpressionNode,
    ExpressionValidationError,
    FormulaCandidate,
    ParsedFormula,
    parse_formula,
)
from .formula_extraction import (
    ExtractionDisposition,
    FormulaExtractionError,
    FormulaExtractionResult,
    build_formula_extraction_prompt,
    parse_formula_extraction_response,
)
from .model_client import (
    JsonModelResponse,
    ModelClientError,
    ModelInvocation,
    OllamaJsonClient,
)

__all__ = [
    "ArtifactWriteError",
    "ArtifactWriteResult",
    "CalculationInventory",
    "CalculationItem",
    "CoverageReport",
    "CoverageResolution",
    "CoverageState",
    "CoverageValidationError",
    "DerivationType",
    "EntailmentValidationError",
    "ExpressionNode",
    "ExpressionValidationError",
    "ExtractionDisposition",
    "FormulaCandidate",
    "FormulaEntailmentReport",
    "FormulaExtractionError",
    "FormulaExtractionResult",
    "InventoryValidationError",
    "JsonModelResponse",
    "ModelClientError",
    "ModelInvocation",
    "NodeEntailment",
    "NodeStatus",
    "OllamaJsonClient",
    "ParsedFormula",
    "SourceMode",
    "build_entailment_prompt",
    "build_formula_extraction_prompt",
    "build_inventory_prompt",
    "parse_formula",
    "parse_formula_extraction_response",
    "parse_inventory_response",
    "reconcile_coverage",
    "validate_entailment_response",
    "verify_diagnostic_package",
    "write_diagnostic_package",
]
