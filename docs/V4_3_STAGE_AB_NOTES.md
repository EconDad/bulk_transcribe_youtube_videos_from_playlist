# v4.3 Stage A–B Foundation

This milestone adds domain-neutral primitives only:

- Strict calculation-inventory schema and JSON parser
- Domain-neutral inventory prompt construction
- One safe AST parser for all formula source types
- Multi-operator and nested-expression support
- Whitelisted mathematical functions
- Formula-candidate variable-definition validation
- Calculation-to-formula coverage reconciliation

It does not modify the working v4.1.1 runner.

It intentionally contains no subject-specific formula recovery rules or
subject-specific alias tables.

Next milestone:

1. Qwen3 calculation-inventory invocation
2. Formula-candidate extraction per inventory item
3. Expression-node entailment records
4. Persisted inventory, entailment, and coverage artifacts
