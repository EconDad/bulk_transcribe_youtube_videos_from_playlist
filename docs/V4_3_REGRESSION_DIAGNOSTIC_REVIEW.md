# Research v4.3 Regression Diagnostic Review

Review date: 2026-08-28
Implementation verified: 2026-08-29
Scope: `csy91a1iDKU` and `cWzgk-8QHKk`
Disposition: accepted generic repairs implemented and replayed

## Post-review implementation outcome

The accepted repairs were implemented without weakening expression parsing,
entailment, or formula-coverage validation. The full suite passes with
`225 passed, 8 subtests passed`.

- `csy91a1iDKU` now has complete diagnostic formula coverage and finalized as
  `research_ready`. Its verified package SHA-256 is
  `be1e7ad544ca307e6c740857c1ce3c68035588730b7d22df18dcf476672e6d22`.
- `cWzgk-8QHKk` now correctly downgrades `CALC_0009` and `CALC_0010` to
  non-symbolic events. It remains `analysis_failed` only for `CALC_0005`, whose
  source does not support a unit-consistent reusable relationship.

The numeric-leading identifier repair is limited to mechanically prefixing an
identifier-shaped token such as `123_units` with `value_`. It does not change
operators, constants, source claims, meanings, or units, and bare numeric
literals remain untouched.

## Executive conclusion

The two fail-closed packages contain six unresolved calculation events.

- Three are recoverable, domain-neutral model-shape or evidence-localization
  defects with strong transcript support.
- Three should remain non-formula outcomes because the transcript is
  conceptually loose, arithmetically inconsistent, or does not state enough
  information to normalize a reusable relationship.

The expression grammar, formula entailment standard, and package fail-closed
rule should not be weakened.

## `csy91a1iDKU` — What is a Share

### `CALC_0001` — recoverable inventory vocabulary defect

The transcript explicitly states the full calculation:

- S28: the business is valued at `$100,000`;
- S29-S31: it is divided into `10,000` pieces;
- S33: one piece equals `$10`;
- S35-S36: `$100,000` divided by `10,000` is `$10 per share`.

The inventory item originally retained both semantic aliases
(`total_value`, `number_of_shares`, `price_per_share`) and concrete source
values. The audit repair discarded the concrete values and returned only the
semantic aliases, which cannot pass exact lexical grounding because those
names do not occur in the transcript.

Classification: **recoverable model-shape defect**.

Recommended generic repair: when an audit reconciliation replaces variables,
prefer source-extractive phrases or grounded numeric literals from the original
item. Do not accept invented snake-case aliases as evidence. A deterministic
fallback may retain the smallest original-variable subset that appears in the
selected source and still covers the already-claimed operation and result.

### `CALC_0007` — recoverable result-name span defect

The formula candidate `pe_ratio = market_price / eps` is directly supported:

- S159 introduces `PE ratios`;
- S161 names the current market price;
- S162 says to divide it by EPS.

The repaired entailment cited only S161-S162 for every identifier, then used
the quote `PE ratio` for the left-hand identifier. That quote actually occurs
at S159, so strict quote validation correctly rejected it.

Classification: **recoverable evidence-localization defect**.

Recommended generic repair: allow a small bounded adjacent extension to locate
the formula result name when the selected span independently grounds every
operand and operation and the adjacent text explicitly names the calculation
concept. The extension must only ground the result identifier; it must not
supply a missing operation.

### `CALC_0012` — recoverable extraction identifier defect

S226-S227 states that `204 billion dollars` scaled by ten is approximately
`two trillion dollars`. The model emitted:

```text
scaled_value = 204_billion_dollars * 10
```

`204_billion_dollars` is not a valid identifier because it begins with a digit.
The one model repair repeated the same invalid symbol, so extraction validation
correctly failed.

Classification: **recoverable schema-repair defect**.

Recommended generic repair: make the extraction-repair prompt explicitly
validation-aware for numeric-leading pseudo-identifiers: numeric quantities
must be numeric literals or must use a valid semantic identifier beginning
with a letter. Keep the parser rule unchanged. A deterministic repair may add
a fixed domain-neutral prefix to an otherwise identifier-shaped token, without
changing its components or semantic metadata.

## `cWzgk-8QHKk` — What is a Bond

### `CALC_0005` — insufficient source detail

S121-S125 discusses a 5% bond, 4% inflation, and a relative 1% outcome, then
says inflation must be subtracted when figuring bond value. It does not state
a unit-consistent reusable formula: `bond value`, nominal return, inflation
rate, and real return are used loosely and are not distinguished.

The audit repair also substituted `bond return`, a phrase absent from its cited
S124 evidence.

Classification: **genuinely insufficient for formula normalization**.

Recommended disposition: downgrade to a non-symbolic conceptual calculation
or retain `insufficient_source_detail`; do not manufacture a real-return or
inflation-adjustment formula from outside knowledge.

### `CALC_0009` — incoherent operand association

S193-S195 says the bond value rose to `$1,418` and that after adding two years
of coupon payments the amount made was `$518`. The inventory incorrectly treats
`1,418` and `518` as operands of an addition. The source never states the coupon
total in this local span, and `$518` is presented as a gain/result rather than
an addend.

Classification: **non-symbolic reported outcome with insufficient operands**.

Recommended disposition: downgrade through the existing generic
operation-result association logic. A result value must not be reused as an
operand merely because it is adjacent to an addition cue.

### `CALC_0010` — arithmetically inconsistent reported percentage

S185-S196 gives an initial `$1,000`, later reports `$518` made over two years,
and calls that about a `25` return. Direct division would be 51.8%, while an
approximately 25% annual simple average would require an unstated time
normalization. The transcript does not state the operation and the model repair
incorrectly claimed multiplication.

Classification: **non-symbolic/ambiguous reported outcome**.

Recommended disposition: downgrade rather than infer an annualization method.
Generic numeric-consistency checking can flag a reported result that matches no
claimed local operation without deciding what the speaker intended.

## Prioritized generic follow-up

1. Add source-extractive inventory reconciliation for semantic aliases that
   coexist with grounded original numeric operands (`CALC_0001`).
2. Add bounded adjacent result-name grounding after operands and operation are
   independently entailed (`CALC_0007`).
3. Make extraction repair explicitly handle numeric-leading pseudo-identifiers
   without changing the parser (`CALC_0012`).
4. Extend non-symbolic downgrade logic to catch result-as-operand association
   and locally inconsistent reported percentages (`CALC_0009`, `CALC_0010`).
5. Preserve the conceptual inflation example as insufficient/non-symbolic
   unless a future source states a unit-consistent relationship (`CALC_0005`).

Each follow-up should be covered by domain-neutral synthetic tests. These
fixtures may be retained as integration oracles, but production recovery code
must not mention shares, P/E, bonds, inflation, or the specific numeric values.
