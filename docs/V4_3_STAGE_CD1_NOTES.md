# v4.3 Stage C–D.1 Resilience Patch

This patch responds to a confirmed 900-second timeout on a single 20,132-character inventory request.

It adds:

- Bounded, overlapping calculation-inventory chunks
- Deterministic merge and deduplication of chunk inventories
- Stable global calculation IDs after merge
- Stage, segment range, prompt size, elapsed time, and cache status logging
- Bounded `num_predict` generation
- Ollama `keep_alive`
- Environment-driven timeout and chunk settings
- Atomic progress checkpoints for inventory, extraction, and entailment
- Resume behavior keyed by source SHA, prompt version, and stage input SHA
- Stage-aware timeout errors

It remains domain-neutral and contains no subject-specific formula recovery.

Default inventory settings:

- 40 segments per chunk
- 6-segment overlap
- 1,536 predicted tokens per inventory response
- 300-second request timeout
- 30-minute Ollama keep-alive

The production v4.1.1 runner remains untouched.
