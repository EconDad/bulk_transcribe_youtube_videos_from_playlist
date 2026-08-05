# v4.3 Stage C-D.2 Entailment Robustness

This patch is domain-neutral.

It allows a derived AST node to use direct source evidence when no earlier
formula node exists. Derived nodes with dependencies must depend only on
earlier AST nodes.

Invalid formula-extraction or entailment responses are recorded as rejected
diagnostics. They no longer terminate the entire video run.

The prompts and checkpoint version remain unchanged, so valid Stage C-D.1
inventory and extraction checkpoints can be resumed.
