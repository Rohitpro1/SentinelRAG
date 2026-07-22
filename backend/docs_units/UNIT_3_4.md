# Unit 3.4 — VerificationNode: Thin Orchestration Adapter Around VerificationAgent

**Status:** Complete. 10 new tests passing, 286/286 total (276 from
Unit 3.3's corrected count + 10 new).

## What this unit delivers
- `app/orchestration/nodes/verification.py` — `VerificationNode`,
  constructor-injected `VerificationAgent`, reads `state.retrieval_result`,
  writes both `verification_result` (`VerifiedEvidence`) and `diagnostics`
  (`VerificationDiagnostics`) back into state.
- `app/orchestration/graph_builder.py` — chaining logic generalized to a
  simple linear loop over `[retrieval, verification]`, so a third node
  slots in without new branching logic. Compiles
  `planner → [retrieval] → [verification] → END`, including only the
  nodes actually injected.

## Key engineering decisions

**1. `VerificationNode` writes both business and observability outputs
in one update — the split itself is preserved, not flattened.**
`state.model_copy(update={"verification_result": ..., "diagnostics": ...})`
sets both fields from the single `(VerifiedEvidence, VerificationDiagnostics)`
tuple `VerificationAgent.verify()` already returns (Unit 2.9's
business/observability split) — this node doesn't merge or reinterpret
that pair, just relays it.

**2. Defensive default for a missing `retrieval_result`.**
`state.retrieval_result.ranked_chunks if state.retrieval_result else []`
means `VerificationNode` can be tested and invoked standalone (no
`RetrievalNode` having run first) without crashing —
`test_no_retrieval_result_at_all_defaults_to_empty_chunks` covers this
directly. This wasn't explicitly requested but is the same kind of
defensive property `RetrievalNode` (Unit 3.3) didn't need but
`VerificationNode` does, since it's the second node in the chain and
reasonably might run against a state that skipped retrieval in a test.

**3. "Verification failures" required a purpose-built double, since
`VerificationAgent` doesn't normally raise.** Per Unit 2.13's graceful-
degradation design, `VerificationAgent.verify()` is built specifically
*not* to raise under NLI-provider failure. To satisfy instruction 6's
"verification failures" test category honestly, a
`_AlwaysFailingVerificationAgent` double was constructed (bypassing the
real `__init__` entirely) purely to prove `VerificationNode` doesn't add
its own error handling — mirroring `RetrievalNode`'s equivalent test
exactly, and stated here so it's clear this test exists to prove an
absence (of error handling), not a normal operating condition.

**4. `GraphBuilder`'s node-chaining was generalized, not re-special-cased.**
Unit 3.3 special-cased "if retrieval_node, do X, else Y." Adding a third
optional node the same way would have meant a third branch; instead,
`build()` now iterates `[("retrieval", ...), ("verification", ...)]`
and chains whichever are present. This is the shape `DecisionNode`
(next) will extend the same way — a fourth tuple in the list, not a new
conditional block — though `DecisionNode`'s eventual *conditional*
routing will be the point where this stops being a simple straight
line, explicitly noted as future work in the code.

## Test maintenance on prior units' suites (disclosed, not hidden)
Same pattern as Unit 3.3: making `VerificationNode` require constructor DI
broke two tests that assumed zero-arg construction —
`test_orchestration_nodes.py`'s placeholder parametrization (removed
`VerificationNode`, now covers only `DecisionNode`) and
`test_graph_builder.py`'s "nodes unused" test (renamed/re-scoped to cover
only `decision_node`, the one node still genuinely unused). Both fixes
only narrow scope to match current reality.

## Independence and isolation verified, not assumed
- `grep -nE "app\.services\.(retrieval|decision_engine)" app/orchestration/nodes/verification.py`
  → zero matches.
- File-mtime check confirms zero Milestone 2 files modified.

## What this unit deliberately does NOT do
- No retries.
- No decision logic — `state.decision` is untouched by this node.
- No planner behavior — doesn't read or write `planning_metadata`.

## Next unit (anticipated, not yet scoped)
`DecisionNode` wrapping `DecisionEngine` is the natural next step — and
the first node whose real behavior (conditional routing on
`Decision.action`) will require `GraphBuilder` to grow actual
conditional edges instead of the current straight-line chain.
