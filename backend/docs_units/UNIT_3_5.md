# Unit 3.5 — DecisionNode: Thin Orchestration Adapter Around DecisionEngine

**Status:** Complete. 11 new tests passing, 296/296 total (285 from
Unit 3.4's corrected count + 11 new). Graph confirmed still a straight
line — `grep` shows only `add_edge` calls in `graph_builder.py`, zero
`add_conditional_edges`.

## What this unit delivers
- `app/orchestration/nodes/decision.py` — `DecisionNode`,
  constructor-injected `DecisionEngine`, builds a `VerificationReport`
  from `state.verification_result` and calls `evaluate()` exactly once.
- `app/orchestration/graph_builder.py` — chain extended to
  `planner → [retrieval] → [verification] → [decision] → END`.
  `decision_node` now defaults to a real instance (like `planner_node`),
  since `DecisionEngine` is pure with no external I/O — safe to
  default-construct, unlike `RetrieverAgent`/`VerificationAgent`.

## Key clarification worth stating explicitly
Instruction 2 said "read: SearchResponse, VerifiedEvidence,
VerificationDiagnostics." In practice, **`DecisionEngine.evaluate()`
(Milestone 1, frozen) accepts exactly one argument — a
`VerificationReport`** — built from `VerifiedEvidence` via Unit 2.9's
existing `.to_verification_report()` adapter. `SearchResponse` and
`VerificationDiagnostics` are present on `GraphState` and not discarded,
but there is no parameter on the frozen `DecisionEngine` contract to pass
either one into. `DecisionNode` doesn't invent one — it uses exactly what
the frozen engine already accepts. Documented here rather than silently
treated as if all three were consumed, which would misrepresent what the
code actually does.

## Key engineering decisions

**1. `DecisionNode` now defaults into `GraphBuilder`, same as `PlannerNode`.**
Both are pure/cheap with no external dependencies, so both get a safe
default (`PlannerNode()`, `DecisionNode(DecisionEngine(DecisionEngineSettings()))`).
`RetrieverAgent`/`VerificationAgent` still have no defaults — their real
infrastructure dependencies remain a composition-root concern.

**2. This default changed the meaning of "minimal graph" and
"planner-only" in three existing tests — fixed, not hidden.** Once
`DecisionNode` defaults in, even `GraphBuilder()` with zero arguments now
produces a real, non-`None` `Decision` (computed from empty evidence via
`DecisionNode`'s defensive fallback, same pattern as `VerificationNode`'s
missing-`retrieval_result` handling in Unit 3.4). Three tests that
predated this and asserted `decision is None` or "decision passed in as
input is preserved unchanged" were now testing something false:
- `test_graph_builder.py`'s `test_minimal_graph_preserves_populated_optional_fields`
  → replaced with `test_default_graph_recomputes_decision_via_default_decision_node`,
  which asserts the *correct* new behavior (a stale input decision gets
  overwritten by a freshly computed one) rather than the old, now-wrong
  expectation.
- `test_graph_builder.py`'s `test_constructor_accepts_injected_decision_node_without_using_it_yet`
  → its entire premise (some node being "injected but unused") no longer
  applies to *any* node as of this unit, since all four are now genuinely
  implemented and planner/decision both default in. Replaced with
  `test_all_four_nodes_are_used_once_all_are_injected_or_defaulted`,
  which checks the new reality directly.
- `test_graph_planner_node.py`'s `test_planner_node_via_graph_leaves_other_fields_untouched`
  → renamed and rescoped to assert what's still actually true
  (`retrieval_result`/`verification_result` remain `None`), dropping the
  now-false `decision is None` assertion.

**3. `test_orchestration_nodes.py`'s placeholder list is now empty.**
All four node interfaces are genuinely implemented as of this unit —
`_STILL_PLACEHOLDER_NODES` is kept as a named empty list (not deleted)
so this file's structure doesn't need to change again if a future node
interface is ever added to the project.

**4. Empty-evidence + retry budget produces `RETRY_RETRIEVAL`, not
`CLARIFY`, on the first attempt.** This is `DecisionEngine`'s existing,
frozen, unmodified threshold logic (Milestone 1) — with `retry_count=0`
below `max_retrieval_retries`, no chunks routes to retry; only once
`retry_count` reaches the ceiling does it fall to `CLARIFY`. Both paths
are tested explicitly and distinctly
(`test_empty_retrieval_results_with_retry_budget_remaining_returns_retry`
vs. `test_empty_retrieval_results_after_retry_budget_exhausted_returns_clarify`)
rather than assuming one or the other.

## Independence and isolation verified, not assumed
- `grep -nE "app\.services\.(retrieval\.|verification\.)" app/orchestration/nodes/decision.py`
  → zero matches.
- File-mtime check confirms zero Milestone 2 files modified.
- `grep -n "add_conditional_edges\|add_edge" app/orchestration/graph_builder.py`
  → only `add_edge` calls exist, confirming no branching was introduced,
  per explicit instruction 3.

## What this unit deliberately does NOT do
- No graph branching — the chain is still a straight line.
- No migration of `QueryService`'s (Unit 2.14) retry loop into the graph.
  `DecisionNode` computes one `Decision` per graph run; nothing re-invokes
  `RetrievalNode` on `RETRY_RETRIEVAL` yet.

## Next unit (anticipated, not yet scoped)
The natural next step is introducing conditional edges on
`Decision.action` — this is where `GraphBuilder`'s straight-line chain
finally becomes a real branching graph, and where `QueryService`'s
existing retry-loop logic would migrate into a native LangGraph edge
back to the retrieval node.
