# Unit 3.3 — RetrievalNode: First Orchestration Wrapper Around a Milestone 2 Component

**Status:** Complete. 10 new tests passing, 277/277 total (267 from Unit
3.2's corrected count + 10 new).

## What this unit delivers
- `app/orchestration/nodes/retrieval.py` — `RetrievalNode`, genuinely
  implemented: constructor-injected `RetrieverAgent`, uses
  `GraphState.effective_query` (Unit 3.1's property — not reimplemented
  here), calls `search()` once, stores the `SearchResponse`.
- `app/orchestration/graph_builder.py` — now compiles
  `START → planner → retrieval → END` when a `retrieval_node` is
  injected; falls back to Unit 3.2's `START → planner → END` when it
  isn't, preserving the zero-argument construction guarantee.

## Key engineering decisions

**1. `RetrievalNode` is a pure adapter — verified by what it does NOT
contain.** No timeout, no retry, no exception handling. `RetrieverAgent`
(Unit 2.6, frozen) still owns all of that internally via
`EmbeddingService`/`SearchService`/`RerankingService`. A `RetrievalError`
raised by `RetrieverAgent.search()` propagates through `RetrievalNode`
completely unmodified —
`test_retrieval_error_propagates_uncaught` asserts this directly using a
double that always raises, confirming no swallowing, wrapping, or
translation happens at this layer.

**2. `RetrieverAgent` has no default in `GraphBuilder`, unlike
`PlannerNode`.** `PlannerNode()` is cheap and pure, safe to
default-construct. `RetrieverAgent` has real dependencies (embedder,
vector repository, reranker) — `GraphBuilder` never constructs one
itself; if `retrieval_node` isn't injected, the graph simply doesn't
include a retrieval step, falling back to Unit 3.2's shape rather than
failing. `test_without_retrieval_node_injected_graph_falls_back_to_planner_only`
confirms zero regression for existing zero-arg callers.

**3. `effective_query` is read, never written.** `RetrievalNode` uses
`state.effective_query` (falls back to `original_query` if no
`rewritten_query`) but never sets `rewritten_query` itself — query
rewriting stays entirely out of this node's responsibility, confirmed by
`test_rewritten_query_field_itself_is_unaffected_by_retrieval_node`.

## Test maintenance on Unit 3.1/3.2's own suite (disclosed, not hidden)
Making `RetrievalNode` require constructor DI (no zero-arg default) broke
two existing tests that assumed `RetrievalNode()` could be bare-constructed:
- `test_orchestration_nodes.py`'s "still placeholder" parametrization
  included `RetrievalNode` — removed (it's no longer a bare placeholder;
  it now requires DI and has its own dedicated real-behavior tests here).
- `test_graph_builder.py`'s test asserting `retrieval_node` stays "unused"
  is no longer true once injected — renamed and re-scoped to cover only
  `verification_node`/`decision_node`, which remain genuinely unused.

Both fixes only narrow what's asserted to match current, correct reality;
nothing was weakened to force a pass.

## Independence and isolation verified, not assumed
- `grep -nE "app\.services\.(verification|decision_engine)" app/orchestration/nodes/retrieval.py`
  → zero matches. `RetrievalNode` only imports `RetrieverAgent` and the
  schemas it needs.
- File-mtime check confirms zero Milestone 2 files (`app/services`,
  `app/repositories`, `app/api`, `app/infrastructure`) were modified.

## What this unit deliberately does NOT do
- No retries at the graph level (Unit 2.6's `RetrieverAgent` already
  retries transient failures internally; that's unchanged and untouched).
- No verification or decision logic.
- No query rewriting — `rewritten_query` is read, never set.

## Next unit (anticipated, not yet scoped)
Likely candidates: `VerificationNode` wrapping `VerificationAgent` (the
natural next step in the pipeline, with `RetrievalNode`'s
`SearchResponse` as its input), or beginning the query-rewrite logic that
`PlannerNode`/`RetrievalNode` both currently defer — whichever the next
instruction specifies.
