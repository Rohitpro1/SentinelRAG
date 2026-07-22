# Unit 3.8 — LangGraph Integration: QueryService Now Delegates to the Compiled Graph

**Status:** Complete. 9 new tests passing, 339/339 total (330 from Unit
3.7 + 9 new). `query_router.py` confirmed byte-for-byte unmodified since
Milestone 2 (mtime check). Core domain logic (`DecisionEngine`,
`EmbeddingService`, `SearchService`, `FusionService`, `RerankingService`,
`RetrieverAgent`, all of `app/services/verification/`) confirmed
untouched (mtime check).

## What this unit delivers
- `app/orchestration/graph_state.py` — gained `top_k`, `rerank_top_n`,
  `document_filter`, `request_id`, `trace_id`. Additive, defaults match
  `SearchRequest`'s own defaults exactly.
- `app/orchestration/nodes/retrieval.py` — `RetrievalNode` now threads
  all five new fields into `SearchRequest`.
- `app/orchestration/nodes/decision.py` — `DecisionNode` now passes
  `request_id`/`trace_id` into `DecisionEngine.evaluate()`.
- `app/services/query/query_service.py` — **rewritten**: constructor now
  takes a `CompiledStateGraph` (not three agents); `handle_query()`
  builds a `GraphState`, calls `ainvoke()`, reconstructs the final state,
  translates it into the unchanged `QueryResult`. The Python `while True`
  retry loop is gone — the graph's own conditional routing (Unit 3.6)
  does that job now.
- `app/api/dependencies.py` — `get_query_service()` now builds a
  compiled graph via `GraphBuilder` and injects it; every other function
  in that file (agent/service factories) is untouched.

## A gap that had to be closed before the integration could be correct
`GraphState` never carried `top_k`/`rerank_top_n`/`document_filter`/
`request_id`/`trace_id` — no unit before this one needed the graph to
represent a full API-equivalent request. Had I integrated the graph
without fixing this, `QueryService`'s public parameters would have
silently become no-ops the moment its internals switched to graph
execution — a real regression that instruction 3 ("preserve the public
API") explicitly forbids, even though the *route* itself would have
looked unchanged. Caught and fixed before writing `QueryService`'s
rewrite, not discovered after. `RetrievalNode`/`DecisionNode` are
orchestration-layer files (Units 3.3/3.5), not Milestone 2 — modifying
them was in scope for instruction 7's "Milestone 2 business logic"
constraint, which protects `DecisionEngine`/`RetrieverAgent`/
`VerificationAgent`/their sub-components, not the graph nodes wrapping them.

## Key engineering decisions

**1. `QueryService`'s public contract verified identical, not just
claimed unchanged.** `test_query_result_shape_matches_pre_graph_integration_contract`
asserts `QueryResult`'s field set is exactly `{decision, diagnostics,
retry_count}` — nothing added, nothing removed.
`test_handle_query_signature_accepts_all_original_parameters` exercises
every Unit 2.14 parameter. And critically: **every assertion in Unit
2.14's original `test_query_service.py` passes unchanged** — I only
rewrote how the test *builds* a `QueryService` (via a compiled graph
instead of three raw agents), never what it *expects*.

**2. DI is checked structurally, not just by signature.**
`test_query_service_never_imports_graphbuilder_itself` inspects the
module's actual `import`/`from` lines (excluding docstring prose that
legitimately mentions "GraphBuilder" when explaining the design) to
confirm `QueryService` genuinely never constructs its own graph.

**3. No defensive fallback for a misconfigured graph.** If the injected
graph is missing a `DecisionNode` or `VerificationNode`, `QueryResult`'s
required fields (`decision`, `diagnostics`) simply fail Pydantic
validation with a clear error, rather than `QueryService` silently
working around a broken composition-root wiring. Matches this project's
established fail-loudly-on-misconfiguration principle.

**4. The low-confidence test was built from settings, not embedding
coincidence.** An earlier draft tried to land `LOW_CONFIDENCE_RESPONSE`
by picking "weak-looking" retrieval text, but with an exact query/chunk
text match, `DeterministicEmbedder` gives similarity `1.0` — and with
default weights, `0.45×1.0 + 0.20×1.0` alone already exceeds the
`0.60` low-confidence threshold regardless of reliability, making the
scenario unreachable that way. Fixed by using
`min_retrieval_similarity=0.0` (guarantees no retry) and
`low_confidence_threshold=0.99` (guarantees this low-reliability
evidence can't clear it) — a deterministic construction instead of a
coincidental one.

## Test coverage against instruction 6's exact six areas
1. **Successful graph execution** — full `QueryResult` with explainability.
2. **Retry execution** — loop actually iterates and terminates at the ceiling.
3. **Clarification flow** — end-to-end through `CLARIFY`.
4. **Low-confidence response** — `LOW_CONFIDENCE_RESPONSE`, deterministically constructed.
5. **Human-review outcome** — contradiction detection routes through to `HUMAN_REVIEW`.
6. **API compatibility** — both the response shape and the full parameter
   surface verified unchanged; the 13 pre-existing FastAPI endpoint tests
   (Unit 2.14) all pass unmodified in assertions, only their helper's
   construction changed.

## Independence and isolation verified, not assumed
- `find app/services/decision_engine app/services/retrieval/{embedding,search,fusion,reranking}_service.py app/services/retrieval/retriever_agent.py app/services/verification -newer <Milestone-2-baseline>` →
  zero matches.
- `find app/api/v1/query_router.py -newer <Milestone-2-baseline>` → zero
  matches (route file byte-for-byte unchanged).

## What this unit deliberately does NOT do
- Does not touch `DecisionEngine`, `RetrieverAgent`, `VerificationAgent`,
  or any of their sub-components.
- Does not change the FastAPI route, request/response DTOs, or exception
  handlers.
- Does not add query-rewriting improvements — Unit 3.7's rewrite rule is
  used exactly as-is through the now-integrated graph.

## Next unit (anticipated, not yet scoped)
With `QueryService` now graph-backed and `PROCEED`/`LOW_CONFIDENCE_RESPONSE`
producing a `Decision` but no natural-language answer, a
`ReasoningAgent`/`ResponseGenerator` node is the natural next gap — the
Retrieval Domain Design's original pipeline diagram still has a
"Reasoning Agent" stage this project hasn't built yet.
