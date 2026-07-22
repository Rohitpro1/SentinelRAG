# Unit 3.6 — Conditional Routing: The Graph Becomes a Real Loop

**Status:** Complete. 18 new tests passing, 314/314 total (296 from
Unit 3.5 + 18 new). Zero Milestone 2 files modified — file-mtime check
confirms this, and the only `retry_count`-touching code this unit adds
is `RetryIncrementNode`, entirely in `app/orchestration/`.

## Naming reconciliation — flagged before implementation, not discovered after
The instruction specified four conceptual outcomes: `ANSWER`,
`RETRY_RETRIEVAL`, `CLARIFY`, `FAIL`. **The frozen `DecisionAction` enum
(Milestone 1) has no `ANSWER` or `FAIL` member** — its actual five values
are `PROCEED`, `RETRY_RETRIEVAL`, `CLARIFY`, `LOW_CONFIDENCE_RESPONSE`,
`HUMAN_REVIEW`. Adding either missing value to a frozen enum was not an
option. The mapping used, stated plainly in `app/orchestration/routing.py`:

| Instruction's term | Actual `DecisionAction` value(s) | Edge |
|---|---|---|
| ANSWER | `PROCEED`, `LOW_CONFIDENCE_RESPONSE` | → `END` |
| CLARIFY | `CLARIFY` | → `END` |
| FAIL | `HUMAN_REVIEW` (closest analog — no human-review-waiting mechanism exists yet, so this terminates rather than loops) | → `END` |
| RETRY_RETRIEVAL | `RETRY_RETRIEVAL` (exact match) | → `RetryIncrementNode` → `RetrievalNode` |

`LOW_CONFIDENCE_RESPONSE` grouping with `PROCEED` isn't arbitrary — the
frozen Architecture Enhancements lifecycle design explicitly states a
transparent low-confidence answer is a completed response, not a failure.

## What this unit delivers
- `app/orchestration/routing.py` — `route_after_decision(state) -> str`,
  a standalone pure function (not a method), so the routing decision
  itself is unit-testable without compiling a graph at all.
- `app/orchestration/nodes/retry_increment.py` — `RetryIncrementNode`,
  the *only* place in the entire codebase that increments a retry count
  as part of this loop — confirmed by grep against every Milestone 2
  service file.
- `app/orchestration/graph_builder.py` — now wires
  `add_conditional_edges("decision", route_after_decision, {"retry": "retry_increment", "end": END})`
  when a `retrieval_node` is present; falls back to Unit 3.5's straight
  `decision → END` when it isn't (nothing to retry into).

## Key engineering decisions

**1. Retry bookkeeping is a dedicated node, not inline routing-function
logic.** `route_after_decision()` only returns a string (`"retry"`/`"end"`)
— LangGraph conditional-edge functions don't mutate state. The increment
had to live somewhere else in the graph, so `RetryIncrementNode` sits
between the conditional edge and `RetrievalNode`'s re-entry. This keeps
the routing *decision* (pure, stateless) separate from the state
*mutation* (also pure, but a distinct concern) — two single-responsibility
pieces instead of one function doing both.

**2. The loop is bounded by `DecisionEngine`'s existing frozen ceiling —
not a new limit invented here.** Same principle established in Unit
2.14's `QueryService` and reaffirmed in Unit 3.5: once `retry_count`
reaches `DecisionEngineSettings.max_retrieval_retries`, the engine
returns `CLARIFY` instead of `RETRY_RETRIEVAL`, and the conditional edge
routes to `END`. `test_retry_ceiling_is_respected_exactly_not_off_by_one`
verifies this at three different ceiling values (0, 1, 3), and
`test_graph_terminates_and_does_not_hang_or_hit_recursion_limit` confirms
a generous ceiling (5) still resolves well within LangGraph's default
recursion limit — checked empirically, not assumed.

**3. A real test bug surfaced and was fixed, not the code.** The first
version of `test_answer_path_proceed_terminates_graph_without_retry`
wired only `RetrievalNode` (no `VerificationNode`) and expected `PROCEED`.
It failed — correctly. `DecisionNode` (Unit 3.5) only ever reads
`state.verification_result`, never `state.retrieval_result` directly; with
no `VerificationNode` in the chain, `verification_result` stays `None`,
and `DecisionNode`'s own defensive fallback (also Unit 3.5) correctly
treats that as "no evidence" — regardless of how much real evidence
`RetrievalNode` actually found. This is exactly right: it's a test that
didn't wire a realistic pipeline, not a bug in `DecisionNode`. Fixed by
adding `VerificationNode` to that test, matching how the full pipeline is
meant to be assembled.

## Test coverage against instruction 7's exact eight areas
1. **Answer path** — `PROCEED`/`LOW_CONFIDENCE_RESPONSE` → `END`, `retry_count` stays 0.
2. **Clarify path** — empty evidence past the retry ceiling → `CLARIFY`.
3. **Fail path** — `HUMAN_REVIEW` (the actual analog) → `END`, via both a unit-level routing check and a full compiled-graph run with a decision-engine double.
4. **Retry path** — loops back to `RetrievalNode`, then resolves.
5. **Retry ceiling respected** — checked at three different configured ceilings, plus a generous one for recursion-limit safety.
6. **Graph termination** — confirmed no hang, no recursion-limit error.
7. **`retry_count` updates** — starts at 0, increments exactly once per loop pass, never off-by-one.
8. **Deterministic execution** — two independent full runs of the same query produce identical `action`, `retry_count`, and `confidence_score`.

## Independence and isolation verified, not assumed
- File-mtime check: zero Milestone 2 files modified.
- `grep -rn "retry_count" app/services/` shows only pre-existing,
  unmodified usages (`QueryService`'s own separate while-loop from Unit
  2.14; `DecisionEngine` reading — never incrementing — `retry_count` as
  a threshold input). The only new increment is `RetryIncrementNode`.

## What this unit deliberately does NOT do
- No query rewriting — `rewritten_query` still isn't touched anywhere in
  the retry path; each retry re-runs `RetrievalNode` against the same
  `effective_query`.
- No migration of `QueryService`'s retry loop itself — `QueryService`
  (Unit 2.14) is untouched and still exists as a separate, working
  orchestration path; this unit builds the graph-native equivalent
  alongside it, not as a replacement (that migration decision, if wanted,
  is a future unit's call).

## Next unit (anticipated, not yet scoped)
Likely candidates: query-rewriting logic for the retry path (finally
giving `PlannerNode`/`RetrievalNode` something meaningful to do with
`rewritten_query` on a retry), or wiring this graph into `QueryService`
or the FastAPI layer as an alternative/replacement execution path.
