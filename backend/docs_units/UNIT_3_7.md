# Unit 3.7 — Planner Query Rewriting: Retry-Aware PlannerNode

**Status:** Complete. 16 new tests passing, 330/330 total (314 from
Unit 3.6 + 16 new). `RetrievalNode` confirmed genuinely untouched (mtime
predates this unit's start). Zero Milestone 2 files modified.

## What this unit delivers
- `app/orchestration/nodes/planner.py` — `PlannerNode` extended with
  `_rewrite_query()`, a deterministic stopword-stripping rewrite rule,
  applied only when `retry_count > 0`.
- `app/orchestration/graph_builder.py` — the retry loop's target changed
  from `retrieval` directly to `planner`, so the rewrite actually
  executes on every retry pass instead of being silently bypassed.

## The architectural point that had to be addressed before any code
`RetrievalNode` already reads `GraphState.effective_query` (Unit 3.1),
which prefers `rewritten_query` when present — but Unit 3.6's retry loop
routed `retry_increment → retrieval` **directly**, skipping `planner`
entirely. Had this unit only touched `PlannerNode`, the new rewrite logic
would never run on a retry — `PlannerNode` only executes once, at graph
entry. Fixing this required changing one edge target in `GraphBuilder`
(`retry_increment → planner` instead of `→ retrieval`; `planner → retrieval`
already existed, so this one change is sufficient). This is an
orchestration-layer wiring change, not a change to `RetrievalNode` or any
Milestone 2 component — confirmed both by what instruction 4/7 actually
forbid (modifying `RetrievalNode` or Milestone 2) and by direct
verification that `retrieval.py`'s file contents are unchanged.

## Key engineering decisions

**1. The rewrite rule is deliberately idempotent, not progressively
broadening.** `_rewrite_query()` always strips the same stopword set from
`original_query`'s normalized form, regardless of which retry attempt
it's on — retry 1, 2, and 3 all produce the identical rewritten query for
the same original query (`test_rewrite_is_stable_across_multiple_retries`
confirms this directly). A progressively broader rewrite (e.g. dropping
an additional keyword per retry) was considered and explicitly not
built, since it wasn't requested and would add behavior beyond what was
asked — documented as a deliberate scope decision in the code, not
silently omitted.

**2. The rewrite always derives fresh from `original_query`, never from
a prior `rewritten_query`.** This keeps the rewrite predictable — "what
would rewriting produce for this original query" has exactly one answer,
independent of retry history. It also means a manually pre-set
`rewritten_query` on the input state gets **overwritten** the moment a
real retry occurs, which is correct: the whole point of retry-aware
planning is that the orchestration layer's own rewrite takes over from
that point on.

**3. Fallback for all-stopword queries.** If stripping stopwords would
leave nothing (e.g. `"what is this"` strips to empty), the rewrite falls
back to the normalized original query unchanged, rather than producing
an empty, unusable search query — `test_rewrite_falls_back_to_original_when_all_words_are_stopwords`
covers this directly.

**4. `planning_metadata` still reflects `original_query`, never the
rewrite.** Classification and normalization are unaffected by whether a
rewrite happened — `test_graph_planning_metadata_still_reflects_original_query_not_rewrite`
confirms this. The rewrite is purely an addition to what `PlannerNode`
already did, not a change to its existing classification behavior.

## Test bug found and fixed (disclosed, not hidden) — same category as Unit 3.6
`test_graph_uses_rewritten_query_end_to_end` (originally Unit 3.3) preset
a `rewritten_query` on the input state and expected it to reach
`RetrievalNode` unchanged. It broke here for the same underlying reason
Unit 3.6's answer-path test broke: `GraphBuilder` has defaulted a real
`DecisionNode` in since Unit 3.5, and without a `VerificationNode` also
wired in, `DecisionNode` correctly sees "no evidence" regardless of what
`RetrievalNode` actually found, triggers an unintended retry, and — as of
this unit — that retry correctly overwrites the preset value with a
freshly-derived rewrite. Fixed by adding `VerificationNode` to the test,
matching how the pipeline is actually meant to be assembled. The failure
was proof the new code works correctly, not a defect in it.

## Test coverage against instruction 7's exact six areas
1. **First-pass planning** — `rewritten_query` stays `None`/unset.
2. **Retry planning** — a real rewrite is generated once `retry_count > 0`.
3. **Rewritten query usage** — `effective_query` reflects the rewrite;
   confirmed both in isolation and through a real compiled-graph run
   where the final retrieval attempt's `SearchRequest.query` shows the
   rewrite was actually used, not just computed and discarded.
4. **Preservation of `original_query`** — asserted across three
   sequential retries in a row.
5. **Multiple retries** — rewrite stability confirmed across retry counts
   1, 2, 3.
6. **Graph execution through the retry path** — a full compiled graph
   with an empty index runs to `CLARIFY` at the configured ceiling, with
   `rewritten_query` populated on the final state.

## Independence and isolation verified, not assumed
- `RetrievalNode`'s file contents confirmed unchanged via mtime
  comparison against `UNIT_3_6.md` (predates this unit's start).
- File-mtime check confirms zero Milestone 2 files modified.

## What this unit deliberately does NOT do
- No LLM-based rewriting — the rule is a fixed stopword-strip, nothing
  probabilistic or model-based.
- No progressive query broadening across multiple retries (see decision 1 above).

## Next unit (anticipated, not yet scoped)
Likely candidates: wiring this LangGraph pipeline into the FastAPI layer
as an alternative to `QueryService` (Unit 2.14), or beginning work on a
`ResponseGenerator`/`ReasoningAgent` node now that `PROCEED`/
`LOW_CONFIDENCE_RESPONSE` outcomes exist but produce no natural-language
answer yet.
