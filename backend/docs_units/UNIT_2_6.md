# Unit 2.6 — RetrieverAgent (Thin Orchestrator) + Its Composed Services

**Status:** Complete. 23/23 new tests passing (4 embedding + 4 search + 4
fusion + 6 reranking + 5 agent), 96/96 total.

## What this unit delivers
Per the review instruction to keep `RetrieverAgent` extremely small and
move logic into dedicated services:

- `EmbeddingService` — timeout + agent-level retry around `BaseEmbedder`,
  raises `EmbeddingError` after exhausting `RetrievalSettings.max_transient_retries`.
- `SearchService` — same pattern around `VectorRepository`, raises
  `RetrievalError(transient=True)`.
- `FusionService` — dedup-by-`chunk_id` + descending sort. Real, tested
  logic, not a stub — see the gap note below.
- `RerankingService` — timeout around `BaseReranker`, but **never raises**:
  any failure (exception or timeout) degrades to similarity-ranked results
  with `rerank_score=None`, per the frozen design's non-fatal-reranker-failure
  policy. `test_never_raises_regardless_of_reranker_failure` asserts this
  invariant directly.
- `RetrieverAgent` — 109 lines including docstrings; composes the four
  services above plus an optional `CacheRepository`. Contains zero
  timeout/retry/degradation logic itself — it only sequences calls, times
  stages for `stage_latencies_ms`, and emits one summary log event.

## Boundary decision worth flagging explicitly
The review's recommended composition listed `VerificationService` as a
stage under `RetrieverAgent`. This was **not** implemented that way:
the frozen Retrieval Domain Design's dependency graph (Section 5) has
`RetrieverAgent` and `VerificationAgent` as siblings, both producing
inputs that flow toward `DecisionEngine`, with `RetrieverAgent` never
depending on verification. Nesting verification inside `RetrieverAgent`
would reopen a frozen architectural decision without a bug or performance/
security justification (the only three conditions under which redesign is
permitted, per the "architecture is locked" instruction). `RetrieverAgent.search()`
returns `SearchResponse`; wiring its output into `VerificationAgent`
remains the caller's job, to be built in Unit 2.9 exactly as planned.

## Known gap, stated explicitly (not hidden)
`FusionService` currently only deduplicates and sorts semantic-search
results — there is no keyword-search fusion yet, because no
`KeywordRepository` interface exists in the frozen design (only
`VectorRepository` was specified). Introducing one now would be a new
architectural surface, out of scope for implementation-mode work.
`FusionService.fuse()` is structured so that adding a keyword-results
parameter and a fusion strategy (e.g. reciprocal rank fusion) later is a
change to this one method's internals, not to `RetrieverAgent` or any
other caller — verified by the fact that `RetrieverAgent` only ever calls
`fusion_service.fuse(semantic_results)` and has no fusion logic of its own
to update.

## Test coverage highlights
- **Retry recovery is actually exercised**, not just retry-then-fail:
  `FlakyEmbedder`/`FlakyVectorRepository` fail a controlled number of times
  then succeed, proving the retry loop recovers within budget, not just
  that it eventually raises.
- **Timeout and generic-exception paths are both covered** for each
  service, using dedicated `AlwaysTimingOut*` test doubles alongside the
  flaky ones.
- **Cache hit/miss is exercised at the agent level**, including that two
  different requests produce two different cache keys (no false-positive
  cache hits across distinct queries).
- Every new service stays under 60 lines; `RetrieverAgent` itself is the
  largest at 109 lines, most of which is constructor wiring and docstring
  — confirmed no God Object emerged from this decomposition.

## Next units
**2.7** is effectively folded into this unit (reranking was built directly
into the service chain rather than added in a separate pass, since the
review instruction restructured the composition before 2.7 would have
run standalone). **2.8** (semantic cache integration) is likewise mostly
complete — `RetrieverAgent` already uses `CacheRepository` for get/set;
remaining work for a dedicated 2.8 pass would be `invalidate()` wiring
into the ingestion/document-delete path, which has no caller yet since
no ingestion API exists. **Next substantive unit: 2.9 — `VerificationAgent`
interface + deterministic NLI verifier**, producing `VerificationOutput`
and proving the full cross-domain path into `DecisionEngine.evaluate()`
using `RetrieverAgent`'s real output rather than hand-built fixtures.
