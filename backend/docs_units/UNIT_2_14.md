# Unit 2.14 — FastAPI `/query` Endpoint + QueryService

**Status:** Complete. 18 new tests passing (5 `QueryService` + 13 endpoint),
224/224 total default suite.

## What this unit delivers
- `app/services/query/query_service.py` — `QueryService`, orchestrating
  `RetrieverAgent` → `VerificationAgent` → `DecisionEngine`, **including
  actually closing the `RETRY_RETRIEVAL` loop** by re-invoking
  `RetrieverAgent`.
- `app/schemas/query.py` — `QueryResult`, the domain-level return type.
- `app/api/dependencies.py` — the sole module permitted to construct
  services/infrastructure; deterministic-backed by default.
- `app/api/v1/schemas.py` — `QueryRequestBody`/`QueryResponseBody`/
  `ErrorResponseBody`, deliberately distinct from domain schemas.
- `app/api/exception_handlers.py` — domain exception → HTTP mapping.
- `app/api/v1/query_router.py` — the 32-line `/query` route.
- `app/main.py` — app assembly (router + exception handlers + logging).

## Key engineering decisions

**1. `QueryService` is the first class in the codebase permitted to
depend on `RetrieverAgent`, `VerificationAgent`, and `DecisionEngine`
simultaneously — verified, not just claimed.** `grep` confirms exactly
two files import `RetrieverAgent`: `app/api/dependencies.py` (DI wiring)
and `query_service.py` itself. This is the intended shape: an
orchestration layer sits *above* the frozen sibling domains and glues
them together; the domains themselves remain unaware of each other.

**2. The `RETRY_RETRIEVAL` loop is real, not a stub.** Before this unit,
`DecisionEngine` could *decide* to retry (Milestone 1) and
`RetrieverAgent` could *be* retried (Unit 2.6), but nothing actually
closed that loop — Unit 2.9's cross-domain test called each agent
exactly once. `QueryService.handle_query()` loops until the decision is
no longer `RETRY_RETRIEVAL`, incrementing `retry_count` each pass. It has
no independent iteration cap of its own — it's bounded entirely by
`DecisionEngine`'s existing, frozen, unmodified threshold logic (once
`retry_count` reaches `max_retrieval_retries`, the engine returns
`CLARIFY` instead). `test_handle_query_retries_on_empty_index_then_clarifies`
proves this terminates correctly at exactly the configured ceiling.

**3. Known gap, stated plainly: retries reuse the same query text.** True
query rewriting needs a Planner Agent, which isn't part of the approved
Milestone 2 unit list (Milestone 3 territory). The loop still terminates
correctly without one — see above — so this is a quality gap (a retried
query with no rewrite may not actually retrieve anything new), not a
correctness or termination risk.

**4. `ValidationError` is deliberately overridden to 400.** FastAPI's
default for `RequestValidationError` is 422; instruction 3 asked for 400
explicitly, so `validation_error_handler` is registered specifically for
`RequestValidationError`, overriding the framework default.
`test_query_empty_string_returns_400_not_422` names this distinction
directly in its test name so a future accidental revert is obvious.

**5. `EmbeddingError` is grouped with `RetrievalError` under 503,** even
though they're distinct exception types — both are retrieval-path
infrastructure failures from a client's perspective. Not explicitly
listed in the instruction's four examples, but a reasonable extension
documented as such.

**6. Every error handler logs the real exception server-side and returns
a generic message — verified, not assumed.** `test_retrieval_error_maps_to_503`,
`..._502`, `..._500` each assert the internal message (`"simulated ..."`)
is **absent** from the response body, not just that the status code is
right.

**7. Default DI wiring is deterministic**, matching the pattern
established across Units 2.10–2.13: the app runs correctly with zero
external services out of the box. Swapping in real infrastructure
(`QdrantVectorRepository`, `OpenAIEmbedder`, `CrossEncoderReranker`,
`LLMBasedNLIVerifier` — all already built) is confined to
`app/api/dependencies.py` alone.
`test_query_endpoint_uses_default_deterministic_wiring_when_not_overridden`
exercises the app's *actual* default wiring (not a test substitute) to
confirm this is real, not aspirational.

## Testing strategy (instruction 5)
All 18 new tests run against deterministic implementations. Endpoint
tests use `app.dependency_overrides` to inject a pre-populated
`QueryService` for the happy path, and lightweight `QueryService`
subclass doubles to trigger each exception-mapping branch on demand —
no real infrastructure anywhere in the default suite. No live-infra
integration test was added for this unit specifically, since the
endpoint's own infrastructure dependencies (Qdrant, LLM providers) are
already covered by Units 2.10–2.13's isolated integration tests; wiring
those into `app/api/dependencies.py` for a full end-to-end live test is
a natural candidate for Unit 2.15 (docker-compose) rather than this one.

## Next unit
**2.15 — docker-compose for local dev infra** (Qdrant + Postgres +
Redis), validated by Unit 2.10's integration test actually passing
against it, completing the originally approved Milestone 2 plan.
