# Unit 2.11 — OpenAIEmbedder (Real Embedding Provider) + EmbeddingResult/EmbedderHealth

**Status:** Complete. 20/20 new deterministic tests passing (164/164
total default suite), plus 3 isolated integration tests that skip
cleanly without a configured live endpoint.

## What this unit delivers
- `app/schemas/embedding.py` — `EmbeddingResult` (vector as primary
  output + `provider`, `model_name`, `embedding_dimensions`,
  `generation_latency_ms`, `model_version`) and `EmbedderHealth`
  (`READY`/`DEGRADED`/`UNAVAILABLE` + optional detail string).
- `app/services/embedding/result_builder.py` — one shared function that
  wraps any `BaseEmbedder.embed_query` call into an `EmbeddingResult`,
  used by both `DeterministicEmbedder` and `OpenAIEmbedder` so the
  metadata-wrapping logic exists exactly once.
- `app/services/embedding/openai_embedder.py` — `OpenAIEmbedder`,
  implementing `BaseEmbedder` exactly (unchanged interface), plus the
  additive `embed_query_with_result()` and `health()` capabilities.
- `app/infrastructure/embedding_client_factory.py` — the one place
  permitted to construct the `httpx.AsyncClient` used by `OpenAIEmbedder`.
- `EmbeddingSettings` gained provider connection config
  (`api_base_url`, `api_key`, `request_timeout_ms`,
  `unavailable_after_consecutive_failures`) — additive fields only.

## Key engineering decisions

**1. `BaseEmbedder` was NOT modified.** `embed_query`, `embed_batch`,
`dimensions` are exactly as Unit 2.4 defined them. `EmbeddingResult` and
`health()` are additive capabilities present on both concrete
implementations (`DeterministicEmbedder`, `OpenAIEmbedder`) via a shared
helper, not new abstract methods on the interface. This was the central
design question in this unit — the instruction explicitly forbade
changing the interface "unnecessarily," and adding required abstract
methods would have broken the principle that any `BaseEmbedder`
implementation, including a hypothetical minimal future one, only ever
needs to implement the original three members.

**2. Deterministic-first was honored literally**, not just in spirit:
`DeterministicEmbedder` gained `embed_query_with_result()`/`health()`
*before* `OpenAIEmbedder`'s tests were written, and
`test_embed_query_with_result_vector_matches_embed_query` proves the new
convenience method doesn't diverge from the existing one — the pattern
was proven deterministic first, then reused for the real provider.

**3. Retry asymmetry mirrors Unit 2.10 exactly, for the same reason.**
`embed_query()` has zero internal retry (`EmbeddingService`, Unit 2.6,
already owns that). `embed_batch()` gets one bounded retry, because no
`IngestionService` exists yet to own retry for the batch/ingestion path —
flagged in the code as something to remove once that service exists, not
a permanent choice. `test_embed_query_never_retries_internally` and
`test_embed_batch_gives_up_after_bounded_retry` assert this directly.

**4. `EmbeddingError` stayed always-transient, per the frozen design.**
Unlike `QdrantVectorRepository`'s `RetrievalError` (which distinguishes
4xx from 5xx), `EmbeddingError` doesn't — that asymmetry was decided in
Unit 2.6/the Architecture Enhancements addendum, not introduced here.
`test_embedding_error_is_always_transient_per_frozen_design` exists
specifically to catch a future accidental change to that frozen behavior.

**5. One class covers OpenAI and OpenAI-compatible endpoints** (Ollama,
vLLM, etc.) via `api_base_url`, rather than one class per provider.
Documented as a deliberate scope decision — a provider with a genuinely
different request/response shape would need its own class, but nothing
in scope today has one.

## Network constraint, stated plainly
This development sandbox has no route to `api.openai.com` or any
embedding provider. All 20 new tests validate real request-building,
response-parsing, and error-translation logic via `httpx.MockTransport`
(zero real network) — this is a legitimate, complete test of the
provider's own logic, but it is **not** the same claim as "verified
against a live OpenAI account." The integration test file is written to
run for real against any OpenAI-compatible endpoint and was executed here
only in its skip path (confirmed: 3/3 skip cleanly rather than erroring),
exactly like Unit 2.10's live-Qdrant test.

## Next unit
**2.12 — real cross-encoder behind `BaseReranker`.** Same pattern
expected: interface untouched, `DeterministicReranker` stays default,
real implementation's HTTP/model-loading logic tested via mocks/fakes
first, live-model integration test isolated and optional.
