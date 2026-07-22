# Unit 2.12 — CrossEncoderReranker (Real Reranking Provider) + RerankResult/RerankerCapabilities

**Status:** Complete. 18/18 new deterministic tests passing (182/182
total default suite), plus 2 isolated integration tests that skip
cleanly without a configured live endpoint.

## What this unit delivers
- `app/schemas/reranking.py` — `RerankResult` (`ranked_chunks` primary +
  `provider`, `model_name`, `model_version`, `rerank_latency_ms`) and
  `RerankerCapabilities` (`supports_batching`, `max_batch_size`,
  `max_input_tokens`, `model_dimensions`).
- `app/core/settings/reranking.py` — `RerankingSettings`, the reranking
  domain's first dedicated settings class, registered in the composition
  root alongside the six existing domains.
- `app/services/reranking/result_builder.py` — shared `RerankResult`
  construction, used by both `DeterministicReranker` and
  `CrossEncoderReranker`.
- `app/services/reranking/cross_encoder_reranker.py` — `CrossEncoderReranker`,
  implementing `BaseReranker` exactly (unchanged interface), plus
  `rerank_with_result()` and `capabilities()`.
- `app/infrastructure/reranking_client_factory.py` — sole place permitted
  to construct the `httpx.AsyncClient` used by `CrossEncoderReranker`.

## Key engineering decisions

**1. `BaseReranker` was NOT modified** — identical reasoning to Unit
2.11's `BaseEmbedder` decision. `RerankResult` and `capabilities()` are
additive capabilities on both concrete classes via the shared
`build_rerank_result()` helper, not new abstract methods.

**2. Capabilities are strictly configuration-driven — verified, not just
claimed.** `test_capabilities_never_makes_a_network_call` asserts zero
HTTP calls happen when `capabilities()` is invoked. `RerankingSettings`
owns the numbers; nothing queries the provider to ask what it supports.

**3. `supports_batching=False` degrades to one-request-per-candidate
rather than failing.** This wasn't explicitly specified but follows
directly from "capabilities remain configuration-driven": if a configured
provider doesn't batch, the reranker still has to work — it just costs
more requests. `test_supports_batching_false_sends_one_request_per_candidate`
confirms this path.

**4. Zero internal retry — the strongest case yet for that pattern.**
Unlike Units 2.10/2.11 (where retry-owning services exist for the read
path but not always the write/batch path), `RerankingService` (Unit 2.6)
already catches *any* exception from `rerank()` and degrades immediately.
A retry loop here would only delay a path explicitly designed to fail
fast — there's no scenario where retrying helps. This is documented as a
stronger, not just consistent, version of the reasoning used in the prior
two units.

**5. Batch splitting and result merging are real, tested logic** —
`max_batch_size=3` against 7 candidates produces batches of `[3, 3, 1]`
(asserted directly), and results from multiple batches are correctly
merged, re-sorted, and truncated to `top_n` with no duplicates or drops
(`test_results_merged_correctly_across_batches`).

## Network constraint, stated plainly (same as Unit 2.11)
No route to any reranking provider from this sandbox. All 18 new tests
validate real request-building, batching, response-parsing, and
error-translation logic via `httpx.MockTransport`. The integration test
targets the Cohere-style rerank API shape and was executed here only in
its skip path (2/2 confirmed skipping cleanly, not erroring).

## Next unit
**2.13 — real NLI model behind `BaseNLIVerifier`.** Expected to follow
the exact same shape: interface untouched, `DeterministicNLIVerifier`
stays default, real implementation tested via mocks first, live-model
integration isolated and optional. This will complete the fake→real
sequence for all three of `RetrieverAgent`'s/`VerificationAgent`'s model
dependencies (embedding, reranking, NLI).
