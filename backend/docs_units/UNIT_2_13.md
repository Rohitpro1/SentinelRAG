# Unit 2.13 — LLMBasedNLIVerifier (Real NLI Provider) + NLIResult + Graceful Degradation

**Status:** Complete. 206/206 tests passing in the default suite (up from
182 before this unit — new tests: 16 for LLMBasedNLIVerifier, 6 for
ContradictionDetector's new degradation behavior, 3 additive to
DeterministicNLIVerifier's test file, plus 2 isolated integration tests
that skip cleanly without a configured live endpoint).

## What this unit delivers
- `app/schemas/nli.py` — `NLIResult` (`label` primary business output +
  `provider`, `model_name`, `model_version`, `latency_ms`, `confidence`
  as observability — see the deliberate confidence-placement note below).
- `app/core/settings/nli.py` — `NLISettings`, registered in the
  composition root.
- `app/services/verification/nli_result_builder.py` — shared `NLIResult`
  construction, mirroring Units 2.11/2.12's builders.
- `app/services/verification/llm_nli_verifier.py` — `LLMBasedNLIVerifier`,
  implementing `BaseNLIVerifier` exactly (unchanged interface), plus
  `verify_pair_with_result()` and `health()`.
- `app/infrastructure/nli_client_factory.py` — sole place permitted to
  construct the `httpx.AsyncClient` used by `LLMBasedNLIVerifier`.
- **`ContradictionDetector` (Unit 2.9) gained graceful degradation** —
  see below, this is the most consequential change in this unit.

## Key engineering decisions

**1. `confidence` is bucketed as observability, not business — a
deliberate departure from Units 2.9/2.11/2.12, done because the
instruction said so explicitly, not by oversight.** In every prior unit
(`RankedChunk.rerank_score`, `RetrievedChunk.similarity_score`), the
numeric score was part of the business output. Here, per this unit's
explicit instruction 1, only the categorical `label` is business —
`confidence` moved to observability. Documented directly in
`nli.py`'s docstring so this asymmetry reads as intentional, not
inconsistent, to whoever reads the code next.

**2. No new health abstraction — `EmbedderHealth` reused verbatim.**
`LLMBasedNLIVerifier.health()` and `DeterministicNLIVerifier.health()`
both return `app.schemas.embedding.EmbedderHealth`. The type's name is
historically embedding-specific; renaming to something generic like
`ProviderHealth` would be a clean non-breaking future improvement, but
wasn't done here because it would touch Unit 2.11's already-approved
tests for zero functional gain.

**3. Reranking (Unit 2.12) still has no `health()` method.** The
instruction's "maintain consistency across Embedders, Rerankers, and NLI
providers" is read here as guidance for *this unit's* shape (reuse the
same type), not a mandate to retrofit an already-approved unit. Flagged
explicitly as a good candidate for a future non-breaking addition to
`CrossEncoderReranker`, not done in this pass.

**4. The most important change in this unit: `ContradictionDetector`
(Unit 2.9) had zero error handling around `verify_pair()` calls.** This
was invisible until now because `DeterministicNLIVerifier` never fails.
The moment a real, fallible provider exists, any failure would have
propagated uncaught through `VerificationAgent.verify()` and failed the
entire request — exactly the "silently masking" instruction 5 warns
against, except inverted (it would have been a hard crash, not a silent
mask, but still not the graceful degradation asked for). Fixed by
wrapping each pair's `verify_pair()` call individually: a failure
degrades that one pair to `(NEUTRAL, 0.0)` — "no signal obtained," not a
false claim either way — logged as a `WARNING` with both chunk IDs and
the error, and processing continues for the remaining pairs.
`test_partial_failure_only_degrades_the_failing_pair` proves this is
per-pair, not all-or-nothing. The constructor's new `logger` parameter is
optional with a default, so `ContradictionDetector(verifier)` — every
existing call site — still works unchanged.

**5. Zero internal retry in `LLMBasedNLIVerifier`, same reasoning as
Unit 2.12's reranker** — now `ContradictionDetector` degrades any
failure gracefully, so retrying inside the verifier would only delay
that degradation.

## Scope decision: LLM-as-judge, not a dedicated NLI API
Unlike embeddings (OpenAI `/embeddings`) and reranking (Cohere-style
`/rerank`), there's no widely adopted single-vendor REST convention for
pairwise NLI classification as a hosted service. `LLMBasedNLIVerifier`
uses an OpenAI-compatible chat-completions call with a structured JSON
response, which is genuinely how this is done in practice — documented
plainly as a scope decision rather than presented as if a standard existed.

## Network constraint, stated plainly (same as Units 2.11/2.12)
No route to any LLM provider from this sandbox. All 16 `LLMBasedNLIVerifier`
tests validate real request-building, JSON parsing, label-mapping, and
error-translation via `httpx.MockTransport`. The integration test was
executed here only in its skip path (2/2 confirmed).

## Fake→real sequence now complete
This closes out the three model dependencies identified back in the
Retrieval Domain Design: `BaseEmbedder` (Unit 2.11), `BaseReranker`
(Unit 2.12), `BaseNLIVerifier` (this unit) all now have both a
deterministic default and a real, network-tested-via-mock implementation,
with isolated optional integration tests for each.

## Next unit
**2.14 — FastAPI `/query` endpoint.** First unit assembling the full
pipeline (`RetrieverAgent` → `VerificationAgent` → `DecisionEngine`)
behind a thin HTTP layer that only validates input and delegates,
per the frozen layered-architecture rule (API endpoints never contain
business logic).
