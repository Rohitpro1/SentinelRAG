# Unit 2.4 — BaseEmbedder Interface + Deterministic Fake

**Status:** Complete. 7/7 new tests passing, 66/66 total.

## What this unit delivers
- `app/services/embedding/base.py` — `BaseEmbedder` ABC: `embed_query()`
  (single text, batch size 1, latency-sensitive per Design Section 8),
  `embed_batch()` (ingestion-time batch embedding), and a `dimensions`
  property so callers can validate against `EmbeddingSettings.dimensions`
  without a real model loaded.
- `app/services/embedding/fake.py` — `FakeEmbedder`: SHA-256-seeded,
  fully deterministic, unit-normalized pseudo-embeddings. No semantic
  meaning whatsoever — explicitly documented as test/dev-only.

## Key implementation decision
The fake expands a single 32-byte SHA-256 digest into an arbitrary
`dimensions`-length vector by re-hashing `f"{text}:{counter}"` for
increasing `counter` until enough bytes are available, then normalizes to
unit length. Unit-normalizing matters concretely here: Unit 2.2's
`InMemoryVectorRepository.search()` uses cosine similarity, and cosine
similarity between unit vectors is just the dot product — keeping fake
embeddings unit-normalized means later units (2.6+) exercise the same
numeric regime a real embedding model's output would produce, rather than
arbitrary unnormalized magnitudes that would make similarity thresholds
(`RetrievalSettings`, Milestone 1's `DecisionEngineSettings.min_retrieval_similarity`)
behave unrealistically in tests.

## Test coverage
Determinism (same text twice → identical vector), differentiation
(different texts → different vectors), dimensional consistency against a
configured `dimensions` value, batch/individual-call equivalence (so
`embed_batch` isn't a separate, potentially-diverging code path), and the
unit-normalization invariant explicitly asserted.

## Next unit
**2.5 — `BaseReranker` interface + deterministic fake.** Same pattern:
interface first, then a reproducible fake with a simple deterministic
reordering rule, so `RetrieverAgent`'s reranking integration (Unit 2.7) is
testable without a real cross-encoder.
