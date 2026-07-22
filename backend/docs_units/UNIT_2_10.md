# Unit 2.10 — QdrantVectorRepository (First Real Infrastructure)

**Status:** Complete. 15/15 new deterministic unit tests passing (144/144
total, default suite), plus 4 isolated integration tests that correctly
skip without a live Qdrant instance and are excluded from the default run.

## What this unit delivers
- `app/infrastructure/qdrant_client_factory.py` — `create_qdrant_client()`
  / `close_qdrant_client()`. The **only** place permitted to construct an
  `AsyncQdrantClient`.
- `app/repositories/qdrant/vector_repository.py` — `QdrantVectorRepository`,
  implementing `VectorRepository` exactly (same three methods, same
  signatures as `InMemoryVectorRepository`, Unit 2.2).
- `StorageSettings` gained `qdrant_timeout_ms` and `qdrant_vector_size`
  (additive config fields, no architectural change).
- `pytest.ini` gained an `integration` marker, excluded by default
  (`addopts = -m "not integration"`).

## Engineering trade-offs (instruction 8)

**1. Point ID mapping (`chunk_id` → UUID5).** Qdrant's server rejects
arbitrary string point IDs (only unsigned ints or UUIDs are valid), but
`Chunk.chunk_id` is an arbitrary string (frozen Milestone 1 schema).
Rather than constrain the domain schema to satisfy an infrastructure
detail — which would violate "domain must never depend on infrastructure"
— this repository derives a deterministic `uuid5(chunk_id)` for the point
ID and stores the original `chunk_id` in the payload as source of truth.
Verified deterministic and collision-resistant across different IDs by
test.

**2. Retry policy is deliberately asymmetric between `search()` and
`upsert()`/`delete()`.** `search()` has **zero** internal retry — `SearchService`
(Unit 2.6) already wraps the entire call with `asyncio.wait_for` + backoff.
Adding a second retry loop here would multiply actual attempts
unpredictably (outer × inner). `upsert()`/`delete()` get a single bounded
retry (initial attempt + 1), because no `IngestionService` exists yet to
own that responsibility — this is flagged in the code as something to
**remove** once that service is built, not a permanent design choice.
`test_search_does_not_retry_internally` and
`test_upsert_gives_up_after_bounded_retry_exhausted` assert this asymmetry
directly rather than leaving it implicit.

**3. Exception translation defaults unknown errors to `transient=True`.**
Network errors and 5xx responses are transient by inspection; 4xx
responses are not (retrying a malformed request fails identically every
time). For any *other* exception type not explicitly recognized, the
default is transient — reasoning: the cost of one extra bounded retry is
low, the cost of treating a genuinely-recoverable failure as permanent
(and giving up immediately on what might be a passing blip) is higher,
which is exactly the "design for partial failures" instruction (5).

**4. Similarity score is clamped, not rescaled.** Qdrant's COSINE distance
score and `InMemoryVectorRepository`'s cosine similarity (Unit 2.2) are
both clamped into `[0, 1]` rather than rescaled from `[-1, 1]`. This keeps
the two `VectorRepository` implementations numerically comparable —
rescaling would silently redefine what "0.5 similarity" means depending on
which implementation happens to be wired in, which is exactly the kind of
implementation-leaking-through-the-interface bug DI is meant to prevent.

**5. `ensure_collection()` and `close()` are NOT part of `VectorRepository`.**
Both are infra-only lifecycle helpers (collection setup, connection
teardown) — adding them to the domain interface would leak Qdrant-specific
concepts (`vectors_config`, `Distance.COSINE`) into a contract
`InMemoryVectorRepository` also has to satisfy. They exist only on the
concrete `QdrantVectorRepository` class.

## Testing strategy (instruction 6)
- **15 deterministic unit tests** run by default, using a hand-written
  `_FakeAsyncClient` double (not a real connection) to exercise exception
  translation, retry-vs-no-retry behavior, point ID determinism, and
  payload/score conversion — all in well under a second, no network.
- **4 integration tests**, `@pytest.mark.integration`, excluded from the
  default run via `pytest.ini`. Verified in this environment (no live
  Qdrant available) that they **skip** cleanly via a connectivity-check
  fixture — not fail, not error, not silently pass. Anyone with a local
  Qdrant (`docker run -p 6333:6333 qdrant/qdrant`) can run them with
  `pytest -m integration`. Each test uses a uniquely-named, ensure-created,
  then explicitly-dropped collection, so repeated runs don't accumulate
  state and don't collide with a real application's collection.

## Dependency direction verified, not just asserted
`grep -rl "qdrant_client" app/ --include="*.py"` returns exactly two files:
the factory and this repository. Nothing under `app/schemas/`,
`app/services/retrieval/`, `app/services/verification/`, or
`app/services/decision_engine/` mentions Qdrant at all — checked directly
rather than taken on faith.

## Next unit
**2.11 — real embedding model behind `BaseEmbedder`.** Same pattern:
one infrastructure implementation, deterministic-first testing preserved,
dimensional consistency checked against `EmbeddingSettings.dimensions`.
