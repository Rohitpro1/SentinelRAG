# Unit 2.2 — Repository Interfaces + In-Memory Fakes

**Status:** Complete. 18/18 new tests passing, 54/54 total (all prior units unaffected).

## What this unit delivers
- `app/repositories/interfaces.py` — four ABCs exactly as specified in the
  frozen Retrieval Domain Design (Section 4): `VectorRepository`,
  `MetadataRepository`, `CacheRepository`, `FeedbackRepository`. Zero
  infrastructure imports (no `qdrant_client`, `redis`, `asyncpg`) — verified
  by inspection, enforced by the fact that no such packages are installed
  yet and the module still imports cleanly.
- `app/repositories/fakes/in_memory.py` — one in-memory implementation per
  interface, for Unit 2.3+ to build and test against before any real
  infrastructure exists.

## Key implementation decisions
- **All methods are `async def`**, even though the in-memory fakes have no
  actual I/O. This matches what the real implementations (Qdrant client,
  asyncpg, redis.asyncio) will require, so `RetrieverAgent` (Unit 2.6) is
  written once against the async interface and never needs to change when
  fakes are swapped for real backends.
- **`InMemoryVectorRepository.search()` uses cosine similarity**, clamped to
  `[0, 1]` — matches `RetrievedChunk.similarity_score`'s validation
  constraint (Milestone 1 schema, unmodified). Real cosine similarity can be
  negative; the clamp is a deliberate, tested behavior
  (`test_vector_repo_similarity_score_always_bounded`), not an oversight.
- **`MetadataRepository.find_by_fingerprint()`** is new versus a generic
  metadata store — added because the frozen Architecture Enhancements
  (Section 4) specifies fingerprint-based duplicate detection as a concrete
  mechanism, not an aspiration; the interface needs this method for that
  mechanism to be buildable in a later unit.
- **`CacheRepository.invalidate(document_id)`** in the fake tracks which
  `document_id`s each cache entry's results reference (via a reverse index
  built at `set()` time), so invalidation doesn't require the caller to
  know cache-key structure — mirrors how a real implementation would likely
  use a secondary Redis index or a Postgres join table.
- **TTL is accepted but not enforced** in the fake (`set(..., ttl_seconds)`
  stores the parameter's shape but no background eviction runs). This is
  called out explicitly in the code comment — deferred to the real Redis
  implementation (Unit 2.10+), where TTL is a native feature, rather than
  building a fake wall-clock eviction loop that would make tests slower and
  potentially flaky.
- **`FeedbackRepository` has zero callers yet** — correctly so per the
  frozen design (Section 4): it's consumed by the API layer's feedback
  endpoint and the future Human Review Queue, never by `RetrieverAgent`.
  Its tests exist to prove the fake is correct in isolation, ahead of any
  consumer needing it.

## Test coverage
18 tests across all four fakes: round-trip correctness (upsert→search,
save→get, set→get, record→retrieve), interface-contract edge cases
(mismatched upsert lengths raise `ValueError`, missing keys return empty/`None`
rather than raising), and the two behaviors most likely to have subtle bugs —
`document_filter` scoping in vector search, and `invalidate()` correctly
scoping to only the affected cache entries while leaving others untouched.

## New dependency
`pytest-asyncio` (pinned `1.4.0` in `requirements.txt`) — required because
this is the first unit with `async def` code under test. `pytest.ini` sets
`asyncio_mode = auto` so `async def test_...` functions run without needing
`@pytest.mark.asyncio` on every single test.

## Next unit
**2.3 — `RetrievalSettings`**: new domain settings class (same pattern as
Milestone 1's `DecisionEngineSettings`) covering timeouts, retry counts, and
`top_k`/`rerank_top_n` defaults, per the frozen design's Section 1 (Timeout
Behavior) and Section 8 (Performance Targets) numbers.
