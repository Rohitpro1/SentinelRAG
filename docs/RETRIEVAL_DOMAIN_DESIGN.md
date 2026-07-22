# SentinelRAG — Retrieval Domain Design

**Status:** Design review artifact. No implementation in this document.
**Depends on:** Milestone 1 (frozen) — `app/schemas/retrieval.py`, `app/core/exceptions.py`,
`app/core/logging.py`, `app/core/settings/*`, `DecisionEngine`.
**Constraint honored:** Milestone 1 is not touched. Everything below is additive,
living in new packages (`app/services/retrieval/`, `app/services/verification/`,
`app/repositories/`), consuming Milestone 1's schemas rather than modifying them.

---

## 1. Retriever Agent Interface

### Responsibilities
- Accept a `SearchRequest` (query text + retrieval parameters).
- Embed the query (via an injected embedder dependency, not owned by the agent).
- Run hybrid search (semantic + keyword) against the `VectorRepository`.
- Rerank candidates via a cross-encoder.
- Return a `SearchResponse` containing `RankedChunk` objects.
- Emit telemetry for every stage it performs (embedding, vector search, rerank).
- **Not** responsible for: deciding whether to retry (that's the Decision Engine,
  consuming this agent's output via the Verification Agent), deciding whether
  retrieval quality is "good enough" (same reason), or persisting results
  (that's the caller's job, via `MetadataRepository`/`FeedbackRepository`).

### Inputs
- `SearchRequest` (Section 3).
- Constructor-injected: `VectorRepository`, an embedder (`BaseEmbedder`, owned by
  the Embedding domain, not the Retrieval domain — see dependency graph), a
  reranker (`BaseReranker`), a `CacheRepository`, a new `RetrievalSettings`
  domain settings class (same pattern as Milestone 1's `DecisionEngineSettings`),
  and a logger.

### Outputs
- `SearchResponse` (Section 3), containing zero or more `RankedChunk`.
- Never raises for "no results found" — that's a valid, expected `SearchResponse`
  with an empty chunk list. It raises only for genuine execution failures
  (Section 6).

### Dependencies
- `VectorRepository` (interface, Section 4) — the only way it touches Qdrant.
- `CacheRepository` (interface, Section 4) — semantic cache lookup before
  paying for embedding + search.
- `BaseEmbedder` — lives in the Embedding domain (Milestone 2 sibling), injected,
  not implemented inside the Retrieval Agent.
- `BaseReranker` — cross-encoder abstraction, same shape as `BaseEmbedder`
  (interface owned here since reranking is a retrieval-quality concern,
  implementation deferred).
- Does **not** depend on `MetadataRepository` directly — chunk metadata needed
  for reliability scoring is expected to already be denormalized onto `Chunk`
  at ingestion time (Milestone 1's `Chunk.source_reliability_score` field), so
  the hot query path avoids an extra join/round-trip.

### Failure Modes
| Failure | Raised as | Recoverable by caller? |
|---|---|---|
| Vector DB unreachable / timeout | `RetrievalError` (transient=True) | Yes — agent-level retry, then propagate |
| Embedder call fails (rate limit, timeout) | `RetrievalError` (transient=True) | Yes — agent-level retry |
| Reranker fails | `RerankError` | No — caught internally, degrades (see below), never escapes |
| Cache backend unreachable | swallowed internally, logged as `cache_unavailable` | N/A — cache is an optimization, never a hard dependency |
| Malformed `SearchRequest` (e.g. empty query) | `RetrievalError` (transient=False) | No — caller error, don't retry |

Reranker failure is deliberately **non-fatal**: if the cross-encoder call fails,
the agent logs the failure, sets `RankedChunk.rerank_score = None` for all
candidates, and returns similarity-ranked results rather than raising. Losing
rerank quality is a confidence-score problem the Decision Engine already
accounts for (via `max_similarity` still being populated); losing the entire
retrieval because a secondary quality step failed is a worse outcome.

### Timeout Behavior
- Per-stage timeouts, not one aggregate timeout, so a slow reranker doesn't
  silently eat the whole budget meant for vector search:
  - Embedding call: 300ms
  - Vector search (Qdrant): 400ms
  - Rerank: 250ms
- Total agent-level budget: 1000ms soft ceiling. If the sum of actual stage
  latencies exceeds this, the agent still returns what it has rather than
  truncating — the ceiling is a telemetry/alerting threshold, not a hard cutoff,
  because a hard cutoff on the query path would silently produce degraded
  results indistinguishable from good ones.

### Retry Policy
- **Agent-level retry** (this section) is about *transient infrastructure*
  failures (a dropped connection, a single timed-out request) — up to 2 retries
  with exponential backoff (100ms, 300ms) on `RetrievalError(transient=True)` only.
  Non-transient errors are never retried.
- This is a **distinct concern** from the Decision Engine's `RETRY_RETRIEVAL`
  action (Milestone 1), which retries with a *rewritten query* because the
  first query's *results* were poor — not because the call failed. The two
  retry loops must not be conflated: agent-level retry lives entirely inside
  `RetrieverAgent.search()` and is invisible to the caller on success;
  Decision-Engine-level retry is a new top-level call to `search()` with a
  different `SearchRequest.query`.

### Telemetry
Emitted per call (full catalog in Section 7): `retrieval_latency_ms`,
`embedding_latency_ms`, `vector_search_latency_ms`, `reranker_latency_ms`,
`cache_hit` (bool), `candidates_returned`, `average_similarity`.

---

## 2. Retrieval Pipeline

```
Document Upload
      |
      v
Embedding                  (Milestone 2, Embedding domain)
      |
      v
Qdrant                     (VectorRepository -- write path)
      |
      v
Hybrid Search               <-- query embedding + keyword index
      |
      v
Cross Encoder                (rerank top-K candidates)
      |
      v
Verification                (NLI contradiction check + entailment scoring)
      |
      v
Decision Engine              (Milestone 1, frozen)
```

**Document Upload -> Embedding:** ingestion-time only, not on the query path.
Each `Chunk` (already produced by `BaseChunker` in Milestone 1) is embedded
once and written to Qdrant with its `chunk_id` as the point ID and `Chunk`
fields as payload. Failure here is an ingestion-pipeline concern, not a
retrieval-domain one — the Retrieval Domain only ever *reads* from Qdrant.

**Qdrant:** the vector store of record. `VectorRepository` is the only
component permitted to hold a Qdrant client. Stores embedding + payload
(`document_id`, `source_reliability_score`, `ocr_confidence`, metadata) so
the query path never needs a metadata join for these fields.

**Hybrid Search:** semantic search (embedding cosine similarity via Qdrant)
combined with keyword/BM25-style search (servable from Qdrant's payload index
or a secondary keyword index — implementation choice deferred to Milestone
2.x) via reciprocal rank fusion or a weighted score combination. Rationale:
pure semantic search misses exact-match terms (IDs, codes, proper nouns)
common in enterprise documents; pure keyword search misses paraphrase.
Output: `list[RetrievedChunk]`, similarity-scored but not yet reranked.

**Cross Encoder:** reranks the top-K (e.g. top 20) `RetrievedChunk`s from
hybrid search using a cross-encoder model (jointly encodes query+chunk,
more accurate than bi-encoder similarity alone, more expensive — hence only
applied to a shortlist, not the full candidate set). Output: `list[RankedChunk]`,
each carrying both `similarity_score` (inherited) and `rerank_score`.

**Verification:** takes the top-N `RankedChunk`s and the query, runs pairwise
NLI between chunks (contradiction detection) and entailment scoring between
each chunk and the candidate answer context. Produces `VerificationOutput`,
which is transformed into Milestone 1's `VerificationReport` — this is the
seam between the new Retrieval Domain and the frozen Decision Engine domain.

**Decision Engine:** unchanged, frozen. Consumes `VerificationReport` exactly
as it does today. The Retrieval Domain's job ends at producing a valid
`VerificationReport`; it has zero knowledge of what the Decision Engine does
with it.

---

## 3. Retriever Contracts (design only — no implementation)

All are Pydantic `BaseModel`s, in a new `app/schemas/retrieval_domain.py`
module (kept separate from Milestone 1's `app/schemas/retrieval.py`, which
stays frozen — this new module *builds on* it by importing `Chunk`,
`RetrievedChunk`, `PairwiseNLIResult`, `VerificationReport`, rather than
editing them).

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20                    # candidates before reranking
    rerank_top_n: int = 5              # candidates kept after reranking
    document_filter: Optional[dict] = None   # e.g. restrict to a document_id set
    retry_count: int = 0                # propagated from Decision Engine retries
    request_id: Optional[str] = None   # tracing
    trace_id: Optional[str] = None


class SearchResponse(BaseModel):
    request: SearchRequest
    ranked_chunks: list[RankedChunk]
    cache_hit: bool
    stage_latencies_ms: dict[str, float]   # {"embedding": .., "vector_search": .., "rerank": ..}


# RetrievedChunk: already defined in Milestone 1 (app/schemas/retrieval.py).
# Reused as-is -- the output of hybrid search, before reranking.


class RankedChunk(BaseModel):
    retrieved_chunk: RetrievedChunk
    rerank_score: Optional[float]      # None if reranker degraded (see failure modes)
    rank: int                          # final position after reranking


class VerificationInput(BaseModel):
    query: str
    ranked_chunks: list[RankedChunk]
    retry_count: int


class VerificationOutput(BaseModel):
    # Superset of what's needed to construct Milestone 1's VerificationReport.
    query: str
    retrieved_chunks: list[RetrievedChunk]   # flattened from ranked_chunks for reuse
    nli_results: list[PairwiseNLIResult]
    retry_count: int

    def to_verification_report(self) -> VerificationReport:
        """Adapter method -- the one allowed seam into the frozen Decision Engine domain."""
```

Note the explicit `to_verification_report()` adapter: this is the single
allowed coupling point between the new Retrieval Domain and Milestone 1's
frozen schema. Everything else in the Retrieval Domain is free to evolve
without risking a Milestone 1 regression, because Milestone 1 only ever
sees a `VerificationReport`, never a `VerificationOutput`.

---

## 4. Repository Interfaces (design only — no implementations)

All four follow the same shape as Milestone 1's `BaseChunker`: an `ABC`,
constructor-injected into whatever service needs it, with a fake/in-memory
implementation for tests and a real implementation wired in later (Milestone 2.10+).

```python
class VectorRepository(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int,
                      document_filter: Optional[dict] = None) -> list[RetrievedChunk]: ...

    @abstractmethod
    async def delete(self, document_id: str) -> None: ...


class MetadataRepository(ABC):
    @abstractmethod
    async def get_document_metadata(self, document_id: str) -> dict: ...

    @abstractmethod
    async def save_document_metadata(self, document_id: str, metadata: dict) -> None: ...

    @abstractmethod
    async def get_ingestion_status(self, document_id: str) -> str: ...


class CacheRepository(ABC):
    @abstractmethod
    async def get(self, cache_key: str) -> Optional[SearchResponse]: ...

    @abstractmethod
    async def set(self, cache_key: str, response: SearchResponse, ttl_seconds: int) -> None: ...

    @abstractmethod
    async def invalidate(self, document_id: str) -> None: ...
    # invalidate() matters: a semantic cache entry referencing a chunk from a
    # document that just got re-ingested/deleted must not be served stale.


class FeedbackRepository(ABC):
    @abstractmethod
    async def record_feedback(self, query_id: str, rating: int, comment: Optional[str] = None) -> None: ...

    @abstractmethod
    async def get_feedback_for_query(self, query_id: str) -> list[dict]: ...
    # Feeds the Human-in-the-Loop Review Queue and Continuous Learning components
    # (architecture layer 4). Deliberately NOT a dependency of RetrieverAgent --
    # feedback is written by the API layer after the fact, never read on the
    # query path, avoiding a write-heavy dependency on the hot path.
```

---

## 5. Dependency Graph

```
                     +-----------------------+
                     |  app/schemas/*        |   <- shared kernel, no
                     |  (Milestone 1 +       |      dependencies on
                     |   retrieval_domain.py)|      anything below
                     +-----------^-----------+
                                 |  (everyone depends downward on schemas)
        +------------------------+-------------------------+
        |                        |                          |
+-------+--------+      +--------+--------+       +---------+--------+
| DecisionEngine  |      | RetrieverAgent  |       | VerificationAgent |
| (Milestone 1,   |      | (Retrieval      |       | (Retrieval        |
|  frozen)        |      |  Domain)        |       |  Domain)          |
+-----------------+      +--------^--------+       +---------^---------+
                                  | constructor-injected       |
                    +-------------+--------------+            |
                    |             |              |             |
           +--------+--+ +--------+--+ +---------+--+  +-------+----+
           |VectorRepo  | |CacheRepo  | |BaseEmbedder | |BaseReranker|
           |(interface) | |(interface)| |(Embedding   | |(interface) |
           |            | |           | | domain)     | |            |
           +------------+ +-----------+ +-------------+ +------------+

     MetadataRepository, FeedbackRepository -- depended on by the API/ingestion
     layer and the Human-Review/Feedback components, NOT by RetrieverAgent.
```

**No cycles, by construction:**
- `DecisionEngine` depends only on `app/schemas` (Milestone 1's
  `VerificationReport`) — it has zero import of anything in the Retrieval
  Domain package. The seam is one-directional: Retrieval Domain produces a
  `VerificationReport` and hands it to `DecisionEngine`; `DecisionEngine`
  never imports `RetrieverAgent`, `VerificationAgent`, or any repository.
- `RetrieverAgent` and `VerificationAgent` depend on repository *interfaces*,
  never on concrete implementations (Dependency Inversion) — so a future
  concrete `QdrantVectorRepository` depends on `VectorRepository` (the ABC),
  and `RetrieverAgent` also depends on `VectorRepository` (the ABC); neither
  depends on the other, avoiding the classic "interface owner imports the
  implementation to type-hint it" cycle.
- Repository interfaces depend on `app/schemas` only (for the types they
  return), never on the services that consume them.

---

## 6. Error Strategy

New exceptions, all subclassing Milestone 1's `SentinelRAGError` (additive to
`app/core/exceptions.py`, not an edit of Milestone 1's existing classes):

```python
class RetrievalError(SentinelRAGError):
    """Already declared as a placeholder in Milestone 1 -- now given real shape."""
    def __init__(self, message, *, transient: bool, context: Optional[dict] = None):
        super().__init__(message, context=context)
        self.transient = transient   # drives agent-level retry eligibility


class VerificationError(SentinelRAGError):
    """Already declared as a placeholder. NLI model failures raise this."""


class EmbeddingError(SentinelRAGError):
    """New -- embedding service call failures. Always treated as transient."""


class RerankError(SentinelRAGError):
    """New -- per Section 1, RetrieverAgent catches this internally and
    degrades gracefully; it should rarely, if ever, escape to the caller."""
```

**Recoverability table:**

| Exception | Recoverable? | Who recovers | How |
|---|---|---|---|
| `RetrievalError(transient=True)` | Yes | `RetrieverAgent` internally | Exponential backoff, max 2 retries |
| `RetrievalError(transient=False)` | No | Caller | Fix the request (e.g. non-empty query) |
| `EmbeddingError` | Yes | `RetrieverAgent` internally | Same retry budget as vector search |
| `RerankError` | Yes (degrades, doesn't fail) | `RetrieverAgent` internally | Fall back to un-reranked results |
| `VerificationError` | Partially | `VerificationAgent`'s caller | If NLI fails entirely, produce a `VerificationOutput` with empty `nli_results` (no contradiction signal) rather than failing the whole query — the Decision Engine already handles a low/absent signal gracefully (lower confidence, not a crash) |

None of these exceptions are ever silently swallowed without a log line —
every catch site logs the original exception (via `log_event(..., level=logging.WARNING)`)
before applying a fallback, so degraded-mode operation is visible in
telemetry even when it doesn't fail the request.

---

## 7. Telemetry Strategy

All metrics are Prometheus-style, labeled where noted, emitted via the
structured logger (Milestone 1's `log_event`) and — in Milestone 2's DevOps
phase — also exported as real Prometheus metrics, not just log lines.

| Metric | Type | Labels | Emitted by | Purpose |
|---|---|---|---|---|
| `retrieval_latency_ms` | Histogram | `cache_hit` | RetrieverAgent | End-to-end search() latency |
| `embedding_latency_ms` | Histogram | — | RetrieverAgent | Query embedding stage only |
| `vector_search_latency_ms` | Histogram | — | RetrieverAgent | Qdrant round-trip only |
| `reranker_latency_ms` | Histogram | `degraded` (bool) | RetrieverAgent | Cross-encoder stage only |
| `cache_hit_rate` | Gauge (derived) | — | Computed from `cache_hit` counter | Semantic cache effectiveness |
| `retrieval_retries` | Counter | `reason` (weak_similarity / no_results) | Decision-Engine-triggered retry loop | How often self-correction actually fires |
| `average_similarity` | Histogram | — | RetrieverAgent | Retrieval quality trend over time |
| `confidence_distribution` | Histogram | `action` (proceed/retry/clarify/...) | DecisionEngine (Milestone 1, additive metric only, no logic change) | Are thresholds well-calibrated? Feeds Milestone 4 calibration |
| `contradiction_rate` | Counter | — | VerificationAgent | How often contradictory sources appear — a headline demo metric |
| `reranker_degraded_total` | Counter | — | RetrieverAgent | Alerting signal: cross-encoder failing repeatedly |
| `human_review_rate` | Counter | — | DecisionEngine (additive) | How often contradictions escalate to a human |

---

## 8. Performance Targets

Engineering targets to design against, not guarantees — to be re-validated
against Milestone 2's eval harness once real models are wired in.

| Stage | p50 target | p95 target | Notes |
|---|---|---|---|
| Query embedding | 50ms | 150ms | Assumes a small/local or fast-API embedding model |
| Qdrant vector search (top-20) | 40ms | 120ms | Local Qdrant instance, collection < 1M points |
| Cross-encoder rerank (20→5) | 80ms | 250ms | CPU inference on a MiniLM-class cross-encoder; GPU would lower this substantially |
| NLI verification (pairwise, ≤5 chunks → ≤10 pairs) | 60ms | 200ms | Same model class as reranker |
| **End-to-end retrieval (embed→search→rerank→verify)** | **~230ms** | **~720ms** | Excludes the Reasoning Agent's LLM call, which dominates total request latency and is out of this domain's scope |

**Throughput:** target ~15–25 requests/sec on a single backend instance with
the above component latencies, assuming the LLM reasoning call (not part of
this domain) is the actual bottleneck in practice — the Retrieval Domain
should not be the limiting factor.

**Memory:** cross-encoder + embedding model loaded in-process, budget
~1.5–2.5GB RSS for MiniLM-class models; Qdrant itself is a separate process/
container, not counted against the API process's memory budget.

**Embedding batch size:** 64 for ingestion-time batch embedding (matches
`EmbeddingSettings.batch_size` from Milestone 1's refactor); query-time
embedding is always batch size 1 (single query, latency-sensitive, not
throughput-sensitive).

**Vector search latency:** see table above; degrades roughly logarithmically
with collection size for HNSW-indexed Qdrant collections, so the 120ms p95
target should hold reasonably well up to several million points without
re-architecture.

---

## 9. Milestone 2 Implementation Plan — Smallest Independently Testable Units

Each unit below must compile and have its own passing tests before the next
unit begins. No unit depends on an unfinished later unit. Real infrastructure
(Qdrant, real embedding/reranker/NLI models) is deliberately pushed to the
back half, so domain logic is fully tested against fakes first — cheaper,
faster, deterministic tests, and a clean seam for swapping fakes for real
implementations later without touching calling code.

1. **2.1 — Schemas.** Add `app/schemas/retrieval_domain.py` (`SearchRequest`,
   `SearchResponse`, `RankedChunk`, `VerificationInput`, `VerificationOutput`,
   including the `to_verification_report()` adapter). Tests: schema
   validation, adapter correctness against Milestone 1's `VerificationReport`.
2. **2.2 — Repository interfaces + in-memory fakes.** `VectorRepository`,
   `MetadataRepository`, `CacheRepository`, `FeedbackRepository` ABCs, each
   with an `InMemory*` fake implementation. Tests: fakes satisfy their own
   interface contracts (round-trip upsert/search, get/set, etc.).
3. **2.3 — `RetrievalSettings`.** New domain settings class (same pattern as
   Milestone 1's `DecisionEngineSettings`) — timeouts, retry counts, top_k
   defaults. Tests: defaults load, env override works.
4. **2.4 — `BaseEmbedder` interface + deterministic fake.** Fake returns a
   hash-based pseudo-embedding (deterministic, no real model) so downstream
   tests are reproducible. Tests: fake embedder is deterministic and
   dimensionally consistent.
5. **2.5 — `BaseReranker` interface + deterministic fake.** Fake rerank
   reorders by a simple deterministic rule so reranking logic is testable
   without a real cross-encoder. Tests: fake reranker contract compliance.
6. **2.6 — `RetrieverAgent` (hybrid search, no rerank yet).** Wire fake
   embedder + fake `VectorRepository` behind `RetrieverAgent.search()`,
   producing `RetrievedChunk`s. Tests: timeout handling, retry-on-transient-error,
   telemetry emission, empty-query rejection.
7. **2.7 — Add reranking to `RetrieverAgent`.** Wire fake reranker in;
   produce `RankedChunk`s. Tests: the reranker-degrades-gracefully-on-failure
   path is explicitly exercised (this is the one that's easy to skip and
   shouldn't be).
8. **2.8 — Semantic cache integration.** Wire `CacheRepository` (in-memory
   fake) into `RetrieverAgent` — cache-key derivation, hit/miss telemetry,
   `invalidate()` on document update. Tests: cache hit skips embed+search+rerank
   entirely; `invalidate()` actually removes stale entries.
9. **2.9 — `VerificationAgent` interface + fake NLI.** `BaseNLIVerifier`
   interface, deterministic fake (flags contradiction via a configurable
   test marker), producing `VerificationOutput`. Tests: `to_verification_report()`
   output feeds Milestone 1's `DecisionEngine.evaluate()` correctly end-to-end
   (the first true cross-domain integration test).
10. **2.10 — Real `QdrantVectorRepository`.** Implements `VectorRepository`
    against a real local Qdrant (docker-compose, dev-only). Tests: integration
    test against a real Qdrant container — upsert then search round-trip.
11. **2.11 — Real embedding model behind `BaseEmbedder`.** Tests: dimensional
    consistency against `EmbeddingSettings.dimensions`, batch-size behavior.
12. **2.12 — Real cross-encoder behind `BaseReranker`.** Tests: latency
    budget check (flags if p95 exceeds the Section 8 target on CI hardware,
    a visible warning rather than a hard test failure).
13. **2.13 — Real NLI model behind `BaseNLIVerifier`.** Tests: known
    entailment/contradiction pairs produce expected relations (a small,
    hand-built regression set).
14. **2.14 — FastAPI `/query` endpoint.** Thin — validates `SearchRequest`
    shape, delegates to `RetrieverAgent` → `VerificationAgent` →
    `DecisionEngine`, returns the `Decision` + `Explainability`. Tests:
    endpoint-level integration test using fakes from 2.1–2.9 (real infra
    from 2.10–2.13 optional, marked `@pytest.mark.integration`, not run in
    the default fast suite).
15. **2.15 — docker-compose for local dev infra.** Qdrant + Postgres + Redis.
    Validated by 2.10's integration test actually passing against it, rather
    than being its own separate test.

Units 1–9 (all fake-backed) can be fully built, reviewed, and tested with
zero new external dependencies beyond what Milestone 1 already has — real
infrastructure risk is entirely isolated to units 10–15, and unit 9 already
proves the cross-domain seam works before any real model is wired in.
