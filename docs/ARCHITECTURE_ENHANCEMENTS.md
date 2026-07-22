# SentinelRAG — Architecture Enhancements (Addendum to Retrieval Domain Design)

**Status:** Design only. No implementation in this document. On approval, the
Retrieval Domain design is frozen in its entirety (base design + this
addendum) and Milestone 2 begins per the approved implementation plan.

---

## 1. Event Bus Abstraction

### Why optional, not mandatory
The synchronous query path (embed → search → rerank → verify → decide) stays
direct method calls — an event bus adds latency and failure surface that a
p95-720ms budget (Section 8 of the base design) cannot absorb for free.
Events are for **everything that isn't on that critical path**: metrics,
audit trails, dashboards, notifications, and future continuous-learning
consumers. This matches the instruction precisely: cross-cutting concerns
subscribe to events; the query path does not wait on them.

### Interface

```python
class DomainEvent(BaseModel):
    """Base class for all events. Immutable (frozen) by convention."""
    model_config = ConfigDict(frozen=True)
    event_id: str
    occurred_at: datetime
    trace_id: Optional[str] = None

class EventBus(ABC):
    @abstractmethod
    def publish(self, event: DomainEvent) -> None: ...

    @abstractmethod
    def subscribe(self, event_type: type[DomainEvent], handler: Callable[[DomainEvent], None]) -> None: ...
```

`publish()` is fire-and-forget from the publisher's perspective — a
subscriber raising an exception must never propagate back to the publishing
service (the query path cannot fail because a dashboard consumer errored).
This is an in-process contract requirement on any `EventBus` implementation,
not an implementation detail deferred to later: `publish()` catches and logs
subscriber exceptions internally, always.

### Event catalog

| Event | Published by | Key payload fields | Consumed by (future) |
|---|---|---|---|
| `DocumentUploaded` | Ingestion API | `document_id`, `fingerprint` (Section 4) | Audit log, duplicate-detection consumer |
| `ChunksCreated` | Chunking service | `document_id`, `chunk_count` | Metrics dashboard |
| `EmbeddingsGenerated` | Embedding service | `document_id`, `chunk_count`, `latency_ms` | Metrics dashboard |
| `RetrievalCompleted` | RetrieverAgent | `query_id`, `candidates_returned`, `average_similarity`, `cache_hit` | Metrics dashboard, telemetry sink |
| `VerificationCompleted` | VerificationAgent | `query_id`, `contradiction_detected` (bool) | Metrics dashboard |
| `DecisionMade` | DecisionEngine (additive publish, no logic change) | `query_id`, `action`, `confidence` | Metrics dashboard, Human Review Queue trigger |
| `ResponseGenerated` | Response Generator (Milestone 3) | `query_id`, `latency_ms` | Metrics dashboard |
| `FeedbackReceived` | Feedback API | `query_id`, `rating` | Continuous-learning consumer (future) |
| `DocumentDeleted` | Ingestion API | `document_id` | Cache invalidation consumer, audit log |

### Where the seam lives
`RetrieverAgent`, `VerificationAgent`, and `DecisionEngine` accept an
**optional** `EventBus` dependency (default `None` — a no-op bus, or simply
skip publishing if absent). This preserves Milestone 1's frozen
`DecisionEngine` constructor signature as an *additive* change only if the
team chooses to wire it in; it is never a required dependency, so nothing
that already depends on `DecisionEngine(settings, logger=None)` breaks.

### Relationship to existing structured logging
The Event Bus and `log_event()` (Milestone 1) are not competing mechanisms.
`log_event()` remains for synchronous, request-scoped structured logs
(latency, action, confidence — Section 7 of the base design). Domain events
are for asynchronous, cross-service fan-out. In practice, one `LoggingTelemetrySink`
(Section 3 below) can itself be a subscriber to the event bus, so logging can
be driven by *either* mechanism without duplication — a service emits an
event, and a logging subscriber turns it into a structured log line, rather
than every service calling both `log_event()` and `bus.publish()` for the
same fact.

---

## 2. Retrieval Lifecycle State Machine

### States

| State | Meaning |
|---|---|
| `PENDING` | Query accepted, not yet started |
| `RETRIEVING` | RetrieverAgent in progress |
| `VERIFYING` | VerificationAgent in progress |
| `DECIDING` | DecisionEngine evaluating |
| `GENERATING` | Reasoning Agent / Response Generator in progress (Milestone 3) |
| `COMPLETED` | Response returned to user |
| `RETRYING` | DecisionEngine chose `RETRY_RETRIEVAL` — returns to `RETRIEVING` with a rewritten query |
| `WAITING_FOR_USER` | DecisionEngine chose `CLARIFY` — awaiting user's clarifying input |
| `WAITING_FOR_HUMAN_REVIEW` | DecisionEngine chose `HUMAN_REVIEW` — awaiting reviewer action |
| `FAILED` | Unrecoverable error (non-transient `RetrievalError`, exhausted agent-level retries, etc.) |

### Valid transitions

```
PENDING            -> RETRIEVING
RETRIEVING         -> VERIFYING              (chunks found, agent-level retries exhausted or succeeded)
RETRIEVING         -> FAILED                 (non-transient error, or transient retries exhausted)
VERIFYING          -> DECIDING
VERIFYING          -> FAILED                 (VerificationError with no graceful fallback available)
DECIDING           -> GENERATING             (action = PROCEED)
DECIDING           -> RETRYING               (action = RETRY_RETRIEVAL)
DECIDING           -> WAITING_FOR_USER       (action = CLARIFY)
DECIDING           -> WAITING_FOR_HUMAN_REVIEW (action = HUMAN_REVIEW)
DECIDING           -> COMPLETED              (action = LOW_CONFIDENCE_RESPONSE -- a transparent
                                               low-confidence answer IS a completed response, not a failure)
RETRYING           -> RETRIEVING             (rewritten query, retry_count incremented)
WAITING_FOR_USER   -> RETRIEVING             (user supplies clarification -> new query)
WAITING_FOR_USER   -> FAILED                 (user abandons / session times out)
WAITING_FOR_HUMAN_REVIEW -> DECIDING         (reviewer resolves contradiction, re-evaluate)
WAITING_FOR_HUMAN_REVIEW -> FAILED           (reviewer escalation times out, per SLA policy)
GENERATING         -> COMPLETED
GENERATING         -> FAILED                 (LLM call failure with no fallback)
```

### Invariants
- `COMPLETED` and `FAILED` are terminal — no outgoing transitions.
- Every transition into `RETRYING` must increment `retry_count`; the state
  machine itself is the enforcement point for the `MAX_RETRIEVAL_RETRIES`
  ceiling (Milestone 1's `DecisionEngineSettings`) — once `retry_count`
  reaches the max, `DECIDING` cannot transition to `RETRYING` again, only to
  `CLARIFY`'s state (`WAITING_FOR_USER`) or `FAILED`, matching the Decision
  Engine's existing routing logic exactly (no new business rule invented
  here — the state machine mirrors, not duplicates, that logic).
- This state machine is a **tracking/observability construct** layered over
  the existing synchronous call chain, not a replacement for it — the
  synchronous path still executes as direct calls; the state machine is what
  a `RetrievalCompleted`/`DecisionMade` event-driven dashboard consumer uses
  to render "where is query X right now," particularly useful for
  `WAITING_FOR_USER` / `WAITING_FOR_HUMAN_REVIEW`, which can be long-lived
  (minutes to hours) compared to the sub-second synchronous stages.

---

## 3. `BaseTelemetrySink` Abstraction

### Interface

```python
class BaseTelemetrySink(ABC):
    @abstractmethod
    def record_latency(self, metric_name: str, value_ms: float, labels: dict[str, str]) -> None: ...

    @abstractmethod
    def record_counter(self, metric_name: str, increment: int, labels: dict[str, str]) -> None: ...

    @abstractmethod
    def record_gauge(self, metric_name: str, value: float, labels: dict[str, str]) -> None: ...
```

### Implementations (present and future)
- `LoggingTelemetrySink` — implemented first (wraps Milestone 1's
  `log_event()`); every metric in Section 7 of the base design routes
  through this by default.
- `PrometheusTelemetrySink` (future) — same interface, exports to a
  Prometheus registry instead of/in addition to log lines.
- `OpenTelemetrySink`, `GrafanaTelemetrySink`, `ConsoleTelemetrySink` (future)
  — same interface.

### Why this matters for the checklist's "business logic independent from
telemetry implementation"
`RetrieverAgent`, `VerificationAgent`, and `DecisionEngine` depend on
`BaseTelemetrySink` (constructor-injected, same DI pattern as everything
else in this project), never on a concrete sink. Swapping `LoggingTelemetrySink`
for `PrometheusTelemetrySink` in production is a DI-wiring change in
`main.py`, not a change to any service's code — identical reasoning to why
`BaseChunker` and `VectorRepository` are interfaces.

---

## 4. Content Fingerprinting

- Every uploaded document is hashed with **SHA-256** over its raw bytes at
  upload time, before OCR/parsing/chunking — the fingerprint identifies the
  *source file*, not its derived chunks.
- Stored as `Document.fingerprint` in `MetadataRepository`, alongside
  `document_id`.

### Use cases and where each is enforced
| Use case | Mechanism |
|---|---|
| Duplicate detection | Ingestion API checks `MetadataRepository` for an existing document with the same fingerprint before running the (expensive) OCR/chunk/embed pipeline; short-circuits with a reference to the existing `document_id` |
| Integrity verification | Re-hash on read/audit; mismatch indicates corruption or tampering between upload and later access |
| Auditability | Every `DocumentUploaded` / `DocumentDeleted` event (Section 1) carries the fingerprint, giving an immutable audit trail independent of mutable `document_id`-keyed metadata |
| Cache invalidation | `CacheRepository.invalidate(document_id)` (base design, Section 4) can be keyed additionally by fingerprint so a re-upload of *changed* content (different fingerprint, same logical document) correctly invalidates stale cached search results, while a re-upload of *identical* content (same fingerprint) is recognized as a no-op via duplicate detection above |
| Future document versioning | A logical document with multiple fingerprints over time (v1, v2, ...) is the natural foundation for a version history feature — not built now, but the fingerprint field is what makes it buildable later without a schema migration |

---

## 5. Circuit Breaker Strategy (design only)

Applied uniformly to three external dependencies: **Embedding provider**,
**Vector database (Qdrant)**, **Cross Encoder**. One circuit breaker
instance per dependency, not shared — a failing embedding provider should
not trip the vector-database breaker.

### States
- **Closed** — normal operation, requests pass through, failures counted.
- **Open** — requests short-circuit immediately (no call attempted), a
  fallback/degradation path is used instead (see per-dependency behavior
  below). No calls reach the failing dependency, giving it time to recover.
- **Half-Open** — after a cooldown period, a limited number of trial
  requests are allowed through. If they succeed, transition to Closed. If
  any fail, transition back to Open and reset the cooldown.

### Failure thresholds (per-dependency defaults, all tunable)
| Dependency | Failure threshold to Open | Cooldown before Half-Open | Half-Open trial requests | Fallback while Open |
|---|---|---|---|---|
| Embedding provider | 5 failures in a 30s rolling window | 15s | 1 | Query fails fast as `EmbeddingError` (no embedding = no retrieval possible; this is a hard dependency, no degraded mode) |
| Vector database (Qdrant) | 5 failures in a 30s rolling window | 15s | 1 | Same — `RetrievalError(transient=True)`, hard dependency |
| Cross Encoder | 3 failures in a 15s rolling window (lower threshold — this dependency degrades, not fails, so tripping the breaker earlier is cheap) | 10s | 2 | Already-designed graceful degradation from the base design: return un-reranked results (`rerank_score = None`) — the circuit breaker here prevents *repeatedly retrying* a known-down reranker on every request, not just handling one-off failures |

### Recovery policy
- Half-Open trial failures reset the cooldown timer and re-open the circuit
  rather than immediately retrying — prevents a flapping dependency from
  causing a tight open/half-open/open loop that itself generates load.
- A circuit transitioning Open → Half-Open → Closed emits a
  `CircuitBreakerRecovered` observability event (via `BaseTelemetrySink`,
  Section 3) so recovery is visible in dashboards without requiring someone
  to notice the absence of `CircuitBreakerOpen` alerts.
- Circuit breaker state is process-local, not shared across instances in
  this design — acceptable because each instance independently protecting
  itself from a genuinely-down dependency is the correct behavior; a
  distributed circuit breaker (shared state via Redis) is a future
  enhancement if independent tripping proves too slow to protect the
  dependency at scale, not a Milestone 2 requirement.

### Where it sits architecturally
The circuit breaker wraps calls made by `VectorRepository` implementations,
`BaseEmbedder` implementations, and `BaseReranker` implementations — i.e. it
lives at the infrastructure boundary, not inside `RetrieverAgent`. This
keeps `RetrieverAgent`'s failure-handling logic (Section 1 of the base
design: catch `RetrievalError`, check `.transient`, retry or propagate)
completely unchanged — the circuit breaker changes *how quickly and how
often* the infrastructure layer decides to raise `RetrievalError`, not
*how the agent responds* to that error. This is the same Dependency
Inversion reasoning already established for repositories: the breaker is
an infrastructure-layer concern, invisible to the domain layer above it.

---

## 6. Retrieval Domain Independence — Reaffirmed

Every addition in this addendum is checked against the base design's
dependency graph (Section 5) and confirmed not to introduce a new violation:

- `EventBus` is a domain-layer interface (lives alongside `VectorRepository`
  et al.); concrete implementations (in-memory, Redis pub/sub, Kafka —
  whichever is chosen later) depend on the `EventBus` ABC, never the reverse.
- `BaseTelemetrySink` is the same shape: domain interface, infrastructure
  implementations depend on it.
- The Retrieval Lifecycle State Machine is pure domain logic (an enum +
  transition table) with no infrastructure dependency at all — it doesn't
  even require the Event Bus to exist, though it's most useful when paired
  with one.
- Content fingerprinting is a pure function (bytes → SHA-256 hex digest) —
  no infrastructure dependency; only its *storage* (`MetadataRepository`)
  is infrastructure, and that dependency direction is already established
  in the base design.
- Circuit breakers live entirely inside infrastructure implementations
  (`QdrantVectorRepository`, the real `BaseEmbedder`/`BaseReranker`
  implementations) — the domain-layer interfaces themselves
  (`VectorRepository`, `BaseEmbedder`, `BaseReranker`) are untouched by this
  addendum, so `RetrieverAgent`'s dependency on those interfaces required
  zero changes.

No new package introduces an upward dependency into `app/services/decision_engine`
or `app/schemas`'s existing frozen contents. The dependency graph from the
base design (Section 5) holds unchanged with these additions layered in
purely at the edges (Event Bus subscribers, telemetry sinks, circuit
breakers around infrastructure implementations).

---

## Freeze Statement

With this addendum approved, the **Retrieval Domain design (base design +
this addendum) is frozen**. Implementation now proceeds per the approved
Milestone 2 plan (units 2.1–2.15), starting at unit 2.1. Any further
architectural change to this domain requires either a discovered bug or an
explicit new design-review request — it is not to be revisited opportunistically
during implementation.
