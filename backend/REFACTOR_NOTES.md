# Architectural Refactor — Pre-Milestone-2

Applied on top of the approved Milestone 1 slice. No functional behavior
changed — same routing rules, same confidence formula, same chunking
algorithm. Verified by re-running the full test suite (26/26 passing)
after the refactor, using tests rewritten against the new interfaces.

## What changed and why

**1. DecisionEngine as a class.** `compute_confidence()`, `evaluate()`,
`explain()` are now methods on `DecisionEngine(settings, logger=None)`
instead of free functions. `evaluate()` wraps the pure routing logic with
structured logging and exception translation; `explain()` is a separate
method so a persisted past `Decision` can be re-rendered without re-running
routing.

**2. Explainability object.** Every `Decision` now carries an
`explainability` field: `action`, `confidence`, `triggered_thresholds`
(each threshold checked, its value, whether it fired), `contributing_signals`
(each signal that fed the confidence score, with its weight), and
`human_readable_reasons`. This is the exact shape the frontend dashboard
should render — no re-derivation needed on the frontend side.

**3. Chunker interface.** `BaseChunker` (abstract) → `SentenceChunker`
(implemented, same logic as Milestone 1) → `TableChunker` / `SemanticChunker`
(both raise `NotImplementedError` — architecture only, no logic, per the
explicit instruction not to implement them yet). Callers should depend on
`BaseChunker`, not `SentenceChunker`, so adding a real `TableChunker` later
is a DI-wiring change, not a call-site change.

**4. Domain-specific settings.** `DecisionEngineSettings`, `ChunkingSettings`,
`StorageSettings`, `SecuritySettings`, `EmbeddingSettings` — each its own
class, own env prefix, own file. `AppSettings` composes them for bootstrap
only; no service should import `AppSettings` directly, only the one domain
settings class it needs (constructor injection, see `DecisionEngine.__init__`
and `SentenceChunker.__init__`).

**5. Structured JSON logging.** `app/core/logging.py` — one JSON object per
log line, verified above to carry `request_id`, `trace_id`, `query_id`,
`latency_ms`, `confidence`, `action`, `retry_count`. `configure_logging()`
called once at process start; `log_event()` is the call-site convenience
wrapper.

**6. Typed exceptions.** `SentinelRAGError` root + `DecisionEngineError`,
`ChunkingError`, `RetrievalError`, `VerificationError`, `SecurityError`.
`DecisionEngine.evaluate()` and `SentenceChunker.chunk()` both translate
unexpected internal failures into their subsystem's typed exception at
the service boundary, rather than leaking raw exceptions.

**7-8. Layering / SOLID / DI.** No global state anywhere in `app/services/`
or `app/core/`. Every service takes its dependencies via `__init__`
(settings, optionally a logger). `BaseChunker` is the Dependency Inversion
seam for chunker strategy; `DecisionEngineSettings` being separate from
e.g. `StorageSettings` is the Single Responsibility split at the config
layer. No API layer exists yet to enforce the "endpoints only validate +
delegate" rule against — that becomes real in Milestone 2 once FastAPI
routes exist.

## Known follow-ups (not silently deferred)
- `TableChunker` / `SemanticChunker` are stubs by design (explicitly requested
  not to implement yet) — calling `.chunk()` on either raises `NotImplementedError`,
  it does not silently no-op.
- No API layer exists yet to demonstrate the "controllers only validate and
  delegate" rule in practice — first real test of that comes in Milestone 2.
- Confidence weights are still the Milestone-1 provisional values; calibration
  is still owed to Milestone 4, unchanged by this refactor.
