# Unit 2.3 — RetrievalSettings

**Status:** Complete. 5/5 new tests passing, 59/59 total.

## What this unit delivers
`app/core/settings/retrieval.py` — `RetrievalSettings`, registered in the
composition root (`app/core/settings/__init__.py`) alongside the four
Milestone 1 domain settings classes, following the identical pattern
established by `DecisionEngineSettings`.

## Key implementation decision
Every default value is a **transcription** of a number already approved in
the frozen Retrieval Domain Design (Section 1: timeouts, retry backoff;
Section 8: `top_k`/`rerank_top_n` defaults) — this unit makes no new
architectural decisions, per the "architecture is locked" instruction.
`test_backoff_schedule_matches_design_example` exists specifically to prove
the settings class reproduces the design doc's worked example (100ms,
300ms) exactly, not just "a" reasonable schedule.

`backoff_schedule_ms()` is a method, not a hardcoded list field, so that
overriding `max_transient_retries` via env var doesn't leave a stale,
wrong-length backoff list behind — the two would otherwise need to be kept
in sync by hand, which is exactly the kind of hidden coupling the project's
DI/SOLID principles are meant to prevent.

## Next unit
**2.4 — `BaseEmbedder` interface + deterministic fake.** First component of
the "fake infrastructure" layer that `RetrieverAgent` (Unit 2.6) will
depend on. The fake must be deterministic (same input text → same output
vector every call) so that Unit 2.6+'s tests are reproducible without a
real embedding model.
