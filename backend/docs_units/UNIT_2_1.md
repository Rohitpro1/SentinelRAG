# Unit 2.1 — Retrieval Domain Schemas

**Status:** Complete. 11/11 new tests passing, 37/37 total (Milestone 1's
26 unaffected).

## What this unit delivers
`app/schemas/retrieval_domain.py`: `SearchRequest`, `RankedChunk`,
`SearchResponse`, `VerificationInput`, `VerificationOutput` — exactly the
five contracts specified in the frozen Retrieval Domain Design, Section 3.
No implementation logic; these are pure data contracts (Pydantic models with
validation only).

## Key implementation decision
`VerificationOutput.to_verification_report()` is the single allowed seam
into Milestone 1's frozen `VerificationReport`. It's implemented as an
explicit method rather than making `VerificationOutput` inherit from or
alias `VerificationReport`, so the two can diverge independently — a future
`VerificationOutput` field (e.g. per-chunk NLI evidence spans) does not
require touching `VerificationReport` or anything that depends on it,
including the frozen `DecisionEngine`.

`VerificationOutput.from_ranked_chunks()` is a convenience constructor
anticipated for `VerificationAgent` (Unit 2.9) — flattens `RankedChunk`
(Retrieval Domain concept) down to `RetrievedChunk` (Milestone 1 concept),
since everything downstream of the Decision Engine boundary only knows
about `RetrievedChunk`.

## Test coverage
- Field validation (empty query rejected, negative retry_count rejected,
  rerank_score bounds enforced, degraded-mode `None` rerank_score allowed).
- Default values match the frozen design (`top_k=20`, `rerank_top_n=5`).
- **Cross-domain integration test**: constructs a `VerificationOutput`,
  adapts it via `.to_verification_report()`, and feeds it directly into a
  real `DecisionEngine.evaluate()` call — confirming the seam works, not
  just that the schema is shaped correctly in isolation.

## Next unit
**2.2 — Repository interfaces + in-memory fakes**: `VectorRepository`,
`MetadataRepository`, `CacheRepository`, `FeedbackRepository` ABCs (frozen
design, Section 4), each with an `InMemory*` fake for testing. No real
Qdrant/Postgres/Redis yet, per the fake-first sequencing in the approved
Milestone 2 plan.
