# Unit 2.9 — VerificationAgent (Business/Observability Split)

**Status:** Complete. 33/33 new tests passing, 129/129 total.

## What this unit delivers
Per the Unit 2.9 review's refinement instructions:

- **Two schema types**, both in `app/schemas/retrieval_domain.py`:
  - `VerifiedEvidence` (business output) — a subclass of Unit 2.1's
    `VerificationOutput`, inheriting `.to_verification_report()` and
    `.from_ranked_chunks()` unchanged. Consumed by `DecisionEngine` today;
    designed to also be consumed by `ReasoningAgent`/`ResponseGenerator`
    once they exist (Milestone 3).
  - `VerificationDiagnostics` (observability output) — `nli_score`,
    `contradiction_detected`, `evidence_coverage`, `unsupported_claims`,
    `reranker_confidence`, `verification_latency_ms`. Never touched by
    `DecisionEngine`; exists for telemetry/dashboard/evaluation/analytics.
- **Four focused components**, each independently tested:
  - `EvidenceValidator` — structural validation (non-empty text, positive
    token count), producing `(valid_evidence, unsupported_claims)`.
  - `ContradictionDetector` — pairwise NLI over valid evidence via an
    injected `BaseNLIVerifier`, producing `list[PairwiseNLIResult]`.
  - `CoverageAnalyzer` — `evidence_coverage` and `reranker_confidence`
    as two independent numeric signals.
  - `DiagnosticsBuilder` — the only component that assembles the final
    `VerificationDiagnostics`, deriving `nli_score` and
    `contradiction_detected` from the raw NLI results.
- **`BaseNLIVerifier` + `DeterministicNLIVerifier`** — deterministic-first,
  named directly (no `Fake*` prefix) per the Unit 2.6 review's naming
  convention. Contradiction is triggered only by an explicit configurable
  marker string present in both texts — fully deterministic, no
  heuristic guessing at semantic conflict.
- **`VerificationAgent`** — 81 lines, sequences the four components and
  returns `tuple[VerifiedEvidence, VerificationDiagnostics]`. Contains no
  validation/NLI/coverage/diagnostics-assembly logic of its own.

## Key implementation decisions
- **`VerifiedEvidence` is a subclass, not a duplicate schema.** Reusing
  `VerificationOutput`'s already-tested adapter logic via inheritance
  (rather than re-deriving `to_verification_report()` from scratch) avoids
  a second, parallel implementation of the one seam into frozen Milestone 1
  code — there is exactly one adapter, and `VerifiedEvidence` inherits it.
- **`evidence_coverage` redefined from the original Milestone-2-plan
  sketch.** Rather than needing `rerank_top_n` (which `VerificationInput`,
  Unit 2.1, does not carry), coverage is defined as *fraction of retrieved
  evidence that survived structural validation* — answerable entirely from
  data `VerificationAgent` already has, without extending a Unit 2.1 schema
  that's already tested and in use.
- **`unsupported_claims` is a structural proxy today, and the code says so.**
  True claim-level verification (does evidence support a specific sentence
  in a draft answer) needs a draft answer from `ReasoningAgent`/
  `ResponseGenerator`, which don't exist yet. Flagging chunk_ids that fail
  structural checks is a real, useful, tested signal in the meantime — not
  a placeholder pretending to be the final feature.

## Boundary preserved
`VerificationAgent` takes a `VerificationInput` built by its caller from a
`RetrieverAgent.search()` result — it never imports or depends on
`RetrieverAgent`, `SearchService`, or anything else in the retrieval
package. `test_full_cross_domain_pipeline_retriever_to_verification_to_decision_engine`
proves this wiring works end-to-end with real components at every stage
(deterministic embedder → in-memory vector repo → deterministic reranker →
deterministic NLI verifier → real `DecisionEngine`), not just that the
types line up on paper.

## Next unit
**2.10 — Real `QdrantVectorRepository`.** First unit touching actual
external infrastructure; per the approved plan, validated via an
integration test against a real local Qdrant container (docker-compose),
separate from the fast fake-backed suite above.
