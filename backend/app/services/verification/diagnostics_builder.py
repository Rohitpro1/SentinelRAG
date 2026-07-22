"""
Unit 2.9 -- DiagnosticsBuilder.

Owns: assembling the final VerificationDiagnostics object from the other
three components' outputs, plus deriving nli_score and
contradiction_detected from the raw PairwiseNLIResult list. This is the
only component that knows the final VerificationDiagnostics shape --
everything upstream deals in primitives (floats, lists), not the
assembled schema.
"""
from __future__ import annotations

from app.schemas.retrieval import NLIRelation, PairwiseNLIResult
from app.schemas.retrieval_domain import VerificationDiagnostics


class DiagnosticsBuilder:
    def build(
        self,
        query: str,
        nli_results: list[PairwiseNLIResult],
        evidence_coverage: float,
        unsupported_claims: list[str],
        reranker_confidence: float,
        verification_latency_ms: float,
    ) -> VerificationDiagnostics:
        contradiction_confidences = [
            r.confidence for r in nli_results if r.relation == NLIRelation.CONTRADICTION
        ]
        max_contradiction_confidence = max(contradiction_confidences) if contradiction_confidences else 0.0
        nli_score = max(0.0, min(1.0, 1.0 - max_contradiction_confidence))

        return VerificationDiagnostics(
            query=query,
            nli_score=nli_score,
            contradiction_detected=bool(contradiction_confidences),
            evidence_coverage=evidence_coverage,
            unsupported_claims=unsupported_claims,
            reranker_confidence=reranker_confidence,
            verification_latency_ms=verification_latency_ms,
        )
