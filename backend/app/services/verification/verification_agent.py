"""
Unit 2.9 -- VerificationAgent.

Deliberately small: sequences EvidenceValidator -> ContradictionDetector
-> CoverageAnalyzer -> DiagnosticsBuilder and returns TWO outputs --
VerifiedEvidence (business, feeds DecisionEngine) and
VerificationDiagnostics (observability, feeds telemetry/dashboard/eval) --
per the Unit 2.9 review's explicit business/observability split. Contains
no validation, NLI, coverage, or diagnostics-assembly logic itself.

Boundary note (frozen dependency graph, Retrieval Domain Design Section 5,
reaffirmed in the Unit 2.6 review): VerificationAgent does NOT depend on
RetrieverAgent. It consumes VerificationInput, which the caller builds
from a RetrieverAgent.search() result -- the two agents remain siblings.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from app.core.logging import get_logger, log_event
from app.schemas.retrieval_domain import VerificationInput, VerificationDiagnostics, VerifiedEvidence
from app.services.verification.contradiction_detector import ContradictionDetector
from app.services.verification.coverage_analyzer import CoverageAnalyzer
from app.services.verification.diagnostics_builder import DiagnosticsBuilder
from app.services.verification.evidence_validator import EvidenceValidator


class VerificationAgent:
    def __init__(
        self,
        evidence_validator: EvidenceValidator,
        contradiction_detector: ContradictionDetector,
        coverage_analyzer: CoverageAnalyzer,
        diagnostics_builder: DiagnosticsBuilder,
        logger: Optional[logging.Logger] = None,
    ):
        self._evidence_validator = evidence_validator
        self._contradiction_detector = contradiction_detector
        self._coverage_analyzer = coverage_analyzer
        self._diagnostics_builder = diagnostics_builder
        self._logger = logger or get_logger(__name__)

    async def verify(self, verification_input: VerificationInput) -> tuple[VerifiedEvidence, VerificationDiagnostics]:
        start = time.perf_counter()

        valid_evidence, unsupported_claims = await self._evidence_validator.validate(
            verification_input.ranked_chunks
        )
        nli_results = await self._contradiction_detector.detect(valid_evidence)
        evidence_coverage, reranker_confidence = await self._coverage_analyzer.analyze(
            valid_evidence, total_retrieved=len(verification_input.ranked_chunks)
        )

        latency_ms = (time.perf_counter() - start) * 1000
        diagnostics = self._diagnostics_builder.build(
            query=verification_input.query,
            nli_results=nli_results,
            evidence_coverage=evidence_coverage,
            unsupported_claims=unsupported_claims,
            reranker_confidence=reranker_confidence,
            verification_latency_ms=round(latency_ms, 3),
        )

        evidence = VerifiedEvidence.from_ranked_chunks(
            query=verification_input.query,
            ranked_chunks=valid_evidence,
            nli_results=nli_results,
            retry_count=verification_input.retry_count,
        )

        log_event(
            self._logger, "verification_completed",
            contradiction_detected=diagnostics.contradiction_detected,
            nli_score=diagnostics.nli_score,
            evidence_coverage=diagnostics.evidence_coverage,
            latency_ms=diagnostics.verification_latency_ms,
            retry_count=verification_input.retry_count,
        )
        return evidence, diagnostics
