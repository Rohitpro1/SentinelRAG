"""
Decision Engine — deterministic, threshold-based routing (not LLM
self-rating), now as an injectable OOP service producing full
Explainability trails for the dashboard.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

from app.core.exceptions import DecisionEngineError
from app.core.logging import get_logger, log_event
from app.core.settings.decision_engine import DecisionEngineSettings
from app.schemas.retrieval import (
    ContributingSignal, Decision, DecisionAction, Explainability,
    TriggeredThreshold, VerificationReport,
)


class DecisionEngine:
    def __init__(self, settings: DecisionEngineSettings, logger: Optional[logging.Logger] = None):
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    def compute_confidence(self, report: VerificationReport) -> tuple[float, list[ContributingSignal]]:
        s = self._settings
        similarity_component = report.max_similarity
        contradiction_penalty = report.max_contradiction_confidence

        ocr_scores = [rc.chunk.ocr_confidence for rc in report.retrieved_chunks if rc.chunk.ocr_confidence is not None]
        ocr_component = (sum(ocr_scores) / len(ocr_scores)) if ocr_scores else 1.0

        reliability_scores = [rc.chunk.source_reliability_score for rc in report.retrieved_chunks]
        reliability_component = (sum(reliability_scores) / len(reliability_scores)) if reliability_scores else 0.0

        raw = (
            s.weight_similarity * similarity_component
            + s.weight_ocr_confidence * ocr_component
            + s.weight_source_reliability * reliability_component
            - s.weight_contradiction_penalty * contradiction_penalty
        )
        confidence = max(0.0, min(1.0, raw))

        signals = [
            ContributingSignal(name="max_similarity", value=similarity_component, weight=s.weight_similarity),
            ContributingSignal(name="ocr_confidence", value=ocr_component, weight=s.weight_ocr_confidence),
            ContributingSignal(name="source_reliability", value=reliability_component, weight=s.weight_source_reliability),
            ContributingSignal(name="contradiction_penalty", value=contradiction_penalty, weight=-s.weight_contradiction_penalty),
        ]
        return confidence, signals

    def evaluate(
        self, report: VerificationReport, *,
        request_id: Optional[str] = None, trace_id: Optional[str] = None, query_id: Optional[str] = None,
    ) -> Decision:
        start = time.perf_counter()
        try:
            decision = self._route(report)
        except DecisionEngineError:
            raise
        except Exception as exc:
            raise DecisionEngineError(
                "Decision Engine failed to evaluate verification report",
                context={"query": report.query, "retry_count": report.retry_count},
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        log_event(
            self._logger, "decision_made",
            request_id=request_id, trace_id=trace_id, query_id=query_id,
            latency_ms=round(latency_ms, 3), confidence=decision.confidence_score,
            action=decision.action.value, retry_count=report.retry_count,
        )
        return decision

    def explain(self, decision: Decision) -> Explainability:
        if decision.explainability is None:
            raise DecisionEngineError(
                "Decision has no attached explainability payload", context={"action": decision.action.value}
            )
        return decision.explainability

    def _route(self, report: VerificationReport) -> Decision:
        s = self._settings
        reasons: list[str] = []
        triggered: list[TriggeredThreshold] = []

        if not report.retrieved_chunks:
            triggered.append(TriggeredThreshold(
                name="max_retrieval_retries", threshold_value=s.max_retrieval_retries,
                observed_value=report.retry_count, triggered=report.retry_count >= s.max_retrieval_retries,
            ))
            if report.retry_count < s.max_retrieval_retries:
                reasons.append("No chunks retrieved; retrying with query rewrite.")
                return self._build(DecisionAction.RETRY_RETRIEVAL, 0.0, reasons, triggered, [])
            reasons.append("No chunks retrieved after max retries.")
            return self._build(DecisionAction.CLARIFY, 0.0, reasons, triggered, [])

        contradiction_triggered = report.has_contradiction and report.max_contradiction_confidence >= s.contradiction_threshold
        triggered.append(TriggeredThreshold(
            name="contradiction_threshold", threshold_value=s.contradiction_threshold,
            observed_value=report.max_contradiction_confidence, triggered=contradiction_triggered,
        ))
        if contradiction_triggered:
            confidence, signals = self.compute_confidence(report)
            reasons.append(
                f"Contradiction detected between retrieved chunks "
                f"(confidence={report.max_contradiction_confidence:.2f} >= threshold={s.contradiction_threshold})."
            )
            return self._build(DecisionAction.HUMAN_REVIEW, confidence, reasons, triggered, signals)

        similarity_triggered = report.max_similarity < s.min_retrieval_similarity
        triggered.append(TriggeredThreshold(
            name="min_retrieval_similarity", threshold_value=s.min_retrieval_similarity,
            observed_value=report.max_similarity, triggered=similarity_triggered,
        ))
        if similarity_triggered:
            if report.retry_count < s.max_retrieval_retries:
                reasons.append(
                    f"Max similarity {report.max_similarity:.2f} below threshold "
                    f"{s.min_retrieval_similarity}; retrying retrieval."
                )
                return self._build(DecisionAction.RETRY_RETRIEVAL, 0.0, reasons, triggered, [])
            confidence, signals = self.compute_confidence(report)
            reasons.append(
                f"Max similarity {report.max_similarity:.2f} still below threshold after {report.retry_count} retries."
            )
            return self._build(DecisionAction.CLARIFY, confidence, reasons, triggered, signals)

        confidence, signals = self.compute_confidence(report)
        low_confidence_triggered = confidence < s.low_confidence_threshold
        triggered.append(TriggeredThreshold(
            name="low_confidence_threshold", threshold_value=s.low_confidence_threshold,
            observed_value=confidence, triggered=low_confidence_triggered,
        ))
        if low_confidence_triggered:
            reasons.append(f"Confidence {confidence:.2f} below threshold {s.low_confidence_threshold}.")
            return self._build(DecisionAction.LOW_CONFIDENCE_RESPONSE, confidence, reasons, triggered, signals)

        reasons.append(f"Confidence {confidence:.2f} meets threshold; proceeding to answer generation.")
        return self._build(DecisionAction.PROCEED, confidence, reasons, triggered, signals)

    @staticmethod
    def _build(action, confidence, reasons, triggered, signals) -> Decision:
        explainability = Explainability(
            action=action, confidence=confidence, triggered_thresholds=triggered,
            contributing_signals=signals, human_readable_reasons=reasons,
        )
        return Decision(action=action, confidence_score=confidence, reasons=reasons, explainability=explainability)
