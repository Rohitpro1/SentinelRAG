"""
Unit 3.9 -- ResponseGenerator service (deterministic-first implementation).

Follows requirement 6: deterministic testing first, no external LLM required.
Generates structured natural-language answers based on DecisionAction:
  - PROCEED: grounded answer built from verified evidence.
  - LOW_CONFIDENCE_RESPONSE: grounded answer marked with a low-confidence caveat.
  - CLARIFY: structured clarification prompt explaining what evidence was missing/unclear.
  - HUMAN_REVIEW: escalation notice indicating conflicting evidence or policy requiring human intervention.
"""
from __future__ import annotations

from typing import Optional

from app.schemas.retrieval import Decision, DecisionAction
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.base import BaseResponseGenerator


class ResponseGenerator(BaseResponseGenerator):
    """
    Deterministic response generator that formats answers based on Decision,
    VerifiedEvidence, and VerificationDiagnostics without requiring external LLMs.
    """

    async def generate(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence] = None,
        diagnostics: Optional[VerificationDiagnostics] = None,
        query: str = "",
    ) -> str:
        action = decision.action

        if action == DecisionAction.PROCEED:
            return self._format_proceed_response(decision, evidence, query)
        elif action == DecisionAction.LOW_CONFIDENCE_RESPONSE:
            return self._format_low_confidence_response(decision, evidence, query)
        elif action == DecisionAction.CLARIFY:
            return self._format_clarify_response(decision, query)
        elif action == DecisionAction.HUMAN_REVIEW:
            return self._format_human_review_response(decision, diagnostics, query)
        elif action == DecisionAction.RETRY_RETRIEVAL:
            # Fallback if invoked with RETRY_RETRIEVAL (e.g. at retry ceiling)
            reasons_str = "; ".join(decision.reasons) if decision.reasons else "Maximum retrieval attempts reached."
            return f"Retry limit reached for query '{query}': {reasons_str}"

        # General fallback
        reasons_str = "; ".join(decision.reasons) if decision.reasons else "No additional details provided."
        return f"Response for query '{query}' (Action: {action.value}): {reasons_str}"

    def _format_proceed_response(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence],
        query: str,
    ) -> str:
        if evidence and evidence.retrieved_chunks:
            chunk_texts = [rc.chunk.text for rc in evidence.retrieved_chunks if rc.chunk and rc.chunk.text]
            if chunk_texts:
                snippets = " | ".join(chunk_texts)
                return f"Based on verified evidence: {snippets}"

        reasons_str = "; ".join(decision.reasons) if decision.reasons else "Confidence threshold satisfied."
        return f"Based on verified evidence for query '{query}': {reasons_str}"

    def _format_low_confidence_response(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence],
        query: str,
    ) -> str:
        base_answer = ""
        if evidence and evidence.retrieved_chunks:
            chunk_texts = [rc.chunk.text for rc in evidence.retrieved_chunks if rc.chunk and rc.chunk.text]
            if chunk_texts:
                base_answer = " | ".join(chunk_texts)

        reasons_str = "; ".join(decision.reasons) if decision.reasons else "Low confidence evidence."
        if base_answer:
            return f"[Low Confidence Response] {base_answer} (Note: {reasons_str})"
        return f"[Low Confidence Response] Limited evidence found for query '{query}'. (Note: {reasons_str})"

    def _format_clarify_response(self, decision: Decision, query: str) -> str:
        reasons_str = "; ".join(decision.reasons) if decision.reasons else "Insufficient evidence to answer query."
        return f"Clarification required for query '{query}': {reasons_str}"

    def _format_human_review_response(
        self,
        decision: Decision,
        diagnostics: Optional[VerificationDiagnostics],
        query: str,
    ) -> str:
        reasons_str = "; ".join(decision.reasons) if decision.reasons else "Contradictory or sensitive evidence detected."
        details = []
        if diagnostics and diagnostics.contradiction_detected:
            details.append("contradiction detected in retrieved sources")
        if details:
            extra = f" ({', '.join(details)})"
        else:
            extra = ""
        return f"Human review required for query '{query}'{extra}: {reasons_str}"
