"""
Unit 3.9 tests -- ResponseGenerator service.
"""
import pytest

from app.schemas.retrieval import Chunk, Decision, DecisionAction, RetrievedChunk
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.service import ResponseGenerator


@pytest.mark.asyncio
async def test_proceed_response_generation_with_evidence():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.9, reasons=["Evidence strong"])
    chunk1 = Chunk(chunk_id="c1", document_id="d1", text="Refunds allowed within 30 days.", token_count=6, source_reliability_score=0.95)
    evidence = VerifiedEvidence(query="refund policy", retrieved_chunks=[RetrievedChunk(chunk=chunk1, similarity_score=0.95)], retry_count=0)

    answer = await generator.generate(decision=decision, evidence=evidence, query="refund policy")
    assert "Based on verified evidence:" in answer
    assert "Refunds allowed within 30 days." in answer


@pytest.mark.asyncio
async def test_proceed_response_generation_without_evidence():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.85, reasons=["General knowledge"])

    answer = await generator.generate(decision=decision, query="what is the policy?")
    assert "Based on verified evidence for query 'what is the policy?'" in answer


@pytest.mark.asyncio
async def test_low_confidence_response_generation():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.LOW_CONFIDENCE_RESPONSE, confidence_score=0.4, reasons=["Low similarity"])
    chunk1 = Chunk(chunk_id="c1", document_id="d1", text="Vague policy text", token_count=3, source_reliability_score=0.3)
    evidence = VerifiedEvidence(query="policy", retrieved_chunks=[RetrievedChunk(chunk=chunk1, similarity_score=0.4)], retry_count=0)

    answer = await generator.generate(decision=decision, evidence=evidence, query="policy")
    assert "[Low Confidence Response]" in answer
    assert "Vague policy text" in answer
    assert "Low similarity" in answer


@pytest.mark.asyncio
async def test_clarify_response_generation():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.CLARIFY, confidence_score=0.0, reasons=["No relevant documents found"])

    answer = await generator.generate(decision=decision, query="unknown topic")
    assert "Clarification required for query 'unknown topic':" in answer
    assert "No relevant documents found" in answer


@pytest.mark.asyncio
async def test_human_review_response_generation():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.HUMAN_REVIEW, confidence_score=0.2, reasons=["Contradiction between doc1 and doc2"])
    diagnostics = VerificationDiagnostics(
        query="contradictory query",
        nli_score=0.1,
        contradiction_detected=True,
        evidence_coverage=0.5,
        unsupported_claims=[],
        reranker_confidence=0.5,
        verification_latency_ms=1.0,
    )

    answer = await generator.generate(decision=decision, diagnostics=diagnostics, query="contradictory query")
    assert "Human review required for query 'contradictory query'" in answer
    assert "contradiction detected in retrieved sources" in answer
    assert "Contradiction between doc1 and doc2" in answer


@pytest.mark.asyncio
async def test_retry_retrieval_fallback_response():
    generator = ResponseGenerator()
    decision = Decision(action=DecisionAction.RETRY_RETRIEVAL, confidence_score=0.0, reasons=["Empty index"])

    answer = await generator.generate(decision=decision, query="retry test")
    assert "Retry limit reached for query 'retry test'" in answer
