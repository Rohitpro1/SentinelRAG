"""
Unit 3.9 tests -- Response Generation & Reasoning Layer end-to-end integration tests.

Covers the 6 exact requirement areas:
 1. Normal answer (PROCEED)
 2. Low confidence answer (LOW_CONFIDENCE_RESPONSE)
 3. Clarification (CLARIFY)
 4. Human review (HUMAN_REVIEW)
 5. Retry completion
 6. API compatibility
"""
import pytest

from app.api.v1.schemas import QueryResponseBody
from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.response_generation import ResponseGenerationNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk, DecisionAction
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.query.query_service import QueryService
from app.services.reranking.deterministic import DeterministicReranker
from app.services.response_generation.service import ResponseGenerator
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.retriever_agent import RetrieverAgent
from app.services.retrieval.search_service import SearchService
from app.services.verification.contradiction_detector import ContradictionDetector
from app.services.verification.coverage_analyzer import CoverageAnalyzer
from app.services.verification.diagnostics_builder import DiagnosticsBuilder
from app.services.verification.evidence_validator import EvidenceValidator
from app.services.verification.nli_deterministic import DeterministicNLIVerifier
from app.services.verification.verification_agent import VerificationAgent


async def build_test_service(populate=True, decision_settings=None, conflict_marker="[CONTRADICTION]"):
    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    if populate:
        chunks = [
            Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.95)
            for i in range(5)
        ]
        embeddings = [await embedder.embed_query(c.text) for c in chunks]
        await vector_repo.upsert(chunks, embeddings)

    retriever = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, retrieval_settings),
        search_service=SearchService(vector_repo, retrieval_settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), retrieval_settings),
        settings=retrieval_settings,
    )
    verification_agent = VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier(conflict_marker=conflict_marker)),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )
    decision_engine = DecisionEngine(decision_settings or DecisionEngineSettings())
    response_generator = ResponseGenerator()

    compiled_graph = GraphBuilder(
        retrieval_node=RetrievalNode(retriever),
        verification_node=VerificationNode(verification_agent),
        decision_node=DecisionNode(decision_engine),
        response_generation_node=ResponseGenerationNode(response_generator),
    ).build()

    return QueryService(compiled_graph), vector_repo, embedder


# --- 1. Normal answer ---

@pytest.mark.asyncio
async def test_normal_answer_generated_on_proceed():
    service, _, _ = await build_test_service(populate=True)
    result = await service.handle_query("refund policy detail number 2")

    assert result.decision.action in {DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE}
    assert result.answer is not None
    assert len(result.answer) > 0
    assert "refund policy detail number 2" in result.answer or "verified evidence" in result.answer.lower()


# --- 2. Low confidence answer ---

@pytest.mark.asyncio
async def test_low_confidence_answer_generated():
    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    weak_chunk = Chunk(
        chunk_id="c1", document_id="doc-1", text="refund policy detail number 1",
        token_count=10, source_reliability_score=0.1,
    )
    await vector_repo.upsert([weak_chunk], [await embedder.embed_query(weak_chunk.text)])

    retriever = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, retrieval_settings),
        search_service=SearchService(vector_repo, retrieval_settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), retrieval_settings),
        settings=retrieval_settings,
    )
    verification_agent = VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier()),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )
    decision_engine = DecisionEngine(
        DecisionEngineSettings(min_retrieval_similarity=0.0, low_confidence_threshold=0.99)
    )
    response_generator = ResponseGenerator()

    compiled_graph = GraphBuilder(
        retrieval_node=RetrievalNode(retriever),
        verification_node=VerificationNode(verification_agent),
        decision_node=DecisionNode(decision_engine),
        response_generation_node=ResponseGenerationNode(response_generator),
    ).build()
    service = QueryService(compiled_graph)

    result = await service.handle_query("refund policy detail number 1")
    assert result.decision.action == DecisionAction.LOW_CONFIDENCE_RESPONSE
    assert result.answer is not None
    assert "[Low Confidence Response]" in result.answer


# --- 3. Clarification ---

@pytest.mark.asyncio
async def test_clarification_answer_generated():
    settings = DecisionEngineSettings(max_retrieval_retries=1)
    service, _, _ = await build_test_service(populate=False, decision_settings=settings)

    result = await service.handle_query("unanswerable query")
    assert result.decision.action == DecisionAction.CLARIFY
    assert result.answer is not None
    assert "Clarification required" in result.answer


# --- 4. Human review ---

@pytest.mark.asyncio
async def test_human_review_answer_generated_on_contradiction():
    service, vector_repo, embedder = await build_test_service(populate=False, conflict_marker="[X]")
    contradicting_chunks = [
        Chunk(chunk_id="c1", document_id="doc-1", text="refunds allowed [X]", token_count=10, source_reliability_score=0.95),
        Chunk(chunk_id="c2", document_id="doc-1", text="refunds not allowed [X]", token_count=10, source_reliability_score=0.95),
    ]
    embeddings = [await embedder.embed_query(c.text) for c in contradicting_chunks]
    await vector_repo.upsert(contradicting_chunks, embeddings)

    result = await service.handle_query("refunds allowed [X]")
    assert result.decision.action == DecisionAction.HUMAN_REVIEW
    assert result.answer is not None
    assert "Human review required" in result.answer


# --- 5. Retry completion ---

@pytest.mark.asyncio
async def test_retry_completion_generates_final_answer():
    settings = DecisionEngineSettings(max_retrieval_retries=2)
    service, _, _ = await build_test_service(populate=False, decision_settings=settings)

    result = await service.handle_query("anything at all")
    assert result.retry_count == 2
    assert result.decision.action == DecisionAction.CLARIFY
    assert result.answer is not None
    assert "Clarification required" in result.answer


# --- 6. API compatibility ---

@pytest.mark.asyncio
async def test_api_compatibility_and_dto_mapping():
    service, _, _ = await build_test_service(populate=True)
    result = await service.handle_query("refund policy detail number 1")

    dto = QueryResponseBody.from_query_result(result)
    assert dto.action == result.decision.action.value
    assert dto.retry_count == result.retry_count
    assert dto.answer == result.answer
    assert dto.answer is not None
