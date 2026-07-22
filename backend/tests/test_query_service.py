"""
Unit 2.14 tests -- QueryService orchestration.
Unit 3.8 update: `make_query_service()` now builds QueryService from a
COMPILED LANGGRAPH (via GraphBuilder) instead of three raw agents,
matching QueryService's new constructor (Unit 3.8). Every test body
below is UNCHANGED from Unit 2.14 -- same assertions, same behavior
expected -- proving the public contract (handle_query()'s signature and
QueryResult's shape) held steady across the internal rewrite, exactly as
instruction 3 required.
"""
import pytest

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk, DecisionAction
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.query.query_service import QueryService
from app.services.reranking.deterministic import DeterministicReranker
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


async def make_query_service(populate=True, decision_settings=None):
    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    if populate:
        chunks = [
            Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.9)
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
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier()),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )
    decision_engine = DecisionEngine(decision_settings or DecisionEngineSettings())

    compiled_graph = GraphBuilder(
        retrieval_node=RetrievalNode(retriever),
        verification_node=VerificationNode(verification_agent),
        decision_node=DecisionNode(decision_engine),
    ).build()

    return QueryService(compiled_graph)


@pytest.mark.asyncio
async def test_handle_query_returns_query_result_with_decision_and_diagnostics():
    service = await make_query_service()
    result = await service.handle_query("refund policy detail number 2", top_k=5, rerank_top_n=3)
    assert result.decision.action in {DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE}
    assert result.diagnostics.query == "refund policy detail number 2"
    assert result.retry_count == 0


@pytest.mark.asyncio
async def test_handle_query_retries_on_empty_index_then_clarifies():
    """
    An empty vector index -> no chunks retrieved -> DecisionEngine
    returns RETRY_RETRIEVAL up to max_retrieval_retries, then CLARIFY.
    Proves the GRAPH's retry loop (Unit 3.6) actually re-invokes
    RetrievalNode (not just returns RETRY_RETRIEVAL once and stops) and
    terminates via DecisionEngine's own frozen ceiling -- same guarantee
    Unit 2.14's Python while-loop provided, now delivered by the graph.
    """
    settings = DecisionEngineSettings(max_retrieval_retries=2)
    service = await make_query_service(populate=False, decision_settings=settings)
    result = await service.handle_query("anything")
    assert result.decision.action == DecisionAction.CLARIFY
    assert result.retry_count == 2  # retried exactly up to the configured ceiling, then stopped


@pytest.mark.asyncio
async def test_handle_query_does_not_retry_when_confidence_is_sufficient_immediately():
    service = await make_query_service()
    result = await service.handle_query("refund policy detail number 1", top_k=5, rerank_top_n=3)
    assert result.retry_count == 0


@pytest.mark.asyncio
async def test_handle_query_propagates_document_filter():
    service = await make_query_service()
    result = await service.handle_query(
        "refund policy", top_k=5, rerank_top_n=3, document_filter={"document_id": "nonexistent-doc"}
    )
    # Filtered out everything -> no chunks -> eventually CLARIFY, proving
    # the filter actually reached SearchService/VectorRepository THROUGH
    # GraphState.document_filter -> RetrievalNode's SearchRequest (Unit 3.8).
    assert result.decision.action == DecisionAction.CLARIFY


@pytest.mark.asyncio
async def test_handle_query_passes_through_tracing_ids_without_error():
    service = await make_query_service()
    result = await service.handle_query(
        "refund policy detail number 3", top_k=5, rerank_top_n=3, request_id="req-1", trace_id="trace-1"
    )
    assert result.decision is not None
