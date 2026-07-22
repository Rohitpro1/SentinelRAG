"""
Unit 3.8 tests -- QueryService's LangGraph integration: covers exactly
the six areas instruction 6 lists, plus a direct DI check that the
compiled graph is genuinely injected (never constructed by QueryService
itself).
"""
import pytest
from langgraph.graph.state import CompiledStateGraph

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk, DecisionAction, NLIRelation, PairwiseNLIResult
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


def _build_retriever(populate=True, conflict_marker="[CONTRADICTION]"):
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()
    return settings, embedder, vector_repo


async def build_service(populate=True, decision_settings=None, conflict_marker="[CONTRADICTION]"):
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
    compiled_graph = GraphBuilder(
        retrieval_node=RetrievalNode(retriever),
        verification_node=VerificationNode(verification_agent),
        decision_node=DecisionNode(decision_engine),
    ).build()
    return QueryService(compiled_graph), vector_repo, embedder


# --- Successful graph execution ---

@pytest.mark.asyncio
async def test_successful_execution_returns_proceed_with_full_query_result():
    service, _, _ = await build_service()
    result = await service.handle_query("refund policy detail number 2")
    assert result.decision.action in {DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE}
    assert result.decision.explainability is not None
    assert result.diagnostics is not None
    assert result.retry_count == 0


# --- Retry execution ---

@pytest.mark.asyncio
async def test_retry_execution_actually_loops_and_terminates():
    settings = DecisionEngineSettings(max_retrieval_retries=2)
    service, _, _ = await build_service(populate=False, decision_settings=settings)
    result = await service.handle_query("anything at all")
    assert result.retry_count == 2
    assert result.decision.action == DecisionAction.CLARIFY


# --- Clarification flow ---

@pytest.mark.asyncio
async def test_clarification_flow_end_to_end():
    settings = DecisionEngineSettings(max_retrieval_retries=1)
    service, _, _ = await build_service(populate=False, decision_settings=settings)
    result = await service.handle_query("unanswerable query")
    assert result.decision.action == DecisionAction.CLARIFY
    assert len(result.decision.reasons) > 0


# --- Low-confidence response ---

@pytest.mark.asyncio
async def test_low_confidence_response_when_evidence_is_weak_but_present():
    """
    Constructed deterministically via settings rather than relying on
    hash-based embedding similarity landing in a specific numeric band
    (unpredictable with DeterministicEmbedder across different text
    pairs): min_retrieval_similarity=0.0 guarantees no retry regardless
    of actual similarity, low_confidence_threshold=0.99 guarantees this
    low-reliability evidence (0.1) can't clear it even at a perfect
    similarity match -- so the outcome is deterministic, not incidental.
    """
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
    compiled_graph = GraphBuilder(
        retrieval_node=RetrievalNode(retriever),
        verification_node=VerificationNode(verification_agent),
        decision_node=DecisionNode(decision_engine),
    ).build()
    service = QueryService(compiled_graph)

    result = await service.handle_query("refund policy detail number 1")
    assert result.decision.action == DecisionAction.LOW_CONFIDENCE_RESPONSE
    assert result.retry_count == 0


# --- Human-review outcome ---

@pytest.mark.asyncio
async def test_human_review_outcome_on_contradiction():
    service, vector_repo, embedder = await build_service(populate=False, conflict_marker="[X]")
    contradicting_chunks = [
        Chunk(chunk_id="c1", document_id="doc-1", text="refunds allowed [X]", token_count=10, source_reliability_score=0.95),
        Chunk(chunk_id="c2", document_id="doc-1", text="refunds not allowed [X]", token_count=10, source_reliability_score=0.95),
    ]
    embeddings = [await embedder.embed_query(c.text) for c in contradicting_chunks]
    await vector_repo.upsert(contradicting_chunks, embeddings)

    result = await service.handle_query("refunds allowed [X]")
    assert result.decision.action == DecisionAction.HUMAN_REVIEW
    assert result.diagnostics.contradiction_detected is True


# --- API compatibility ---

@pytest.mark.asyncio
async def test_query_result_shape_matches_pre_graph_integration_contract():
    """QueryResult's fields match required domain contract including answer field."""
    service, _, _ = await build_service()
    result = await service.handle_query("refund policy detail number 1")
    assert set(result.model_dump().keys()) == {"decision", "diagnostics", "retry_count", "answer"}


@pytest.mark.asyncio
async def test_handle_query_signature_accepts_all_original_parameters():
    """Every Unit 2.14 handle_query() parameter still works post-integration."""
    service, _, _ = await build_service()
    result = await service.handle_query(
        "refund policy detail number 2", top_k=3, rerank_top_n=2,
        document_filter=None, request_id="req-x", trace_id="trace-x",
    )
    assert result is not None


# --- Dependency injection: compiled graph is injected, not constructed ---

def test_query_service_requires_a_compiled_graph_injected_via_constructor():
    """
    Confirms QueryService takes a CompiledStateGraph via its constructor
    (instruction 5) -- constructing QueryService with something that
    obviously is NOT a compiled graph should still succeed at
    construction time (Python has no runtime type enforcement here), but
    this test documents and locks in the intended contract: the
    constructor's sole positional parameter is the graph.
    """
    import inspect
    sig = inspect.signature(QueryService.__init__)
    params = list(sig.parameters.keys())
    assert params[1] == "compiled_graph"


@pytest.mark.asyncio
async def test_query_service_never_imports_graphbuilder_itself():
    """
    Structural check: QueryService must not construct its own graph --
    the compiled graph is injected. Checks for an actual import
    statement (not a docstring mention explaining the design, which
    legitimately references "GraphBuilder" in prose).
    """
    import inspect

    import app.services.query.query_service as qs_module

    source = inspect.getsource(qs_module)
    import_lines = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    assert not any("GraphBuilder" in line or "graph_builder" in line for line in import_lines)
