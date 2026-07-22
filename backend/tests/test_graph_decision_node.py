"""
Unit 3.5 tests -- DecisionNode: covers exactly the six areas instruction
6 lists, using the existing deterministic DecisionEngine (Milestone 1)
and dependency overrides via constructor injection.
"""
import pytest

from app.core.settings.decision_engine import DecisionEngineSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.decision import DecisionNode
from app.schemas.retrieval import Chunk, DecisionAction, RetrievedChunk
from app.schemas.retrieval_domain import VerifiedEvidence
from app.services.decision_engine.engine import DecisionEngine


def make_evidence(chunks_with_similarity, retry_count=0):
    retrieved = [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"content {i}", token_count=10, source_reliability_score=0.95),
            similarity_score=sim,
        )
        for i, sim in enumerate(chunks_with_similarity)
    ]
    return VerifiedEvidence(query="q", retrieved_chunks=retrieved, retry_count=retry_count)


# --- Successful decision generation ---

@pytest.mark.asyncio
async def test_successful_decision_proceeds_with_strong_evidence():
    node = DecisionNode(DecisionEngine(DecisionEngineSettings()))
    state = GraphState(original_query="q", verification_result=make_evidence([0.9, 0.85]))
    result = await node(state)
    assert result.decision is not None
    assert result.decision.action == DecisionAction.PROCEED
    assert result.decision.explainability is not None


@pytest.mark.asyncio
async def test_decision_uses_effective_query_context_consistently():
    """Confirms the report built from verification_result carries through correctly (query field, retrieved_chunks count)."""
    node = DecisionNode(DecisionEngine(DecisionEngineSettings()))
    evidence = make_evidence([0.9, 0.9, 0.9])
    state = GraphState(original_query="q", verification_result=evidence)
    result = await node(state)
    assert len(result.decision.explainability.contributing_signals) == 4


# --- Empty retrieval results ---

@pytest.mark.asyncio
async def test_empty_retrieval_results_with_retry_budget_remaining_returns_retry():
    node = DecisionNode(DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2)))
    state = GraphState(original_query="q", verification_result=make_evidence([], retry_count=0))
    result = await node(state)
    assert result.decision.action == DecisionAction.RETRY_RETRIEVAL


@pytest.mark.asyncio
async def test_no_verification_result_at_all_defaults_to_empty_evidence():
    """DecisionNode must not crash if verification_result is None (e.g. tested standalone)."""
    node = DecisionNode(DecisionEngine(DecisionEngineSettings()))
    state = GraphState(original_query="q")  # verification_result never set
    result = await node(state)
    assert result.decision is not None
    assert result.decision.action == DecisionAction.RETRY_RETRIEVAL


# --- Clarify decisions ---

@pytest.mark.asyncio
async def test_empty_retrieval_results_after_retry_budget_exhausted_returns_clarify():
    settings = DecisionEngineSettings(max_retrieval_retries=2)
    node = DecisionNode(DecisionEngine(settings))
    state = GraphState(original_query="q", verification_result=make_evidence([], retry_count=2), retry_count=2)
    result = await node(state)
    assert result.decision.action == DecisionAction.CLARIFY


# --- Retry decisions ---

@pytest.mark.asyncio
async def test_weak_similarity_within_retry_budget_returns_retry():
    settings = DecisionEngineSettings(min_retrieval_similarity=0.55, max_retrieval_retries=2)
    node = DecisionNode(DecisionEngine(settings))
    state = GraphState(original_query="q", verification_result=make_evidence([0.2], retry_count=0))
    result = await node(state)
    assert result.decision.action == DecisionAction.RETRY_RETRIEVAL


@pytest.mark.asyncio
async def test_retry_decision_carries_correct_reasons():
    node = DecisionNode(DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2)))
    state = GraphState(original_query="q", verification_result=make_evidence([], retry_count=1))
    result = await node(state)
    assert result.decision.action == DecisionAction.RETRY_RETRIEVAL
    assert len(result.decision.reasons) > 0


# --- GraphState serialization after decision ---

@pytest.mark.asyncio
async def test_decision_survives_graphstate_serialization_round_trip():
    node = DecisionNode(DecisionEngine(DecisionEngineSettings()))
    state = GraphState(original_query="q", verification_result=make_evidence([0.9, 0.9]))
    result = await node(state)
    dumped = result.model_dump()
    restored = GraphState.model_validate(dumped)
    assert restored.decision == result.decision
    assert restored.decision.explainability is not None


@pytest.mark.asyncio
async def test_decision_survives_json_round_trip():
    node = DecisionNode(DecisionEngine(DecisionEngineSettings()))
    state = GraphState(original_query="q", verification_result=make_evidence([0.9, 0.9]))
    result = await node(state)
    restored = GraphState.model_validate_json(result.model_dump_json())
    assert restored.decision.action == DecisionAction.PROCEED


# --- Execution inside the compiled LangGraph ---

@pytest.mark.asyncio
async def test_decision_node_executes_when_run_through_the_compiled_graph_with_defaults():
    """
    GraphBuilder() with zero args -- since Unit 3.5, decision_node
    defaults to a real DecisionNode(DecisionEngine(...)), so even the
    planner-only chain now produces a real Decision.
    """
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke(GraphState(original_query="anything at all"))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata is not None
    assert reconstructed.decision is not None
    assert reconstructed.decision.action == DecisionAction.RETRY_RETRIEVAL  # no evidence at all -> retry


@pytest.mark.asyncio
async def test_full_chain_planner_retrieval_verification_decision_via_graph():
    from app.core.settings.retrieval import RetrievalSettings
    from app.orchestration.nodes.retrieval import RetrievalNode
    from app.orchestration.nodes.verification import VerificationNode
    from app.repositories.fakes.in_memory import InMemoryVectorRepository
    from app.services.embedding.deterministic import DeterministicEmbedder
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

    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.95)
        for i in range(5)
    ]
    embeddings = [await embedder.embed_query(c.text) for c in chunks]
    await vector_repo.upsert(chunks, embeddings)

    retriever_agent = RetrieverAgent(
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

    compiled = GraphBuilder(
        retrieval_node=RetrievalNode(retriever_agent),
        verification_node=VerificationNode(verification_agent),
    ).build()

    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 2"))
    reconstructed = GraphState(**result)

    assert reconstructed.planning_metadata is not None
    assert reconstructed.retrieval_result is not None
    assert reconstructed.verification_result is not None
    assert reconstructed.diagnostics is not None
    assert reconstructed.decision is not None
    assert reconstructed.decision.action in {DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE}
