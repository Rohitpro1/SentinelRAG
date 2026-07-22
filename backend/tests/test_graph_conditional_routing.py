"""
Unit 3.6 tests -- conditional routing: covers exactly the eight areas
instruction 7 lists. Uses deterministic components throughout, no real
network, and exercises the ACTUAL frozen DecisionAction enum (see
app/orchestration/routing.py's naming-reconciliation docstring) rather
than the instruction's conceptual ANSWER/FAIL terms, which don't exist
in the frozen enum.
"""
import pytest

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.retry_increment import RetryIncrementNode
from app.orchestration.routing import route_after_decision
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk, Decision, DecisionAction
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.reranking.deterministic import DeterministicReranker
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.retriever_agent import RetrieverAgent
from app.services.retrieval.search_service import SearchService


def make_state_with_decision(action: DecisionAction) -> GraphState:
    decision = Decision(action=action, confidence_score=0.5, reasons=["test"])
    return GraphState(original_query="q", decision=decision)


async def make_populated_retrieval_node() -> RetrievalNode:
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.95)
        for i in range(5)
    ]
    embeddings = [await embedder.embed_query(c.text) for c in chunks]
    await vector_repo.upsert(chunks, embeddings)
    agent = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(vector_repo, settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
    )
    return RetrievalNode(agent)


def make_empty_retrieval_node() -> RetrievalNode:
    """Empty index -- forces RETRY_RETRIEVAL, then CLARIFY once the retry ceiling is hit."""
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    agent = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(InMemoryVectorRepository(), settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
    )
    return RetrievalNode(agent)


# --- Unit-level routing function tests (no graph needed) ---

def test_route_returns_retry_for_retry_retrieval_action():
    assert route_after_decision(make_state_with_decision(DecisionAction.RETRY_RETRIEVAL)) == "retry"


@pytest.mark.parametrize(
    "action", [DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE, DecisionAction.CLARIFY, DecisionAction.HUMAN_REVIEW]
)
def test_route_returns_end_for_every_non_retry_action(action):
    assert route_after_decision(make_state_with_decision(action)) == "end"


def test_route_returns_end_when_decision_is_none():
    assert route_after_decision(GraphState(original_query="q")) == "end"


# --- RetryIncrementNode ---

@pytest.mark.asyncio
async def test_retry_increment_node_increments_by_exactly_one():
    node = RetryIncrementNode()
    result = await node(GraphState(original_query="q", retry_count=3))
    assert result.retry_count == 4


@pytest.mark.asyncio
async def test_retry_increment_node_from_zero():
    node = RetryIncrementNode()
    result = await node(GraphState(original_query="q"))
    assert result.retry_count == 1


# --- Answer path (PROCEED / LOW_CONFIDENCE_RESPONSE -> END) ---

@pytest.mark.asyncio
async def test_answer_path_proceed_terminates_graph_without_retry():
    """
    DecisionNode (Unit 3.5) only ever reads state.verification_result --
    not state.retrieval_result directly -- so a realistic PROCEED outcome
    requires VerificationNode in the chain too (matching how the full
    pipeline is actually meant to be wired; DecisionNode with no
    verification_result correctly sees "no evidence" regardless of what
    retrieval found, per its own defensive-fallback design).
    """
    from app.orchestration.nodes.verification import VerificationNode
    from app.services.verification.contradiction_detector import ContradictionDetector
    from app.services.verification.coverage_analyzer import CoverageAnalyzer
    from app.services.verification.diagnostics_builder import DiagnosticsBuilder
    from app.services.verification.evidence_validator import EvidenceValidator
    from app.services.verification.nli_deterministic import DeterministicNLIVerifier
    from app.services.verification.verification_agent import VerificationAgent

    retrieval_node = await make_populated_retrieval_node()
    verification_agent = VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier()),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )
    compiled = GraphBuilder(
        retrieval_node=retrieval_node, verification_node=VerificationNode(verification_agent)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 2"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision.action in {DecisionAction.PROCEED, DecisionAction.LOW_CONFIDENCE_RESPONSE}
    assert reconstructed.retry_count == 0  # never entered the retry loop


# --- Clarify path ---

@pytest.mark.asyncio
async def test_clarify_path_terminates_graph_after_retry_ceiling():
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="anything"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision.action == DecisionAction.CLARIFY
    assert reconstructed.retry_count == 2


# --- Fail path (HUMAN_REVIEW, the closest actual analog -- see routing.py) ---

@pytest.mark.asyncio
async def test_human_review_action_terminates_graph_without_retry():
    """
    HUMAN_REVIEW is the frozen enum's closest analog to the instruction's
    conceptual FAIL outcome (see routing.py's reconciliation docstring).
    Directly exercises the routing function with this action inside a
    real compiled graph (not just the unit-level route_after_decision
    test above) to prove it terminates the graph, not just returns "end"
    in isolation.
    """
    from app.orchestration.nodes.decision import DecisionNode as DN

    class _AlwaysHumanReviewDecisionEngine:
        def evaluate(self, report, **kwargs):
            return Decision(action=DecisionAction.HUMAN_REVIEW, confidence_score=0.5, reasons=["contradiction found"])

    compiled = GraphBuilder(
        retrieval_node=await make_populated_retrieval_node(),
        decision_node=DN(_AlwaysHumanReviewDecisionEngine()),
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 1"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision.action == DecisionAction.HUMAN_REVIEW
    assert reconstructed.retry_count == 0


# --- Retry path ---

@pytest.mark.asyncio
async def test_retry_path_loops_back_to_retrieval_and_eventually_resolves():
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=1))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="anything"))
    reconstructed = GraphState(**result)
    # max_retrieval_retries=1 -> one RETRY_RETRIEVAL, then CLARIFY
    assert reconstructed.retry_count == 1
    assert reconstructed.decision.action == DecisionAction.CLARIFY


@pytest.mark.asyncio
async def test_retrieval_node_actually_re_invoked_on_retry_not_just_counted():
    """Confirms the SearchRequest sent on the final attempt carries the incremented retry_count -- proving RetrievalNode really re-ran, not just that retry_count changed."""
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="q"))
    reconstructed = GraphState(**result)
    assert reconstructed.retrieval_result.request.retry_count == 2


# --- Retry ceiling respected ---

@pytest.mark.asyncio
async def test_retry_ceiling_is_respected_exactly_not_off_by_one():
    for max_retries in (0, 1, 3):
        decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=max_retries))
        compiled = GraphBuilder(
            retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
        ).build()
        result = await compiled.ainvoke(GraphState(original_query="q"))
        reconstructed = GraphState(**result)
        assert reconstructed.retry_count == max_retries
        assert reconstructed.decision.action == DecisionAction.CLARIFY


# --- Graph termination ---

@pytest.mark.asyncio
async def test_graph_terminates_and_does_not_hang_or_hit_recursion_limit():
    """A generous max_retrieval_retries still terminates within LangGraph's default recursion limit."""
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=5))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="q"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision.action == DecisionAction.CLARIFY
    assert reconstructed.retry_count == 5


# --- retry_count updates ---

@pytest.mark.asyncio
async def test_retry_count_starts_at_zero_and_increments_exactly_once_per_loop_pass():
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=3))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="q", retry_count=0))
    reconstructed = GraphState(**result)
    assert reconstructed.retry_count == 3  # exactly 3 increments, not 2 or 4


# --- Deterministic execution ---

@pytest.mark.asyncio
async def test_repeated_runs_produce_identical_outcomes():
    decision_engine_settings = DecisionEngineSettings(max_retrieval_retries=2)

    async def run_once():
        compiled = GraphBuilder(
            retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(DecisionEngine(decision_engine_settings))
        ).build()
        result = await compiled.ainvoke(GraphState(original_query="deterministic test query"))
        return GraphState(**result)

    first = await run_once()
    second = await run_once()
    assert first.decision.action == second.decision.action
    assert first.retry_count == second.retry_count
    assert first.decision.confidence_score == second.decision.confidence_score


# --- No retrieval node injected: retry loop unreachable, straight-line preserved ---

@pytest.mark.asyncio
async def test_without_retrieval_node_decision_still_goes_straight_to_end():
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke(GraphState(original_query="q"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision is not None
    assert reconstructed.retry_count == 0
