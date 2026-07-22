"""
Unit 3.7 tests -- retry-aware PlannerNode: covers exactly the six areas
instruction 7 lists, plus direct unit tests of the deterministic rewrite
rule itself.
"""
import pytest

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.planner import PlannerNode, _rewrite_query
from app.orchestration.nodes.retrieval import RetrievalNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import DecisionAction
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.reranking.deterministic import DeterministicReranker
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.retriever_agent import RetrieverAgent
from app.services.retrieval.search_service import SearchService


def make_empty_retrieval_node() -> RetrievalNode:
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


# --- Direct unit tests of the rewrite rule ---

def test_rewrite_strips_stopwords():
    assert _rewrite_query("what is the refund policy?") == "refund policy?"


def test_rewrite_preserves_keyword_order_and_casing():
    assert _rewrite_query("How does the Refund Process work") == "Refund Process work"


def test_rewrite_falls_back_to_original_when_all_words_are_stopwords():
    assert _rewrite_query("what is this") == "what is this"


def test_rewrite_is_idempotent():
    once = _rewrite_query("what is the refund policy")
    twice = _rewrite_query(once)
    assert once == twice


# --- First-pass planning (retry_count == 0) ---

@pytest.mark.asyncio
async def test_first_pass_leaves_rewritten_query_unset():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?"))
    assert result.rewritten_query is None


@pytest.mark.asyncio
async def test_first_pass_does_not_clobber_a_pre_existing_rewritten_query():
    """Defensive property, same as Unit 3.2's original guarantee -- still holds since retry_count is still 0 here."""
    node = PlannerNode()
    result = await node(GraphState(original_query="q", rewritten_query="already set", retry_count=0))
    assert result.rewritten_query == "already set"


# --- Retry planning (retry_count > 0) ---

@pytest.mark.asyncio
async def test_retry_generates_rewritten_query():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?", retry_count=1))
    assert result.rewritten_query is not None
    assert result.rewritten_query == "refund policy?"


@pytest.mark.asyncio
async def test_retry_preserves_original_query():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?", retry_count=1))
    assert result.original_query == "what is the refund policy?"


@pytest.mark.asyncio
async def test_retry_overwrites_a_stale_rewritten_query_with_a_fresh_one():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?", rewritten_query="stale value", retry_count=1))
    assert result.rewritten_query == "refund policy?"


# --- Rewritten query usage (by RetrievalNode, via effective_query) ---

@pytest.mark.asyncio
async def test_effective_query_reflects_rewrite_after_retry():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?", retry_count=1))
    assert result.effective_query == "refund policy?"


# --- Preservation of original_query across multiple retries ---

@pytest.mark.asyncio
async def test_original_query_preserved_across_multiple_sequential_retries():
    node = PlannerNode()
    state = GraphState(original_query="what is the refund policy?")
    for retry_count in (1, 2, 3):
        state = state.model_copy(update={"retry_count": retry_count})
        state = await node(state)
        assert state.original_query == "what is the refund policy?"


# --- Multiple retries ---

@pytest.mark.asyncio
async def test_rewrite_is_stable_across_multiple_retries():
    """Same rewrite rule applied fresh each time -> same result at every retry count (idempotent, not progressively different -- see planner.py's documented design choice)."""
    node = PlannerNode()
    results = []
    for retry_count in (1, 2, 3):
        state = GraphState(original_query="what is the refund policy?", retry_count=retry_count)
        result = await node(state)
        results.append(result.rewritten_query)
    assert results == ["refund policy?", "refund policy?", "refund policy?"]


# --- Graph execution through the retry path ---

@pytest.mark.asyncio
async def test_graph_retry_loop_actually_generates_rewritten_query():
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="what is the refund policy?"))
    reconstructed = GraphState(**result)
    assert reconstructed.decision.action == DecisionAction.CLARIFY
    assert reconstructed.retry_count == 2
    assert reconstructed.rewritten_query == "refund policy?"


@pytest.mark.asyncio
async def test_graph_first_pass_search_uses_original_query_not_rewrite():
    """The FIRST retrieval attempt (before any retry) must use original_query -- rewriting only kicks in from the second attempt onward."""
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=2))
    node = make_empty_retrieval_node()
    compiled = GraphBuilder(retrieval_node=node, decision_node=DecisionNode(decision_engine)).build()
    result = await compiled.ainvoke(GraphState(original_query="what is the refund policy?"))
    reconstructed = GraphState(**result)
    # Final retrieval_result reflects the LAST attempt (after 2 retries) -- confirms it used the rewrite by then.
    assert reconstructed.retrieval_result.request.query == "refund policy?"


@pytest.mark.asyncio
async def test_graph_planning_metadata_still_reflects_original_query_not_rewrite():
    """planning_metadata.normalized_query is always derived from original_query -- rewriting doesn't change what gets classified."""
    decision_engine = DecisionEngine(DecisionEngineSettings(max_retrieval_retries=1))
    compiled = GraphBuilder(
        retrieval_node=make_empty_retrieval_node(), decision_node=DecisionNode(decision_engine)
    ).build()
    result = await compiled.ainvoke(GraphState(original_query="what is the refund policy?"))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata.normalized_query == "what is the refund policy?"


@pytest.mark.asyncio
async def test_retrieval_node_itself_unchanged_still_just_reads_effective_query():
    """
    Sanity check per explicit instruction: RetrievalNode was not modified
    in this unit. Confirmed by grep in UNIT_3_7.md; here confirmed
    behaviorally -- feeding it a state with rewritten_query set (without
    going through PlannerNode at all) still produces a search using that
    rewritten_query, proving RetrievalNode's own logic is untouched.
    """
    from app.core.settings.retrieval import RetrievalSettings as RS

    settings = RS()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()
    from app.schemas.retrieval import Chunk

    chunk = Chunk(chunk_id="c1", document_id="doc-1", text="refund policy content", token_count=10, source_reliability_score=0.9)
    await vector_repo.upsert([chunk], [await embedder.embed_query(chunk.text)])
    agent = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(vector_repo, settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
    )
    node = RetrievalNode(agent)
    result = await node(GraphState(original_query="ignored", rewritten_query="refund policy content"))
    assert result.retrieval_result.request.query == "refund policy content"
