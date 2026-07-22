"""
Unit 3.3 tests -- RetrievalNode: covers exactly the six areas instruction
6 lists, using deterministic RetrieverAgent implementations (Unit 2.6),
dependency overrides via constructor injection, and no real network.
"""
import pytest

from app.core.exceptions import RetrievalError
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.retrieval import RetrievalNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.reranking.deterministic import DeterministicReranker
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.retriever_agent import RetrieverAgent
from app.services.retrieval.search_service import SearchService


async def make_populated_agent() -> RetrieverAgent:
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.9)
        for i in range(5)
    ]
    embeddings = [await embedder.embed_query(c.text) for c in chunks]
    await vector_repo.upsert(chunks, embeddings)

    return RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(vector_repo, settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
    )


def make_empty_agent() -> RetrieverAgent:
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    return RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(InMemoryVectorRepository(), settings),  # never populated
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
    )


class _AlwaysFailingRetrieverAgent(RetrieverAgent):
    def __init__(self):
        pass  # bypass RetrieverAgent.__init__ entirely -- this double never touches its parent's dependencies

    async def search(self, request):
        raise RetrievalError("simulated vector DB outage", transient=True)


# --- Original query path ---

@pytest.mark.asyncio
async def test_uses_original_query_when_no_rewrite_present():
    agent = await make_populated_agent()
    node = RetrievalNode(agent)
    state = GraphState(original_query="refund policy detail number 2")
    result = await node(state)
    assert result.retrieval_result is not None
    assert result.retrieval_result.request.query == "refund policy detail number 2"
    assert len(result.retrieval_result.ranked_chunks) > 0


# --- Rewritten query path ---

@pytest.mark.asyncio
async def test_uses_rewritten_query_when_present_instead_of_original():
    agent = await make_populated_agent()
    node = RetrievalNode(agent)
    state = GraphState(original_query="totally unrelated original text", rewritten_query="refund policy detail number 3")
    result = await node(state)
    assert result.retrieval_result.request.query == "refund policy detail number 3"


@pytest.mark.asyncio
async def test_rewritten_query_field_itself_is_unaffected_by_retrieval_node():
    """RetrievalNode reads rewritten_query via effective_query but must not modify it -- query rewriting is out of scope."""
    agent = await make_populated_agent()
    node = RetrievalNode(agent)
    state = GraphState(original_query="q", rewritten_query="rewritten q")
    result = await node(state)
    assert result.rewritten_query == "rewritten q"
    assert result.original_query == "q"


# --- Empty retrieval results ---

@pytest.mark.asyncio
async def test_empty_index_returns_empty_ranked_chunks_not_an_error():
    agent = make_empty_agent()
    node = RetrievalNode(agent)
    result = await node(GraphState(original_query="anything at all"))
    assert result.retrieval_result is not None
    assert result.retrieval_result.ranked_chunks == []


# --- Retrieval errors ---

@pytest.mark.asyncio
async def test_retrieval_error_propagates_uncaught():
    """
    Per instruction 3, RetrievalNode implements no retries and no error
    handling -- a RetrievalError from RetrieverAgent must propagate
    exactly as raised, not be caught, wrapped, or swallowed.
    """
    node = RetrievalNode(_AlwaysFailingRetrieverAgent())
    with pytest.raises(RetrievalError):
        await node(GraphState(original_query="anything"))


# --- GraphState serialization after retrieval ---

@pytest.mark.asyncio
async def test_retrieval_result_survives_graphstate_serialization_round_trip():
    agent = await make_populated_agent()
    node = RetrievalNode(agent)
    result = await node(GraphState(original_query="refund policy detail number 1"))
    dumped = result.model_dump()
    restored = GraphState.model_validate(dumped)
    assert restored.retrieval_result == result.retrieval_result
    assert len(restored.retrieval_result.ranked_chunks) > 0


@pytest.mark.asyncio
async def test_retrieval_result_survives_json_round_trip():
    agent = await make_populated_agent()
    node = RetrievalNode(agent)
    result = await node(GraphState(original_query="refund policy"))
    restored = GraphState.model_validate_json(result.model_dump_json())
    assert restored.retrieval_result.request.query == "refund policy"


# --- Execution inside the compiled LangGraph ---

@pytest.mark.asyncio
async def test_retrieval_node_executes_when_run_through_the_compiled_graph():
    agent = await make_populated_agent()
    compiled = GraphBuilder(retrieval_node=RetrievalNode(agent)).build()
    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 4"))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata is not None  # planner still ran first
    assert reconstructed.retrieval_result is not None
    assert len(reconstructed.retrieval_result.ranked_chunks) > 0


@pytest.mark.asyncio
async def test_graph_uses_rewritten_query_end_to_end():
    """
    Since Unit 3.5, GraphBuilder defaults a real DecisionNode into every
    graph. Without a VerificationNode also wired in, DecisionNode
    correctly sees "no evidence" (its own defensive fallback) regardless
    of what RetrievalNode actually found, triggers a retry, and -- as of
    Unit 3.7 -- that retry correctly overwrites this test's preset
    rewritten_query with a freshly-derived one. None of that is a bug;
    it's this test not wiring a realistic pipeline. Fixed by adding
    VerificationNode so decision resolves to PROCEED on the first pass,
    matching how the full pipeline is actually meant to be assembled --
    same fix pattern as Unit 3.6's equivalent discovery.
    """
    from app.orchestration.nodes.verification import VerificationNode
    from app.services.verification.contradiction_detector import ContradictionDetector
    from app.services.verification.coverage_analyzer import CoverageAnalyzer
    from app.services.verification.diagnostics_builder import DiagnosticsBuilder
    from app.services.verification.evidence_validator import EvidenceValidator
    from app.services.verification.nli_deterministic import DeterministicNLIVerifier
    from app.services.verification.verification_agent import VerificationAgent

    agent = await make_populated_agent()
    verification_agent = VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier()),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )
    compiled = GraphBuilder(
        retrieval_node=RetrievalNode(agent), verification_node=VerificationNode(verification_agent)
    ).build()
    result = await compiled.ainvoke(
        GraphState(original_query="irrelevant", rewritten_query="refund policy detail number 0")
    )
    reconstructed = GraphState(**result)
    assert reconstructed.retrieval_result.request.query == "refund policy detail number 0"


@pytest.mark.asyncio
async def test_without_retrieval_node_injected_graph_falls_back_to_planner_only():
    """GraphBuilder() with no retrieval_node preserves Unit 3.2's shape exactly -- no regression."""
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke(GraphState(original_query="q"))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata is not None
    assert reconstructed.retrieval_result is None
