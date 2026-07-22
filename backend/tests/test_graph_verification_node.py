"""
Unit 3.4 tests -- VerificationNode: covers exactly the five areas
instruction 6 lists, using deterministic VerificationAgent
implementations (Unit 2.9), constructor-injected dependency overrides,
and no real network.
"""
import pytest

from app.core.exceptions import VerificationError
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk, SearchRequest, SearchResponse
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
from app.core.settings.retrieval import RetrievalSettings


def make_verification_agent(conflict_marker="[CONTRADICTION]") -> VerificationAgent:
    return VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier(conflict_marker=conflict_marker)),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )


def make_ranked_chunk(chunk_id, text, rerank_score=0.8):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text=text, token_count=10, source_reliability_score=0.9)
    return RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=rerank_score, rank=0)


def make_search_response(query, ranked_chunks):
    return SearchResponse(request=SearchRequest(query=query), ranked_chunks=ranked_chunks)


async def make_populated_retrieval_node() -> RetrievalNode:
    settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.9)
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


class _AlwaysFailingVerificationAgent(VerificationAgent):
    def __init__(self):
        pass  # bypass parent's __init__ entirely -- this double never touches its dependencies

    async def verify(self, verification_input):
        raise VerificationError("simulated NLI provider outage")


# --- Successful verification ---

@pytest.mark.asyncio
async def test_successful_verification_populates_both_outputs():
    agent = make_verification_agent()
    node = VerificationNode(agent)
    ranked = [make_ranked_chunk("c1", "refund policy detail"), make_ranked_chunk("c2", "more refund detail")]
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", ranked))
    result = await node(state)
    assert result.verification_result is not None
    assert len(result.verification_result.retrieved_chunks) == 2
    assert result.diagnostics is not None
    assert result.diagnostics.query == "q"


@pytest.mark.asyncio
async def test_verification_uses_effective_query_not_original():
    agent = make_verification_agent()
    node = VerificationNode(agent)
    ranked = [make_ranked_chunk("c1", "content")]
    state = GraphState(
        original_query="irrelevant", rewritten_query="the real query", retrieval_result=make_search_response("q", ranked)
    )
    result = await node(state)
    assert result.diagnostics.query == "the real query"


@pytest.mark.asyncio
async def test_contradiction_detected_and_surfaced_in_diagnostics():
    agent = make_verification_agent(conflict_marker="[X]")
    node = VerificationNode(agent)
    ranked = [make_ranked_chunk("c1", "refunds allowed [X]"), make_ranked_chunk("c2", "refunds not allowed [X]")]
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", ranked))
    result = await node(state)
    assert result.diagnostics.contradiction_detected is True


# --- Empty retrieval results ---

@pytest.mark.asyncio
async def test_empty_ranked_chunks_still_produces_valid_outputs():
    agent = make_verification_agent()
    node = VerificationNode(agent)
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", []))
    result = await node(state)
    assert result.verification_result.retrieved_chunks == []
    assert result.diagnostics.evidence_coverage == 0.0


@pytest.mark.asyncio
async def test_no_retrieval_result_at_all_defaults_to_empty_chunks():
    """VerificationNode must not crash if retrieval_result is None (e.g. tested standalone, no RetrievalNode ran)."""
    agent = make_verification_agent()
    node = VerificationNode(agent)
    state = GraphState(original_query="q")  # retrieval_result never set
    result = await node(state)
    assert result.verification_result.retrieved_chunks == []


# --- Verification failures ---

@pytest.mark.asyncio
async def test_verification_error_propagates_uncaught():
    """Per instruction 3, VerificationNode implements no retries/error handling -- failures propagate exactly as raised."""
    node = VerificationNode(_AlwaysFailingVerificationAgent())
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", []))
    with pytest.raises(VerificationError):
        await node(state)


# --- GraphState serialization after verification ---

@pytest.mark.asyncio
async def test_verification_outputs_survive_graphstate_serialization_round_trip():
    agent = make_verification_agent()
    node = VerificationNode(agent)
    ranked = [make_ranked_chunk("c1", "content")]
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", ranked))
    result = await node(state)

    dumped = result.model_dump()
    restored = GraphState.model_validate(dumped)
    assert restored.verification_result == result.verification_result
    assert restored.diagnostics == result.diagnostics


@pytest.mark.asyncio
async def test_verification_outputs_survive_json_round_trip():
    agent = make_verification_agent()
    node = VerificationNode(agent)
    ranked = [make_ranked_chunk("c1", "content")]
    state = GraphState(original_query="q", retrieval_result=make_search_response("q", ranked))
    result = await node(state)
    restored = GraphState.model_validate_json(result.model_dump_json())
    assert restored.diagnostics.query == "q"
    assert len(restored.verification_result.retrieved_chunks) == 1


# --- Execution inside the compiled LangGraph ---

@pytest.mark.asyncio
async def test_verification_node_executes_when_run_through_the_compiled_graph():
    retrieval_node = await make_populated_retrieval_node()
    verification_agent = make_verification_agent()
    compiled = GraphBuilder(retrieval_node=retrieval_node, verification_node=VerificationNode(verification_agent)).build()

    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 2"))
    reconstructed = GraphState(**result)

    assert reconstructed.planning_metadata is not None
    assert reconstructed.retrieval_result is not None
    assert reconstructed.verification_result is not None
    assert reconstructed.diagnostics is not None
    assert len(reconstructed.verification_result.retrieved_chunks) > 0


@pytest.mark.asyncio
async def test_without_verification_node_injected_graph_falls_back_to_retrieval_only():
    """GraphBuilder with only retrieval_node preserves Unit 3.3's shape exactly -- no regression."""
    retrieval_node = await make_populated_retrieval_node()
    compiled = GraphBuilder(retrieval_node=retrieval_node).build()
    result = await compiled.ainvoke(GraphState(original_query="refund policy detail number 1"))
    reconstructed = GraphState(**result)
    assert reconstructed.retrieval_result is not None
    assert reconstructed.verification_result is None
    assert reconstructed.diagnostics is None
