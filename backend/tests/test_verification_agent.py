"""
Unit 2.9 tests -- VerificationAgent orchestration, and the full
cross-domain integration: RetrieverAgent -> VerificationAgent ->
DecisionEngine, using each unit's REAL output (not hand-built fixtures)
at every step.
"""
import pytest

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk
from app.schemas.retrieval_domain import SearchRequest, VerificationInput
from app.services.decision_engine.engine import DecisionEngine
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


def make_verification_agent(conflict_marker="[CONTRADICTION]"):
    return VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(DeterministicNLIVerifier(conflict_marker=conflict_marker)),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )


def make_ranked_chunks_input(texts, rerank_top_n=None):
    from app.schemas.retrieval import RetrievedChunk
    from app.schemas.retrieval_domain import RankedChunk

    ranked = []
    for i, text in enumerate(texts):
        chunk = Chunk(chunk_id=f"c{i}", document_id="doc-1", text=text, token_count=10, source_reliability_score=0.9)
        ranked.append(RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=0.8, rank=i))
    return VerificationInput(query="q", ranked_chunks=ranked, retry_count=0)


@pytest.mark.asyncio
async def test_verify_returns_both_outputs():
    agent = make_verification_agent()
    vi = make_ranked_chunks_input(["policy A applies here.", "further detail on policy A."])
    evidence, diagnostics = await agent.verify(vi)
    assert evidence.query == "q"
    assert len(evidence.retrieved_chunks) == 2
    assert diagnostics.query == "q"
    assert diagnostics.verification_latency_ms >= 0


@pytest.mark.asyncio
async def test_contradiction_surfaces_in_diagnostics_not_lost():
    agent = make_verification_agent(conflict_marker="[X]")
    vi = make_ranked_chunks_input(["refunds allowed [X]", "refunds not allowed [X]"])
    evidence, diagnostics = await agent.verify(vi)
    assert diagnostics.contradiction_detected is True
    assert evidence.to_verification_report().has_contradiction is True  # raw NLI results still flow to Decision Engine


@pytest.mark.asyncio
async def test_business_output_excludes_diagnostics_fields():
    agent = make_verification_agent()
    vi = make_ranked_chunks_input(["a", "b"])
    evidence, _ = await agent.verify(vi)
    assert not hasattr(evidence, "nli_score")
    assert not hasattr(evidence, "reranker_confidence")


@pytest.mark.asyncio
async def test_unsupported_evidence_excluded_from_business_output_but_flagged_in_diagnostics():
    from app.schemas.retrieval import RetrievedChunk
    from app.schemas.retrieval_domain import RankedChunk

    good_chunk = Chunk(chunk_id="c1", document_id="doc-1", text="valid", token_count=10, source_reliability_score=0.9)
    bad_chunk = Chunk(chunk_id="c2", document_id="doc-1", text="   ", token_count=10, source_reliability_score=0.9)
    vi = VerificationInput(
        query="q",
        ranked_chunks=[
            RankedChunk(retrieved_chunk=RetrievedChunk(chunk=good_chunk, similarity_score=0.9), rerank_score=0.8, rank=0),
            RankedChunk(retrieved_chunk=RetrievedChunk(chunk=bad_chunk, similarity_score=0.9), rerank_score=0.8, rank=1),
        ],
    )
    agent = make_verification_agent()
    evidence, diagnostics = await agent.verify(vi)
    assert len(evidence.retrieved_chunks) == 1
    assert diagnostics.unsupported_claims == ["c2"]
    assert diagnostics.evidence_coverage == 0.5


@pytest.mark.asyncio
async def test_verified_evidence_adapts_to_decision_report_correctly():
    agent = make_verification_agent()
    vi = make_ranked_chunks_input(["a", "b"])
    evidence, _ = await agent.verify(vi)
    report = evidence.to_verification_report()
    assert report.query == "q"
    assert len(report.retrieved_chunks) == 2


@pytest.mark.asyncio
async def test_full_cross_domain_pipeline_retriever_to_verification_to_decision_engine():
    """
    The single most important test in this unit: RetrieverAgent's REAL
    output feeds VerificationAgent, whose REAL VerifiedEvidence output
    feeds a REAL DecisionEngine.evaluate() call -- no hand-built fixtures
    at any step. This is what actually proves the frozen dependency graph
    (Retrieval Domain <-> Decision Engine seam) holds under real data flow.
    """
    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail number {i}", token_count=10, source_reliability_score=0.9)
        for i in range(4)
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
    search_response = await retriever.search(SearchRequest(query="refund policy detail number 2", top_k=4, rerank_top_n=3))
    assert len(search_response.ranked_chunks) == 3

    verification_agent = make_verification_agent()
    verification_input = VerificationInput(
        query=search_response.request.query, ranked_chunks=search_response.ranked_chunks, retry_count=0
    )
    verified_evidence, diagnostics = await verification_agent.verify(verification_input)
    assert len(verified_evidence.retrieved_chunks) == 3
    assert diagnostics.contradiction_detected is False  # no [CONTRADICTION] markers in synthetic data

    decision_engine = DecisionEngine(DecisionEngineSettings())
    decision = decision_engine.evaluate(verified_evidence.to_verification_report())
    assert decision.action.value in {"proceed", "low_confidence_response"}
    assert decision.explainability is not None
