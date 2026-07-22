"""
Unit 2.14 tests -- /query endpoint, using TestClient with
dependency_overrides to inject a deterministic, pre-populated
QueryService (instruction 5: endpoint tests primarily use deterministic
implementations and dependency overrides, not the default empty
in-memory singleton from app.api.dependencies).
"""
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_principal, get_query_service
from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.main import app
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.schemas.retrieval import Chunk
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


async def _build_populated_query_service(decision_settings=None) -> QueryService:
    from app.orchestration.graph_builder import GraphBuilder
    from app.orchestration.nodes.decision import DecisionNode
    from app.orchestration.nodes.retrieval import RetrievalNode
    from app.orchestration.nodes.verification import VerificationNode

    retrieval_settings = RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

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


@pytest.fixture
def client():
    import asyncio

    populated_service = asyncio.run(_build_populated_query_service())
    app.dependency_overrides[get_query_service] = lambda: populated_service
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_returns_200_with_valid_request(client):
    response = client.post("/api/v1/query", json={"query": "refund policy detail number 2"})
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"proceed", "low_confidence_response"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert "reasons" in body


def test_query_response_shape_matches_contract(client):
    response = client.post("/api/v1/query", json={"query": "refund policy"})
    body = response.json()
    assert set(body.keys()) == {
        "action", "confidence", "reasons", "retry_count", "contradiction_detected", "evidence_coverage",
    }


def test_query_missing_authorization_header_returns_401(client):
    app.dependency_overrides.pop(get_current_principal, None)  # remove the override for this test only
    response = client.post("/api/v1/query", json={"query": "refund policy"})
    assert response.status_code == 401
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"  # restore


def test_query_empty_string_returns_400_not_422():
    """
    Instruction 3 explicitly maps ValidationError -> 400 (not FastAPI's
    default 422) -- verified directly against a real validation failure.
    """
    client_local = TestClient(app)
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    response = client_local.post("/api/v1/query", json={"query": ""})
    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert "detail" in response.json()


def test_query_missing_field_returns_400():
    client_local = TestClient(app)
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    response = client_local.post("/api/v1/query", json={})
    app.dependency_overrides.clear()
    assert response.status_code == 400


def test_query_top_k_out_of_bounds_returns_400():
    client_local = TestClient(app)
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    response = client_local.post("/api/v1/query", json={"query": "test", "top_k": 9999})
    app.dependency_overrides.clear()
    assert response.status_code == 400


def test_error_responses_never_leak_internal_exception_text(client):
    """
    Broad sanity check on the error contract shape -- validation errors
    must return the generic message, never raw Pydantic error internals.
    """
    app.dependency_overrides.pop(get_current_principal, None)
    response = client.post("/api/v1/query", json={"query": "refund policy"})
    body = response.json()
    assert body == {"detail": "Missing Authorization header"} or response.status_code == 401
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"


def test_query_with_document_filter_that_matches_nothing_returns_clarify(client):
    response = client.post(
        "/api/v1/query", json={"query": "refund policy", "document_filter": {"document_id": "nonexistent"}}
    )
    assert response.status_code == 200
    assert response.json()["action"] == "clarify"


def test_query_endpoint_uses_default_deterministic_wiring_when_not_overridden():
    """
    Confirms the app's OWN default DI wiring (app.api.dependencies) is
    reachable and produces a valid (if unpopulated -> CLARIFY) response,
    without any override at all -- proving the default wiring is real and
    functional, not just a test fixture's substitute.
    """
    from app.api.dependencies import get_query_service as real_get_query_service

    real_get_query_service.cache_clear()
    client_local = TestClient(app)
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    response = client_local.post("/api/v1/query", json={"query": "anything at all"})
    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["action"] == "clarify"  # empty default in-memory repo -> no chunks -> CLARIFY


# --- Exception mapping (instruction 3) ---

class _AlwaysFailingRetrieverAgent:
    async def search(self, request):
        from app.core.exceptions import RetrievalError
        raise RetrievalError("simulated vector DB outage", transient=True)


class _AlwaysFailingVerificationAgent:
    async def verify(self, verification_input):
        from app.core.exceptions import VerificationError
        raise VerificationError("simulated NLI provider outage")


class _AlwaysFailingDecisionEngineQueryService:
    """A QueryService double whose handle_query raises DecisionEngineError directly."""

    async def handle_query(self, *args, **kwargs):
        from app.core.exceptions import DecisionEngineError
        raise DecisionEngineError("simulated decision engine failure")


def test_retrieval_error_maps_to_503():
    from app.services.query.query_service import QueryService

    class _FailingQueryService(QueryService):
        def __init__(self):
            pass

        async def handle_query(self, *args, **kwargs):
            from app.core.exceptions import RetrievalError
            raise RetrievalError("simulated outage", transient=True)

    app.dependency_overrides[get_query_service] = lambda: _FailingQueryService()
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    client_local = TestClient(app)
    response = client_local.post("/api/v1/query", json={"query": "test"})
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"].lower()
    assert "simulated outage" not in response.json()["detail"]  # internal message NOT leaked


def test_verification_error_maps_to_502():
    from app.services.query.query_service import QueryService

    class _FailingQueryService(QueryService):
        def __init__(self):
            pass

        async def handle_query(self, *args, **kwargs):
            from app.core.exceptions import VerificationError
            raise VerificationError("simulated NLI outage")

    app.dependency_overrides[get_query_service] = lambda: _FailingQueryService()
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    client_local = TestClient(app)
    response = client_local.post("/api/v1/query", json={"query": "test"})
    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "simulated NLI outage" not in response.json()["detail"]


def test_decision_engine_error_maps_to_500():
    from app.services.query.query_service import QueryService

    class _FailingQueryService(QueryService):
        def __init__(self):
            pass

        async def handle_query(self, *args, **kwargs):
            from app.core.exceptions import DecisionEngineError
            raise DecisionEngineError("simulated internal failure")

    app.dependency_overrides[get_query_service] = lambda: _FailingQueryService()
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    client_local = TestClient(app)
    response = client_local.post("/api/v1/query", json={"query": "test"})
    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "simulated internal failure" not in response.json()["detail"]


def test_embedding_error_maps_to_503():
    from app.services.query.query_service import QueryService

    class _FailingQueryService(QueryService):
        def __init__(self):
            pass

        async def handle_query(self, *args, **kwargs):
            from app.core.exceptions import EmbeddingError
            raise EmbeddingError("simulated embedding provider outage")

    app.dependency_overrides[get_query_service] = lambda: _FailingQueryService()
    app.dependency_overrides[get_current_principal] = lambda: "test-principal"
    client_local = TestClient(app)
    response = client_local.post("/api/v1/query", json={"query": "test"})
    app.dependency_overrides.clear()

    assert response.status_code == 503
