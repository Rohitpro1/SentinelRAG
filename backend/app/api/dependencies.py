"""
Unit 2.14 -- FastAPI dependency-injection wiring.

Per instruction 4 ("the route must not instantiate services or
infrastructure clients directly"): this is the ONLY module that
constructs QueryService and everything it transitively depends on. The
route (query_router.py) only ever calls `Depends(get_query_service)`.

DEFAULT WIRING IS DETERMINISTIC, not real infrastructure. This is a
deliberate choice, not an oversight: it means `uvicorn app.main:app` runs
correctly with zero external services (no Qdrant, no LLM provider)
out of the box, which matters for local development, CI, and this
sandbox (no network route to real infra/providers, same constraint noted
in Units 2.10-2.13). Swapping in real infrastructure (QdrantVectorRepository,
OpenAIEmbedder, CrossEncoderReranker, LLMBasedNLIVerifier -- all already
built in Units 2.10-2.13) is a change to THIS FILE ONLY when that
becomes the deployment target -- no other module needs to change, which
is exactly what constructor DI throughout this codebase has been
building toward.

FastAPI's dependency-caching (each Depends-decorated function is called
once per request by default; combined with @lru_cache on the singleton
factories below, these are effectively process-wide singletons) means
this wiring is cheap to call on every request.
Unit 3.8 -- get_query_service() now builds a compiled LangGraph (via
GraphBuilder, wrapping the same RetrieverAgent/VerificationAgent/
DecisionEngine this file already constructed) and injects the COMPILED
GRAPH into QueryService, rather than injecting the three agents directly.
This is the only change this unit makes here -- every other function in
this file (the individual agent/service factories) is untouched, since
GraphBuilder's nodes still need exactly the same underlying components.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import-not-found,import-untyped]

from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.response_generation import ResponseGenerationNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.repositories.interfaces import VectorRepository
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.query.query_service import QueryService
from app.services.reranking.base import BaseReranker
from app.services.reranking.deterministic import DeterministicReranker
from app.services.response_generation.base import BaseResponseGenerator
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
from app.services.verification.nli_base import BaseNLIVerifier
from app.services.verification.nli_deterministic import DeterministicNLIVerifier
from app.services.verification.verification_agent import VerificationAgent


# ----------------------------------------------------------------------
# Placeholder authentication (instruction 1: "authentication (placeholder
# if not yet implemented)"). SecuritySettings (Milestone 1) already
# declares AUTH0_DOMAIN/AUTH0_AUDIENCE fields for when real Auth0
# integration is built (Milestone 3) -- this dependency does not read
# them yet. It exists so the route has an explicit auth seam from day
# one, rather than bolting auth on later by editing the route itself.
# ----------------------------------------------------------------------
async def get_current_principal(authorization: str = Header(default=None)) -> str:
    """
    PLACEHOLDER -- does not verify a real token. Any non-empty
    Authorization header is accepted; a missing header is rejected with
    401. This is intentionally weak and clearly marked as such: replacing
    this function's body with real Auth0 JWT verification (Milestone 3)
    is the entire scope of that future change -- the route and everything
    below it never needs to change, since they only depend on this
    function's signature (returns a principal identifier string).
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return authorization


# ----------------------------------------------------------------------
# Deterministic-backed singleton wiring
# ----------------------------------------------------------------------
@lru_cache
def get_vector_repository() -> VectorRepository:
    return InMemoryVectorRepository()


@lru_cache
def get_embedder() -> BaseEmbedder:
    return DeterministicEmbedder(dimensions=16)


@lru_cache
def get_reranker() -> BaseReranker:
    return DeterministicReranker()


@lru_cache
def get_nli_verifier() -> BaseNLIVerifier:
    return DeterministicNLIVerifier()


@lru_cache
def get_response_generator() -> BaseResponseGenerator:
    return ResponseGenerator()


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()


@lru_cache
def get_decision_engine_settings() -> DecisionEngineSettings:
    return DecisionEngineSettings()


@lru_cache
def get_retriever_agent() -> RetrieverAgent:
    retrieval_settings = get_retrieval_settings()
    return RetrieverAgent(
        embedding_service=EmbeddingService(get_embedder(), retrieval_settings),
        search_service=SearchService(get_vector_repository(), retrieval_settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(get_reranker(), retrieval_settings),
        settings=retrieval_settings,
    )


@lru_cache
def get_verification_agent() -> VerificationAgent:
    return VerificationAgent(
        evidence_validator=EvidenceValidator(),
        contradiction_detector=ContradictionDetector(get_nli_verifier()),
        coverage_analyzer=CoverageAnalyzer(),
        diagnostics_builder=DiagnosticsBuilder(),
    )


@lru_cache
def get_decision_engine() -> DecisionEngine:
    return DecisionEngine(get_decision_engine_settings())


@lru_cache
def get_compiled_graph() -> CompiledStateGraph:
    """
    Unit 3.8 & 3.9: assembles the full LangGraph -- planner -> retrieval ->
    verification -> decision -> response_generation, with DecisionNode's
    conditional routing closing the RETRY_RETRIEVAL loop and PlannerNode's
    retry-aware rewriting engaging on each retry pass.
    """
    return GraphBuilder(
        retrieval_node=RetrievalNode(get_retriever_agent()),
        verification_node=VerificationNode(get_verification_agent()),
        decision_node=DecisionNode(get_decision_engine()),
        response_generation_node=ResponseGenerationNode(get_response_generator()),
    ).build()


@lru_cache
def get_query_service() -> QueryService:
    return QueryService(get_compiled_graph())
