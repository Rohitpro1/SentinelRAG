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

from fastapi import Depends, Header, HTTPException
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import-not-found,import-untyped]

from app.core.settings import get_ai_settings, get_chunking_settings
from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.retrieval import RetrievalSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.response_generation import ResponseGenerationNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode
from app.providers.factory import AIProviderFactory
from app.repositories.fakes.in_memory import InMemoryMetadataRepository, InMemoryVectorRepository
from app.repositories.interfaces import MetadataRepository, VectorRepository
from app.services.decision_engine.engine import DecisionEngine
from app.services.embedding.base import BaseEmbedder
from app.services.ingestion.chunker import BaseChunker, SentenceChunker
from app.services.ingestion.ingestion_service import IngestionService
from app.services.query.query_service import QueryService
from app.services.reranking.base import BaseReranker
from app.services.response_generation.base import BaseResponseGenerator
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
from app.services.verification.verification_agent import VerificationAgent


from typing import Optional
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

http_bearer_scheme = HTTPBearer(auto_error=False)


# ----------------------------------------------------------------------
# OpenAPI HTTPBearer security dependency
# Uses HTTPBearer for Swagger Authorize button & cURL generation.
# Reads request.headers directly for raw cURL header compatibility.
# Rejects missing credentials with 401 Unauthorized.
# ----------------------------------------------------------------------
async def get_current_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer_scheme),
) -> str:
    if credentials and credentials.credentials:
        return f"Bearer {credentials.credentials}"

    raw_auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if raw_auth and raw_auth.strip():
        return raw_auth.strip()

    raise HTTPException(status_code=401, detail="Missing Authorization header")


# ----------------------------------------------------------------------
# Provider-backed singleton wiring
# ----------------------------------------------------------------------
@lru_cache
def get_ai_provider_factory() -> AIProviderFactory:
    return AIProviderFactory(get_ai_settings())


@lru_cache
def get_vector_repository() -> VectorRepository:
    return InMemoryVectorRepository()


@lru_cache
def get_metadata_repository() -> MetadataRepository:
    return InMemoryMetadataRepository()


@lru_cache
def get_chunker() -> BaseChunker:
    return SentenceChunker(get_chunking_settings())


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        chunker=get_chunker(),
        embedder=get_embedder(),
        vector_repo=get_vector_repository(),
        metadata_repo=get_metadata_repository(),
    )


@lru_cache
def get_embedder() -> BaseEmbedder:
    return get_ai_provider_factory().create_embedding_provider()


@lru_cache
def get_reranker() -> BaseReranker:
    return get_ai_provider_factory().create_reranker_provider()


@lru_cache
def get_nli_verifier() -> BaseNLIVerifier:
    return get_ai_provider_factory().create_llm_provider()


@lru_cache
def get_response_generator() -> BaseResponseGenerator:
    return get_ai_provider_factory().create_llm_provider()



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
