"""
Settings composition root — no monolithic Settings god-object.

Each service constructor takes exactly the domain settings it depends
on (see DecisionEngine, SentenceChunker). AppSettings exists only for
application bootstrap (main.py DI wiring), never for a service to
depend on directly.
"""
from __future__ import annotations
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.settings.chunking import ChunkingSettings
from app.core.settings.decision_engine import DecisionEngineSettings
from app.core.settings.nli import NLISettings
from app.core.settings.planner import PlannerSettings
from app.core.settings.embedding import EmbeddingSettings
from app.core.settings.reranking import RerankingSettings
from app.core.settings.retrieval import RetrievalSettings
from app.core.settings.security import SecuritySettings
from app.core.settings.storage import StorageSettings

__all__ = [
    "ChunkingSettings", "DecisionEngineSettings", "EmbeddingSettings", "RetrievalSettings",
    "RerankingSettings", "SecuritySettings", "StorageSettings", "AppSettings",
    "get_app_settings", "get_decision_engine_settings", "get_chunking_settings",
    "get_storage_settings", "get_security_settings", "get_embedding_settings",
    "get_retrieval_settings", "get_reranking_settings", "get_nli_settings", "get_planner_settings",
]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SentinelRAG"
    environment: str = "development"
    debug: bool = False

    decision_engine: DecisionEngineSettings = Field(default_factory=DecisionEngineSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    nli: NLISettings = Field(default_factory=NLISettings)
    planner: PlannerSettings = Field(default_factory=PlannerSettings)


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()


@lru_cache
def get_decision_engine_settings() -> DecisionEngineSettings:
    return DecisionEngineSettings()


@lru_cache
def get_chunking_settings() -> ChunkingSettings:
    return ChunkingSettings()


@lru_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()


@lru_cache
def get_security_settings() -> SecuritySettings:
    return SecuritySettings()


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings()


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()


@lru_cache
def get_reranking_settings() -> RerankingSettings:
    return RerankingSettings()


@lru_cache
def get_nli_settings() -> NLISettings:
    return NLISettings()


@lru_cache
def get_planner_settings() -> PlannerSettings:
    return PlannerSettings()
