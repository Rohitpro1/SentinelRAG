"""
Unit 2.12 -- RerankingSettings.

Same pattern as EmbeddingSettings (Unit 2.11): one domain, one settings
class, one env prefix, provider connection config plus configuration-
driven capability metadata (never runtime-discovered, per instruction 2).
This is the reranking domain's first dedicated settings class -- prior to
this unit, only RetrievalSettings.rerank_timeout_ms existed (the stage-
level timeout enforced by RerankingService, Unit 2.6). That setting is
UNCHANGED and still owns the stage timeout; this new class owns the real
provider's own connection/capability config, mirroring the
StorageSettings/RetrievalSettings split already established for Qdrant
(qdrant_timeout_ms vs vector_search_timeout_ms, Unit 2.10).
"""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKING__", env_file=".env", extra="ignore")

    model_name: str = "rerank-default"
    api_base_url: str = "https://api.cohere.ai/v1"
    api_key: str = ""
    request_timeout_ms: int = 5000

    # Configuration-driven capability metadata (instruction 2) -- set from
    # the provider's documented limits, not discovered at runtime.
    supports_batching: bool = True
    max_batch_size: int = 32
    max_input_tokens: int = 512
    model_dimensions: Optional[int] = None
