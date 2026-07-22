from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE__", env_file=".env", extra="ignore")

    postgres_dsn: str = "postgresql+asyncpg://sentinelrag:sentinelrag@localhost:5432/sentinelrag"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sentinelrag_chunks"
    qdrant_timeout_ms: int = 5000
    qdrant_vector_size: int = 1536
