from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING__", env_file=".env", extra="ignore")

    target_tokens: int = 350
    min_tokens: int = 80
    overlap_tokens: int = 40
