from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class DecisionEngineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DECISION_ENGINE__", env_file=".env", extra="ignore")

    min_retrieval_similarity: float = 0.55
    contradiction_threshold: float = 0.40
    low_confidence_threshold: float = 0.60
    max_retrieval_retries: int = 2

    weight_similarity: float = 0.45
    weight_ocr_confidence: float = 0.20
    weight_source_reliability: float = 0.20
    weight_contradiction_penalty: float = 0.35
