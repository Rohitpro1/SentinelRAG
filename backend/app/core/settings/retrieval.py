"""
Unit 2.3 — RetrievalSettings.

Same pattern as DecisionEngineSettings/ChunkingSettings (Milestone 1):
one domain, one settings class, one env prefix. Default values are taken
directly from the frozen Retrieval Domain Design -- Section 1 (Timeout
Behavior, Retry Policy) and Section 8 (Performance Targets) -- so this
file is a transcription of already-approved numbers, not a new design
decision.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL__", env_file=".env", extra="ignore")

    # --- Result shaping defaults (Retrieval Domain Design, Section 3) ---
    default_top_k: int = 20
    default_rerank_top_n: int = 5

    # --- Per-stage timeouts in ms (Retrieval Domain Design, Section 1) ---
    embedding_timeout_ms: int = 300
    vector_search_timeout_ms: int = 400
    rerank_timeout_ms: int = 250
    # Soft ceiling, not a hard cutoff -- see Section 1's rationale: exceeding
    # this is a telemetry/alerting signal, RetrieverAgent does not truncate
    # results because the sum of stage latencies crossed it.
    total_soft_budget_ms: int = 1000

    # --- Agent-level retry policy for TRANSIENT infra failures only ---
    # (distinct from DecisionEngineSettings.max_retrieval_retries, which is
    # the query-rewrite retry loop -- see Section 1's explicit warning
    # against conflating the two retry concepts).
    max_transient_retries: int = 2
    retry_backoff_base_ms: int = 100
    retry_backoff_multiplier: float = 3.0  # 100ms, 300ms for the 2 default retries

    # --- Cache ---
    cache_ttl_seconds: int = 300

    def backoff_schedule_ms(self) -> list[int]:
        """
        Returns the actual per-attempt backoff delays, e.g. [100, 300] for
        the defaults above. Exposed as a method (not a hardcoded list) so
        changing max_transient_retries via env var automatically produces
        a correctly-sized schedule without a second setting to keep in sync.
        """
        return [
            int(self.retry_backoff_base_ms * (self.retry_backoff_multiplier ** attempt))
            for attempt in range(self.max_transient_retries)
        ]
