"""
Unit 3.2 -- PlannerSettings.

Same per-domain settings pattern as every prior domain (DecisionEngine,
Chunking, Storage, Security, Embedding, Retrieval, Reranking, NLI).
Numeric thresholds only -- the question-word lexicon used for
classification is a module-level constant in nodes/planner.py, not a
runtime setting, matching this project's existing style (no prior
settings class has externalized a word list via env var either).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlannerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLANNER__", env_file=".env", extra="ignore")

    min_words_threshold: int = 2  # non-empty queries with fewer words than this classify as TOO_SHORT
    multi_part_question_mark_threshold: int = 2  # this many '?' characters or more classifies as MULTI_PART
