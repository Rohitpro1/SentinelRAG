"""
Unit 2.3 tests -- defaults match the frozen Retrieval Domain Design's
Section 1/8 numbers, env override works, backoff schedule derivation is correct.
"""
import os

from app.core.settings.retrieval import RetrievalSettings


def test_defaults_match_frozen_design_numbers():
    s = RetrievalSettings()
    assert s.default_top_k == 20
    assert s.default_rerank_top_n == 5
    assert s.embedding_timeout_ms == 300
    assert s.vector_search_timeout_ms == 400
    assert s.rerank_timeout_ms == 250
    assert s.total_soft_budget_ms == 1000
    assert s.max_transient_retries == 2


def test_backoff_schedule_matches_design_example():
    s = RetrievalSettings()
    # Design doc: "up to 2 retries with exponential backoff (100ms, 300ms)"
    assert s.backoff_schedule_ms() == [100, 300]


def test_backoff_schedule_scales_with_retry_count_override():
    s = RetrievalSettings(max_transient_retries=3)
    schedule = s.backoff_schedule_ms()
    assert len(schedule) == 3
    assert schedule == sorted(schedule)  # monotonically increasing


def test_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL__DEFAULT_TOP_K", "50")
    s = RetrievalSettings()
    assert s.default_top_k == 50


def test_cache_ttl_default():
    assert RetrievalSettings().cache_ttl_seconds == 300
