import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.base.embedding_provider import BaseEmbeddingProvider
from app.providers.router.embedding_router import EmbeddingRouterProvider


class MockEmbedder(BaseEmbeddingProvider):
    def __init__(self, name: str, dims: int, should_fail: bool = False, fail_exc: Exception = None):
        self._name = name
        self._dims = dims
        self.should_fail = should_fail
        self.fail_exc = fail_exc or EmbeddingError("Mock service failure")

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed_query(self, text: str) -> list[float]:
        if self.should_fail:
            raise self.fail_exc
        return [0.5] * self._dims

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.should_fail:
            raise self.fail_exc
        return [[0.5] * self._dims for _ in texts]


@pytest.mark.asyncio
async def test_router_primary_success():
    primary = MockEmbedder("gemini", 768)
    secondary = MockEmbedder("voyage", 1024)
    router = EmbeddingRouterProvider(primary=primary, secondary=secondary)

    res = await router.embed_query("hello")
    assert len(res) == 768
    assert router.active_provider_name == "gemini"

    metrics = router.get_metrics()
    assert metrics["gemini_success_count"] == 1
    assert metrics["failover_count"] == 0


@pytest.mark.asyncio
async def test_router_recoverable_failover_to_voyage():
    primary = MockEmbedder("gemini", 768, should_fail=True, fail_exc=EmbeddingError("Gemini 429 Rate Limit"))
    secondary = MockEmbedder("voyage", 1024)
    router = EmbeddingRouterProvider(primary=primary, secondary=secondary)

    res = await router.embed_query("hello")
    assert len(res) == 1024
    assert router.active_provider_name == "voyage"

    metrics = router.get_metrics()
    assert metrics["gemini_failure_count"] == 1
    assert metrics["voyage_usage_count"] == 1
    assert metrics["failover_count"] == 1


@pytest.mark.asyncio
async def test_router_unrecoverable_auth_error_no_failover():
    primary = MockEmbedder("gemini", 768, should_fail=True, fail_exc=SentinelRAGError("Gemini Auth 401"))
    secondary = MockEmbedder("voyage", 1024)
    router = EmbeddingRouterProvider(primary=primary, secondary=secondary)

    with pytest.raises(SentinelRAGError, match="Auth 401"):
        await router.embed_query("hello")

    metrics = router.get_metrics()
    assert metrics["failover_count"] == 0
