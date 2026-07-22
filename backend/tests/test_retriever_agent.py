"""
Unit 2.6 tests -- RetrieverAgent orchestration: full pipeline wiring with
fakes, cache hit/miss behavior, and telemetry-relevant stage_latencies_ms
being populated.
"""
import pytest

from app.core.settings.retrieval import RetrievalSettings
from app.repositories.fakes.in_memory import InMemoryCacheRepository, InMemoryVectorRepository
from app.schemas.retrieval import Chunk
from app.schemas.retrieval_domain import SearchRequest
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.reranking.deterministic import DeterministicReranker
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.retriever_agent import RetrieverAgent
from app.services.retrieval.search_service import SearchService


async def make_populated_agent(cache_repository=None, settings=None):
    settings = settings or RetrievalSettings()
    embedder = DeterministicEmbedder(dimensions=8)
    vector_repo = InMemoryVectorRepository()

    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", text=f"refund policy detail {i}", token_count=10, source_reliability_score=0.9)
        for i in range(5)
    ]
    embeddings = [await embedder.embed_query(c.text) for c in chunks]
    await vector_repo.upsert(chunks, embeddings)

    agent = RetrieverAgent(
        embedding_service=EmbeddingService(embedder, settings),
        search_service=SearchService(vector_repo, settings),
        fusion_service=FusionService(),
        reranking_service=RerankingService(DeterministicReranker(), settings),
        settings=settings,
        cache_repository=cache_repository,
    )
    return agent


@pytest.mark.asyncio
async def test_search_returns_ranked_chunks_end_to_end():
    agent = await make_populated_agent()
    response = await agent.search(SearchRequest(query="refund policy detail 2", top_k=5, rerank_top_n=3))
    assert len(response.ranked_chunks) == 3
    assert response.cache_hit is False


@pytest.mark.asyncio
async def test_stage_latencies_are_populated():
    agent = await make_populated_agent()
    response = await agent.search(SearchRequest(query="refund policy", top_k=5, rerank_top_n=3))
    assert set(response.stage_latencies_ms.keys()) == {"embedding", "vector_search", "rerank"}
    assert all(v >= 0 for v in response.stage_latencies_ms.values())


@pytest.mark.asyncio
async def test_cache_miss_then_hit():
    cache = InMemoryCacheRepository()
    agent = await make_populated_agent(cache_repository=cache)
    request = SearchRequest(query="refund policy", top_k=5, rerank_top_n=3)

    first = await agent.search(request)
    assert first.cache_hit is False

    second = await agent.search(request)
    assert second.cache_hit is True
    # Same underlying ranked chunks served from cache.
    assert [rc.retrieved_chunk.chunk.chunk_id for rc in second.ranked_chunks] == [
        rc.retrieved_chunk.chunk.chunk_id for rc in first.ranked_chunks
    ]


@pytest.mark.asyncio
async def test_different_requests_produce_different_cache_keys():
    cache = InMemoryCacheRepository()
    agent = await make_populated_agent(cache_repository=cache)

    r1 = await agent.search(SearchRequest(query="refund policy", top_k=5, rerank_top_n=3))
    r2 = await agent.search(SearchRequest(query="shipping policy", top_k=5, rerank_top_n=3))
    assert r1.cache_hit is False
    assert r2.cache_hit is False  # different query -> different cache key -> genuine miss


@pytest.mark.asyncio
async def test_search_without_cache_repository_still_works():
    agent = await make_populated_agent(cache_repository=None)
    response = await agent.search(SearchRequest(query="refund policy", top_k=5, rerank_top_n=2))
    assert len(response.ranked_chunks) == 2
