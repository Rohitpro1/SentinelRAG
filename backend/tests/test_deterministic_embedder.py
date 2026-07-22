"""
Unit 2.4 tests -- fake embedder is deterministic and dimensionally consistent.
"""
import math

import pytest

from app.services.embedding.deterministic import DeterministicEmbedder


@pytest.mark.asyncio
async def test_embed_query_is_deterministic():
    embedder = DeterministicEmbedder(dimensions=16)
    v1 = await embedder.embed_query("what is the refund policy?")
    v2 = await embedder.embed_query("what is the refund policy?")
    assert v1 == v2


@pytest.mark.asyncio
async def test_different_texts_produce_different_vectors():
    embedder = DeterministicEmbedder(dimensions=16)
    v1 = await embedder.embed_query("refund policy")
    v2 = await embedder.embed_query("shipping policy")
    assert v1 != v2


@pytest.mark.asyncio
async def test_embed_query_respects_configured_dimensions():
    embedder = DeterministicEmbedder(dimensions=32)
    v = await embedder.embed_query("test")
    assert len(v) == 32
    assert embedder.dimensions == 32


@pytest.mark.asyncio
async def test_embed_batch_matches_individual_embed_query_calls():
    embedder = DeterministicEmbedder(dimensions=16)
    texts = ["a", "b", "c"]
    batch_result = await embedder.embed_batch(texts)
    individual_results = [await embedder.embed_query(t) for t in texts]
    assert batch_result == individual_results


@pytest.mark.asyncio
async def test_embed_batch_empty_list():
    embedder = DeterministicEmbedder(dimensions=16)
    assert await embedder.embed_batch([]) == []


@pytest.mark.asyncio
async def test_vectors_are_unit_normalized():
    embedder = DeterministicEmbedder(dimensions=16)
    v = await embedder.embed_query("normalize me")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_default_dimensions_is_16():
    embedder = DeterministicEmbedder()
    assert embedder.dimensions == 16


# --- Unit 2.11 additions: EmbeddingResult / EmbedderHealth ---

@pytest.mark.asyncio
async def test_embed_query_with_result_shape():
    from app.schemas.embedding import EmbedderHealthState

    embedder = DeterministicEmbedder(dimensions=8)
    result = await embedder.embed_query_with_result("hello")
    assert len(result.vector) == 8
    assert result.provider == "deterministic"
    assert result.model_name == "deterministic-sha256"
    assert result.model_version == "v1"
    assert result.embedding_dimensions == 8
    assert result.generation_latency_ms >= 0


@pytest.mark.asyncio
async def test_embed_query_with_result_vector_matches_embed_query():
    embedder = DeterministicEmbedder(dimensions=8)
    direct = await embedder.embed_query("consistency check")
    via_result = (await embedder.embed_query_with_result("consistency check")).vector
    assert direct == via_result


def test_health_always_ready():
    from app.schemas.embedding import EmbedderHealthState

    embedder = DeterministicEmbedder()
    assert embedder.health().state == EmbedderHealthState.READY
