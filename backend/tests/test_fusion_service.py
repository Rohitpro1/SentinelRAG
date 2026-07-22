"""Unit 2.6 tests -- FusionService: dedup-by-chunk_id + descending sort."""
import pytest

from app.schemas.retrieval import Chunk, RetrievedChunk
from app.services.retrieval.fusion_service import FusionService


def make_rc(chunk_id, similarity):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text="x", token_count=5, source_reliability_score=0.9)
    return RetrievedChunk(chunk=chunk, similarity_score=similarity)


@pytest.mark.asyncio
async def test_sorts_descending_by_similarity():
    results = [make_rc("c1", 0.3), make_rc("c2", 0.9), make_rc("c3", 0.6)]
    fused = await FusionService().fuse(results)
    assert [rc.chunk.chunk_id for rc in fused] == ["c2", "c3", "c1"]


@pytest.mark.asyncio
async def test_dedups_by_chunk_id_keeping_highest_similarity():
    results = [make_rc("c1", 0.3), make_rc("c1", 0.8), make_rc("c1", 0.5)]
    fused = await FusionService().fuse(results)
    assert len(fused) == 1
    assert fused[0].similarity_score == 0.8


@pytest.mark.asyncio
async def test_empty_input():
    assert await FusionService().fuse([]) == []


@pytest.mark.asyncio
async def test_mixed_dedup_and_sort():
    results = [make_rc("c1", 0.2), make_rc("c2", 0.9), make_rc("c1", 0.7), make_rc("c3", 0.5)]
    fused = await FusionService().fuse(results)
    assert [rc.chunk.chunk_id for rc in fused] == ["c2", "c1", "c3"]
    assert fused[1].similarity_score == 0.7
