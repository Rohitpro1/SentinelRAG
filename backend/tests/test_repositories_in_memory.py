"""
Unit 2.2 tests -- each fake is checked against its own interface contract:
round-trip upsert/search, get/set, invalidate correctness, etc.
"""
import pytest

from app.repositories.fakes.in_memory import (
    InMemoryCacheRepository,
    InMemoryFeedbackRepository,
    InMemoryMetadataRepository,
    InMemoryVectorRepository,
)
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk, SearchRequest, SearchResponse


def make_chunk(chunk_id, document_id="doc-1", reliability=0.9):
    return Chunk(
        chunk_id=chunk_id, document_id=document_id, text=f"content {chunk_id}",
        token_count=10, source_reliability_score=reliability,
    )


# --- InMemoryVectorRepository ---

@pytest.mark.asyncio
async def test_vector_repo_upsert_then_search_round_trip():
    repo = InMemoryVectorRepository()
    chunks = [make_chunk("c1"), make_chunk("c2")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    await repo.upsert(chunks, embeddings)

    results = await repo.search(query_embedding=[1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"  # most similar to the query embedding
    assert results[0].similarity_score > results[1].similarity_score


@pytest.mark.asyncio
async def test_vector_repo_upsert_length_mismatch_raises():
    repo = InMemoryVectorRepository()
    with pytest.raises(ValueError):
        await repo.upsert([make_chunk("c1")], [])


@pytest.mark.asyncio
async def test_vector_repo_search_respects_top_k():
    repo = InMemoryVectorRepository()
    chunks = [make_chunk(f"c{i}") for i in range(10)]
    embeddings = [[float(i), 0.0, 0.0] for i in range(10)]
    await repo.upsert(chunks, embeddings)
    results = await repo.search(query_embedding=[5.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_vector_repo_search_respects_document_filter():
    repo = InMemoryVectorRepository()
    await repo.upsert(
        [make_chunk("c1", document_id="doc-A"), make_chunk("c2", document_id="doc-B")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    results = await repo.search(query_embedding=[1.0, 0.0], top_k=5, document_filter={"document_id": "doc-A"})
    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-A"


@pytest.mark.asyncio
async def test_vector_repo_delete_removes_only_target_document():
    repo = InMemoryVectorRepository()
    await repo.upsert(
        [make_chunk("c1", document_id="doc-A"), make_chunk("c2", document_id="doc-B")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    await repo.delete("doc-A")
    assert repo._count() == 1
    results = await repo.search(query_embedding=[1.0, 0.0], top_k=5)
    assert results[0].chunk.document_id == "doc-B"


@pytest.mark.asyncio
async def test_vector_repo_similarity_score_always_bounded():
    repo = InMemoryVectorRepository()
    await repo.upsert([make_chunk("c1")], [[-1.0, -1.0, -1.0]])  # opposite direction from query
    results = await repo.search(query_embedding=[1.0, 1.0, 1.0], top_k=1)
    assert 0.0 <= results[0].similarity_score <= 1.0


# --- InMemoryMetadataRepository ---

@pytest.mark.asyncio
async def test_metadata_repo_save_and_get_round_trip():
    repo = InMemoryMetadataRepository()
    await repo.save_document_metadata("doc-1", {"filename": "policy.pdf", "fingerprint": "abc123"})
    metadata = await repo.get_document_metadata("doc-1")
    assert metadata["filename"] == "policy.pdf"


@pytest.mark.asyncio
async def test_metadata_repo_missing_document_returns_empty_dict():
    repo = InMemoryMetadataRepository()
    assert await repo.get_document_metadata("does-not-exist") == {}


@pytest.mark.asyncio
async def test_metadata_repo_find_by_fingerprint_detects_duplicate():
    repo = InMemoryMetadataRepository()
    await repo.save_document_metadata("doc-1", {"fingerprint": "abc123"})
    found = await repo.find_by_fingerprint("abc123")
    assert found == "doc-1"


@pytest.mark.asyncio
async def test_metadata_repo_find_by_fingerprint_returns_none_when_absent():
    repo = InMemoryMetadataRepository()
    assert await repo.find_by_fingerprint("nonexistent") is None


@pytest.mark.asyncio
async def test_metadata_repo_ingestion_status_defaults_to_pending_then_updatable():
    repo = InMemoryMetadataRepository()
    await repo.save_document_metadata("doc-1", {})
    assert await repo.get_ingestion_status("doc-1") == "pending"
    repo.set_status("doc-1", "completed")
    assert await repo.get_ingestion_status("doc-1") == "completed"


# --- InMemoryCacheRepository ---

def _search_response(document_id="doc-1"):
    ranked = RankedChunk(
        retrieved_chunk=RetrievedChunk(chunk=make_chunk("c1", document_id=document_id), similarity_score=0.9),
        rerank_score=0.8, rank=0,
    )
    return SearchResponse(request=SearchRequest(query="q"), ranked_chunks=[ranked])


@pytest.mark.asyncio
async def test_cache_repo_get_set_round_trip():
    repo = InMemoryCacheRepository()
    assert await repo.get("key-1") is None
    await repo.set("key-1", _search_response(), ttl_seconds=60)
    cached = await repo.get("key-1")
    assert cached is not None
    assert cached.ranked_chunks[0].retrieved_chunk.chunk.chunk_id == "c1"


@pytest.mark.asyncio
async def test_cache_repo_invalidate_removes_entries_referencing_document():
    repo = InMemoryCacheRepository()
    await repo.set("key-doc-A", _search_response(document_id="doc-A"), ttl_seconds=60)
    await repo.set("key-doc-B", _search_response(document_id="doc-B"), ttl_seconds=60)

    await repo.invalidate("doc-A")

    assert await repo.get("key-doc-A") is None
    assert await repo.get("key-doc-B") is not None  # untouched


@pytest.mark.asyncio
async def test_cache_repo_invalidate_nonexistent_document_is_a_no_op():
    repo = InMemoryCacheRepository()
    await repo.set("key-1", _search_response(), ttl_seconds=60)
    await repo.invalidate("doc-does-not-exist")
    assert await repo.get("key-1") is not None


# --- InMemoryFeedbackRepository ---

@pytest.mark.asyncio
async def test_feedback_repo_record_and_get_round_trip():
    repo = InMemoryFeedbackRepository()
    await repo.record_feedback("query-1", rating=5, comment="great answer")
    feedback = await repo.get_feedback_for_query("query-1")
    assert len(feedback) == 1
    assert feedback[0]["rating"] == 5


@pytest.mark.asyncio
async def test_feedback_repo_accumulates_multiple_entries_per_query():
    repo = InMemoryFeedbackRepository()
    await repo.record_feedback("query-1", rating=5)
    await repo.record_feedback("query-1", rating=1, comment="actually wrong")
    feedback = await repo.get_feedback_for_query("query-1")
    assert len(feedback) == 2


@pytest.mark.asyncio
async def test_feedback_repo_unknown_query_returns_empty_list():
    repo = InMemoryFeedbackRepository()
    assert await repo.get_feedback_for_query("nonexistent") == []
