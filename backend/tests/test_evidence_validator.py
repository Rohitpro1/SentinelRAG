"""Unit 2.9 tests -- EvidenceValidator."""
import pytest

from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.verification.evidence_validator import EvidenceValidator


def make_ranked_chunk(chunk_id, text="valid content", token_count=10, rerank_score=0.8):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text=text, token_count=token_count, source_reliability_score=0.9)
    return RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=rerank_score, rank=0)


@pytest.mark.asyncio
async def test_all_valid_evidence_passes_through():
    chunks = [make_ranked_chunk("c1"), make_ranked_chunk("c2")]
    valid, unsupported = await EvidenceValidator().validate(chunks)
    assert len(valid) == 2
    assert unsupported == []


@pytest.mark.asyncio
async def test_empty_text_flagged_as_unsupported():
    chunks = [make_ranked_chunk("c1", text="real content"), make_ranked_chunk("c2", text="   ")]
    valid, unsupported = await EvidenceValidator().validate(chunks)
    assert len(valid) == 1
    assert unsupported == ["c2"]


@pytest.mark.asyncio
async def test_zero_token_count_flagged_as_unsupported():
    chunks = [make_ranked_chunk("c1", token_count=0)]
    valid, unsupported = await EvidenceValidator().validate(chunks)
    assert valid == []
    assert unsupported == ["c1"]


@pytest.mark.asyncio
async def test_empty_input():
    valid, unsupported = await EvidenceValidator().validate([])
    assert valid == []
    assert unsupported == []


@pytest.mark.asyncio
async def test_degraded_rerank_score_none_is_still_valid_evidence():
    chunk = Chunk(chunk_id="c1", document_id="doc-1", text="content", token_count=10, source_reliability_score=0.9)
    ranked = RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=None, rank=0)
    valid, unsupported = await EvidenceValidator().validate([ranked])
    assert len(valid) == 1
    assert unsupported == []
