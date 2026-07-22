"""Unit 2.9 tests -- ContradictionDetector."""
import pytest

from app.schemas.retrieval import Chunk, NLIRelation, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.verification.contradiction_detector import ContradictionDetector
from app.services.verification.nli_deterministic import DeterministicNLIVerifier


def make_ranked_chunk(chunk_id, text):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text=text, token_count=10, source_reliability_score=0.9)
    return RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=0.8, rank=0)


@pytest.mark.asyncio
async def test_no_pairs_for_single_chunk():
    detector = ContradictionDetector(DeterministicNLIVerifier())
    results = await detector.detect([make_ranked_chunk("c1", "text")])
    assert results == []


@pytest.mark.asyncio
async def test_two_chunks_produce_exactly_one_pair():
    detector = ContradictionDetector(DeterministicNLIVerifier())
    results = await detector.detect([make_ranked_chunk("c1", "a"), make_ranked_chunk("c2", "b")])
    assert len(results) == 1
    assert {results[0].chunk_id_a, results[0].chunk_id_b} == {"c1", "c2"}


@pytest.mark.asyncio
async def test_n_chunks_produce_n_choose_2_pairs():
    detector = ContradictionDetector(DeterministicNLIVerifier())
    chunks = [make_ranked_chunk(f"c{i}", f"text {i}") for i in range(5)]
    results = await detector.detect(chunks)
    assert len(results) == 10  # 5 choose 2


@pytest.mark.asyncio
async def test_marker_produces_detectable_contradiction():
    detector = ContradictionDetector(DeterministicNLIVerifier(conflict_marker="[X]"))
    chunks = [make_ranked_chunk("c1", "policy A [X]"), make_ranked_chunk("c2", "policy B [X]")]
    results = await detector.detect(chunks)
    assert results[0].relation == NLIRelation.CONTRADICTION


@pytest.mark.asyncio
async def test_empty_input():
    detector = ContradictionDetector(DeterministicNLIVerifier())
    assert await detector.detect([]) == []


# --- Unit 2.13: graceful degradation when the NLI verifier fails ---

class _AlwaysFailingNLIVerifier:
    async def verify_pair(self, text_a: str, text_b: str):
        raise RuntimeError("provider is down")


class _FlakyNLIVerifier:
    """Fails on the first call, succeeds after."""

    def __init__(self):
        self.calls = 0

    async def verify_pair(self, text_a: str, text_b: str):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient failure")
        return NLIRelation.ENTAILMENT, 0.9


@pytest.mark.asyncio
async def test_failing_verifier_degrades_to_neutral_zero_confidence():
    detector = ContradictionDetector(_AlwaysFailingNLIVerifier())
    results = await detector.detect([make_ranked_chunk("c1", "a"), make_ranked_chunk("c2", "b")])
    assert len(results) == 1
    assert results[0].relation == NLIRelation.NEUTRAL
    assert results[0].confidence == 0.0


@pytest.mark.asyncio
async def test_failing_verifier_does_not_abort_remaining_pairs():
    detector = ContradictionDetector(_AlwaysFailingNLIVerifier())
    chunks = [make_ranked_chunk(f"c{i}", f"text {i}") for i in range(4)]
    results = await detector.detect(chunks)
    assert len(results) == 6  # 4 choose 2 -- every pair still produces a (degraded) result


@pytest.mark.asyncio
async def test_never_raises_regardless_of_verifier_failure():
    detector = ContradictionDetector(_AlwaysFailingNLIVerifier())
    try:
        await detector.detect([make_ranked_chunk("c1", "a"), make_ranked_chunk("c2", "b")])
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"ContradictionDetector.detect() must never raise, but raised: {exc}")


@pytest.mark.asyncio
async def test_partial_failure_only_degrades_the_failing_pair():
    """
    A verifier that fails once then recovers should produce one degraded
    pair and the rest normal -- proving degradation is per-pair, not
    all-or-nothing.
    """
    verifier = _FlakyNLIVerifier()
    detector = ContradictionDetector(verifier)
    chunks = [make_ranked_chunk(f"c{i}", f"text {i}") for i in range(3)]  # 3 pairs
    results = await detector.detect(chunks)
    assert len(results) == 3
    relations = [r.relation for r in results]
    assert relations.count(NLIRelation.NEUTRAL) >= 1  # at least the first (failed) pair
    assert relations.count(NLIRelation.ENTAILMENT) >= 1  # the recovered pairs succeeded normally


@pytest.mark.asyncio
async def test_degradation_is_logged_as_warning(caplog):
    import logging as _logging

    logger = _logging.getLogger("test_contradiction_degradation")
    detector = ContradictionDetector(_AlwaysFailingNLIVerifier(), logger=logger)
    with caplog.at_level(_logging.WARNING, logger="test_contradiction_degradation"):
        await detector.detect([make_ranked_chunk("c1", "a"), make_ranked_chunk("c2", "b")])
    assert any("nli_pair_degraded" in record.message for record in caplog.records)
