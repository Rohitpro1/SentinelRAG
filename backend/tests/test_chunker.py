import pytest
from app.core.exceptions import ChunkingError
from app.core.settings.chunking import ChunkingSettings
from app.services.ingestion.chunker import SemanticChunker, SentenceChunker, TableChunker, split_sentences


def make_chunker(**overrides):
    return SentenceChunker(ChunkingSettings(**overrides))


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three? Yes it is."
    assert split_sentences(text) == ["This is one.", "This is two!", "Is this three?", "Yes it is."]


def test_split_sentences_empty():
    assert split_sentences("   ") == []
    assert split_sentences("") == []


def test_sentence_chunker_respects_target_budget():
    sentence = "The enterprise document contains important information about policy. "
    text = sentence * 30
    chunker = make_chunker(target_tokens=100, min_tokens=20, overlap_tokens=10)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.token_count <= 150


def test_sentence_chunker_no_orphaned_tiny_final_chunk():
    text = (
        "Sentence one is reasonably long and full of content. "
        "Sentence two is also reasonably long and full of content. "
        "Short end."
    )
    chunker = make_chunker(target_tokens=15, min_tokens=10, overlap_tokens=0)
    chunks = chunker.chunk(text)
    assert not any(c.text.strip() == "Short end." for c in chunks)


def test_sentence_chunker_single_short_text_returns_one_chunk():
    chunker = make_chunker()
    chunks = chunker.chunk("Just one short sentence.")
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."


def test_sentence_chunker_empty_text():
    assert make_chunker().chunk("") == []


def test_sentence_chunker_overlap_present_between_chunks():
    sentence = "Policy clause number stated here for testing purposes today. "
    text = sentence * 20
    chunker = make_chunker(target_tokens=60, min_tokens=15, overlap_tokens=20)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    first = set(split_sentences(chunks[0].text))
    second = set(split_sentences(chunks[1].text))
    assert first & second


def test_sentence_chunker_pathological_single_huge_sentence_terminates():
    huge_sentence = "word " * 2000 + "."
    chunker = make_chunker(target_tokens=10, min_tokens=5, overlap_tokens=5)
    chunks = chunker.chunk(huge_sentence)
    assert len(chunks) == 1


def test_table_chunker_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        TableChunker(ChunkingSettings()).chunk("some | table | text")


def test_semantic_chunker_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        SemanticChunker(ChunkingSettings()).chunk("some prose")


def test_sentence_chunker_wraps_internal_failures_as_chunking_error(monkeypatch):
    chunker = make_chunker()
    monkeypatch.setattr(chunker, "_adaptive_chunk", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ChunkingError):
        chunker.chunk("anything")
