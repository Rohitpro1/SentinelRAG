"""
Chunking service — interface-based design.

BaseChunker
  -> SentenceChunker   (implemented: sentence-boundary + token budget)
  -> TableChunker      (future — NOT implemented)
  -> SemanticChunker   (future — NOT implemented)

Milestone 2's Retriever/Embedding services should depend on BaseChunker,
not SentenceChunker directly, so swapping chunkers by document type is
a DI-wiring change, not a calling-code change.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.exceptions import ChunkingError
from app.core.settings.chunking import ChunkingSettings

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class TextChunk:
    text: str
    token_count: int
    sentence_start_idx: int
    sentence_end_idx: int


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[TextChunk]:
        raise NotImplementedError


class SentenceChunker(BaseChunker):
    """Sentence-boundary + token-budget chunker (Milestone 1 logic behind the interface)."""

    def __init__(self, settings: ChunkingSettings):
        self._settings = settings

    def chunk(self, text: str) -> list[TextChunk]:
        try:
            return self._adaptive_chunk(text)
        except Exception as exc:
            raise ChunkingError(f"SentenceChunker failed to chunk input text: {exc}") from exc

    def _adaptive_chunk(self, text: str) -> list[TextChunk]:
        s = self._settings
        sentences = split_sentences(text)
        if not sentences:
            return []

        sentence_tokens = [_approx_token_count(sent) for sent in sentences]
        chunks: list[TextChunk] = []
        start = 0
        n = len(sentences)

        while start < n:
            end = start
            token_sum = 0
            while end < n and token_sum < s.target_tokens:
                token_sum += sentence_tokens[end]
                end += 1

            if end == n and chunks:
                remaining_tokens = sum(sentence_tokens[start:end])
                if remaining_tokens < s.min_tokens:
                    prev = chunks.pop()
                    merged_text = prev.text + " " + " ".join(sentences[start:end])
                    chunks.append(
                        TextChunk(
                            text=merged_text,
                            token_count=_approx_token_count(merged_text),
                            sentence_start_idx=prev.sentence_start_idx,
                            sentence_end_idx=end,
                        )
                    )
                    break

            chunk_text = " ".join(sentences[start:end])
            chunks.append(
                TextChunk(text=chunk_text, token_count=token_sum, sentence_start_idx=start, sentence_end_idx=end)
            )

            if end >= n:
                break

            overlap_start = end
            overlap_sum = 0
            while overlap_start > start and overlap_sum < s.overlap_tokens:
                overlap_start -= 1
                overlap_sum += sentence_tokens[overlap_start]

            start = overlap_start if overlap_start > start else end

        return chunks


class TableChunker(BaseChunker):
    """NOT IMPLEMENTED — planned for Milestone 2's table-extraction path."""

    def __init__(self, settings: ChunkingSettings):
        self._settings = settings

    def chunk(self, text: str) -> list[TextChunk]:
        raise NotImplementedError("TableChunker is not implemented yet (planned for Milestone 2).")


class SemanticChunker(BaseChunker):
    """NOT IMPLEMENTED — needs the Embedding Service (Milestone 2) as a dependency first."""

    def __init__(self, settings: ChunkingSettings):
        self._settings = settings

    def chunk(self, text: str) -> list[TextChunk]:
        raise NotImplementedError("SemanticChunker is not implemented yet (planned post-Milestone 2).")
