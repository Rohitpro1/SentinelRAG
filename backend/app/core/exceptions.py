"""
Typed exception hierarchy — one root, one subclass per subsystem.
Lets API handlers map exception type -> HTTP status without string
matching, and lets tests assert on failure *kind* via pytest.raises().
"""
from __future__ import annotations
from typing import Any, Optional


class SentinelRAGError(Exception):
    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class DecisionEngineError(SentinelRAGError):
    """Raised when the Decision Engine receives invalid input or cannot route a decision."""


class ChunkingError(SentinelRAGError):
    """Raised when a chunker cannot process input text."""


class RetrievalError(SentinelRAGError):
    """
    Raised when the Retriever Agent fails to fetch candidates.
    `transient=True` marks errors eligible for agent-level retry
    (Retrieval Domain Design, Section 1); `transient=False` marks caller
    errors (e.g. malformed request) that must never be retried.
    """

    def __init__(self, message: str, *, transient: bool, context: Optional[dict[str, Any]] = None):
        super().__init__(message, context=context)
        self.transient = transient


class EmbeddingError(SentinelRAGError):
    """Embedding service call failures. Always treated as transient (Architecture Enhancements, Section 5)."""

    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None):
        super().__init__(message, context=context)
        self.transient = True


class VerificationError(SentinelRAGError):
    """Raised when the Verification Agent's NLI/consistency checks fail (Milestone 3)."""


class RerankError(SentinelRAGError):
    """
    Cross-encoder rerank failures. Per Retrieval Domain Design Section 1,
    RetrieverAgent/RerankingService catches this internally and degrades
    to un-reranked results -- it should rarely, if ever, escape to a caller.
    """


class SecurityError(SentinelRAGError):
    """Raised on auth, prompt-injection-detection, or PII-handling failures (Milestone 3)."""
