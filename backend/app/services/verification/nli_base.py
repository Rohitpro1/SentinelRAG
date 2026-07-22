"""
Unit 2.9 -- BaseNLIVerifier interface.

Owned by the Verification domain (sibling to Retrieval and Embedding).
ContradictionDetector depends on this interface, never on a concrete NLI
model -- same Dependency Inversion pattern as BaseEmbedder/BaseReranker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.retrieval import NLIRelation


class BaseNLIVerifier(ABC):
    @abstractmethod
    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        """Return the NLI relation between two texts and a confidence in [0, 1]."""
        raise NotImplementedError
