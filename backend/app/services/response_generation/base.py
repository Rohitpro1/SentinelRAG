"""
Unit 3.9 -- BaseResponseGenerator interface.

Owned by the Response Generation domain.
ResponseGenerationNode depends on this interface, never directly on a concrete
implementation -- same Dependency Inversion pattern as BaseEmbedder/BaseReranker/BaseNLIVerifier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.retrieval import Decision
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence


class BaseResponseGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence] = None,
        diagnostics: Optional[VerificationDiagnostics] = None,
        query: str = "",
    ) -> str:
        """
        Generate natural language answer based on decision, verified evidence,
        diagnostics, and query context.
        """
        raise NotImplementedError
