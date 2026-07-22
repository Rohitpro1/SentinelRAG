from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.retrieval import Decision, NLIRelation
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.base import BaseResponseGenerator
from app.services.verification.nli_base import BaseNLIVerifier


class BaseLLMProvider(BaseResponseGenerator, BaseNLIVerifier, ABC):
    """
    Abstract Base Class for LLM providers in SentinelRAG.
    Combines response generation and NLI verification capabilities.
    """

    @abstractmethod
    async def generate(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence] = None,
        diagnostics: Optional[VerificationDiagnostics] = None,
        query: str = "",
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        raise NotImplementedError
