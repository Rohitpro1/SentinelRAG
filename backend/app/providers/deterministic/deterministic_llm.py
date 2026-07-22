from __future__ import annotations

from typing import Optional

from app.providers.base.llm_provider import BaseLLMProvider
from app.schemas.retrieval import Decision, NLIRelation
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.service import ResponseGenerator
from app.services.verification.nli_deterministic import DeterministicNLIVerifier


class DeterministicLLMProvider(BaseLLMProvider):
    """
    Deterministic LLM provider combining ResponseGenerator and DeterministicNLIVerifier.
    Used for local testing and CI without external network/LLM dependencies.
    """

    def __init__(self) -> None:
        self._generator = ResponseGenerator()
        self._nli = DeterministicNLIVerifier()

    async def generate(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence] = None,
        diagnostics: Optional[VerificationDiagnostics] = None,
        query: str = "",
    ) -> str:
        return await self._generator.generate(
            decision=decision,
            evidence=evidence,
            diagnostics=diagnostics,
            query=query,
        )

    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        return await self._nli.verify_pair(text_a, text_b)
