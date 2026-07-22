from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx

from app.core.exceptions import SentinelRAGError
from app.providers.base.llm_provider import BaseLLMProvider
from app.schemas.retrieval import Decision, NLIRelation
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.service import ResponseGenerator
from app.services.verification.nli_deterministic import DeterministicNLIVerifier

logger = logging.getLogger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Production Gemini LLM Provider using Google Gemini REST API.
    Model: gemini-2.5-flash.
    Handles grounded response generation and NLI verification with retries & backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        timeout: float = 15.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise SentinelRAGError(
                "GeminiLLMProvider requires a non-empty api_key. Provide GEMINI_API_KEY."
            )
        self.api_key = api_key.strip()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._fallback_generator = ResponseGenerator()
        self._fallback_nli = DeterministicNLIVerifier()
        self._endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    async def _call_gemini(self, prompt: str) -> Optional[str]:

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
            }
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self._endpoint}?key={self.api_key}",
                        json=payload,
                    )
                    if response.status_code == 429:
                        logger.warning("Gemini rate limit (429) hit during text gen. Retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception as exc:
                logger.error("Gemini LLM attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        return None

    async def generate(
        self,
        decision: Decision,
        evidence: Optional[VerifiedEvidence] = None,
        diagnostics: Optional[VerificationDiagnostics] = None,
        query: str = "",
    ) -> str:
        prompt = (
            f"You are SentinelRAG's response generation model.\n"
            f"User Query: {query}\n"
            f"Decision Action: {decision.action.value}\n"
            f"Decision Reasons: {', '.join(decision.reasons)}\n"
        )
        if evidence and evidence.retrieved_chunks:
            chunks_text = "\n".join([f"- {rc.chunk.text}" for rc in evidence.retrieved_chunks])
            prompt += f"Verified Evidence:\n{chunks_text}\n"

        prompt += "\nFormulate a clear, grounded natural-language answer strictly matching the verified evidence."

        generated = await self._call_gemini(prompt)
        if generated:
            return generated

        logger.warning("Gemini response generation unavailable. Using fallback response generator.")
        return await self._fallback_generator.generate(decision, evidence, diagnostics, query)

    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        prompt = (
            f"Determine the Natural Language Inference (NLI) relationship between Premises and Hypothesis.\n"
            f"Premise: {text_a}\n"
            f"Hypothesis: {text_b}\n"
            f"Respond in JSON format with fields: \"relation\" (one of ENTAILMENT, CONTRADICTION, NEUTRAL) and \"confidence\" (float 0.0 to 1.0)."
        )

        res_text = await self._call_gemini(prompt)
        if res_text:
            try:
                # Extract JSON from response
                start = res_text.find("{")
                end = res_text.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(res_text[start : end + 1])
                    rel_str = str(parsed.get("relation", "NEUTRAL")).upper()
                    conf = float(parsed.get("confidence", 0.8))

                    if rel_str == "ENTAILMENT":
                        return NLIRelation.ENTAILMENT, conf
                    elif rel_str == "CONTRADICTION":
                        return NLIRelation.CONTRADICTION, conf
                    else:
                        return NLIRelation.NEUTRAL, conf
            except Exception as exc:
                logger.error("Error parsing Gemini NLI JSON response: %s", exc)

        return await self._fallback_nli.verify_pair(text_a, text_b)
