from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Optional

import httpx

from app.core.exceptions import SentinelRAGError
from app.providers.base.llm_provider import BaseLLMProvider
from app.schemas.retrieval import Decision, NLIRelation
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.services.response_generation.service import ResponseGenerator
from app.services.verification.nli_deterministic import DeterministicNLIVerifier

logger = logging.getLogger(__name__)


def resolve_gemini_model_name(model_name: str) -> str:
    return model_name.strip().removeprefix("models/")


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
        raw_model = model.strip()
        self.model = resolve_gemini_model_name(raw_model)
        self.max_retries = max_retries
        self.timeout = timeout
        self._fallback_generator = ResponseGenerator()
        self._fallback_nli = DeterministicNLIVerifier()
        self._endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        # Request-Scoped Prompt Cache & Telemetry Metrics
        self._cache: dict[str, str] = {}
        self._metrics: dict[str, Any] = {
            "llm_calls_per_query": 0,
            "llm_tokens_per_query": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_llm_latency": 0.0,
            "_total_latency": 0.0,
        }

        logger.info(
            "GeminiLLMProvider initialized: resolved_model='%s' (raw='%s'), client='raw REST API (httpx.AsyncClient)', api_version='v1beta', endpoint='%s'",
            self.model,
            raw_model,
            self._endpoint,
        )

    def get_metrics(self) -> dict[str, Any]:
        res = dict(self._metrics)
        res.pop("_total_latency", None)
        return res

    def reset_metrics(self) -> None:
        self._cache.clear()
        self._metrics["llm_calls_per_query"] = 0
        self._metrics["llm_tokens_per_query"] = 0
        self._metrics["prompt_tokens"] = 0
        self._metrics["completion_tokens"] = 0
        self._metrics["cache_hits"] = 0
        self._metrics["cache_misses"] = 0
        self._metrics["average_llm_latency"] = 0.0
        self._metrics["_total_latency"] = 0.0

    async def _call_gemini(self, prompt: str) -> Optional[str]:
        # Request-scoped prompt cache check
        if prompt in self._cache:
            self._metrics["cache_hits"] += 1
            logger.info("Gemini LLM Prompt Cache HIT (0ms).")
            return self._cache[prompt]

        self._metrics["cache_misses"] += 1
        logger.info("Calling Gemini LLM API: model='%s', url='%s'", self.model, self._endpoint)

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

        start_time = asyncio.get_event_loop().time()
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self._endpoint}?key={self.api_key}",
                        json=payload,
                    )
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        backoff = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.uniform(0.1, 1.0)
                        logger.warning("Gemini LLM Rate Limit (HTTP 429) on attempt %d/%d. Backing off for %.2fs...", attempt, self.max_retries, backoff)
                        await asyncio.sleep(backoff)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            result_text = parts[0].get("text", "").strip()
                            self._cache[prompt] = result_text
                            
                            # Update Telemetry Metrics
                            elapsed = asyncio.get_event_loop().time() - start_time
                            self._metrics["llm_calls_per_query"] += 1
                            self._metrics["_total_latency"] += elapsed
                            self._metrics["average_llm_latency"] = round(self._metrics["_total_latency"] / self._metrics["llm_calls_per_query"], 3)
                            
                            p_tokens = len(prompt.split())
                            c_tokens = len(result_text.split())
                            self._metrics["prompt_tokens"] += p_tokens
                            self._metrics["completion_tokens"] += c_tokens
                            self._metrics["llm_tokens_per_query"] += (p_tokens + c_tokens)
                            
                            return result_text
            except httpx.TimeoutException:
                backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                logger.warning("Gemini LLM Transport Timeout (%.1fs) on attempt %d/%d. Retrying in %.2fs...", self.timeout, attempt, self.max_retries, backoff)
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff)
            except Exception as exc:
                logger.error("Gemini LLM attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))

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
