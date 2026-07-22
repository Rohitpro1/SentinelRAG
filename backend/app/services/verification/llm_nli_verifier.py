"""
Unit 2.13 -- LLMBasedNLIVerifier: first real NLI provider.

SCOPE DECISION, stated explicitly: unlike embeddings (OpenAI's
/embeddings) and reranking (Cohere-style /rerank), there is no widely
adopted single-vendor REST convention for pairwise NLI classification as
a hosted service. The realistic real-world approach -- and what this
class implements -- is LLM-as-judge: an OpenAI-compatible chat-
completions call (POST {base_url}/chat/completions), prompting the model
to classify two texts as entailment/contradiction/neutral with a
confidence, at temperature=0 for the most determinism a hosted LLM API
can offer (still not bit-for-bit deterministic across calls or providers
-- unlike DeterministicNLIVerifier, which is exact). Configurable
base_url covers OpenAI itself and any OpenAI-compatible chat endpoint
(Azure OpenAI, local vLLM/Ollama chat servers), same scope decision as
Units 2.11/2.12.

NETWORK NOTE: identical situation to Units 2.11/2.12 -- no route to any
LLM provider from this sandbox. All logic here is tested via
httpx.MockTransport; the isolated integration test skips cleanly without
a configured live endpoint.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from app.core.exceptions import VerificationError
from app.core.logging import get_logger, log_event
from app.core.settings.nli import NLISettings
from app.schemas.embedding import EmbedderHealth, EmbedderHealthState
from app.schemas.nli import NLIResult
from app.schemas.retrieval import NLIRelation
from app.services.verification.nli_base import BaseNLIVerifier
from app.services.verification.nli_result_builder import build_nli_result

_SYSTEM_PROMPT = (
    "You are a natural language inference classifier. Given two texts, "
    "classify their relationship as exactly one of: entailment, contradiction, "
    "neutral. Respond with ONLY a JSON object: "
    '{"label": "<entailment|contradiction|neutral>", "confidence": <0.0-1.0>}. '
    "No other text."
)

_LABEL_MAP = {
    "entailment": NLIRelation.ENTAILMENT,
    "contradiction": NLIRelation.CONTRADICTION,
    "neutral": NLIRelation.NEUTRAL,
}


class LLMBasedNLIVerifier(BaseNLIVerifier):
    def __init__(self, client: httpx.AsyncClient, settings: NLISettings, logger: Optional[logging.Logger] = None):
        # DI per the established pattern: client constructed exclusively
        # by app.infrastructure.nli_client_factory.create_nli_http_client()
        # and injected here.
        self._client = client
        self._settings = settings
        self._logger = logger or get_logger(__name__)
        self._consecutive_failures = 0

    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        # No internal retry -- ContradictionDetector (Unit 2.13 refinement
        # to Unit 2.9) now degrades any verify_pair failure to (NEUTRAL,
        # 0.0) with structured logging, per instruction 5. Retrying here
        # would only delay that degradation, same reasoning as Unit
        # 2.12's reranker.
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._settings.model_name,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": f"Text A: {text_a}\n\nText B: {text_b}"},
                    ],
                },
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            log_event(
                self._logger, "nli_request_failed", level=logging.WARNING,
                error=str(exc), error_type=type(exc).__name__, consecutive_failures=self._consecutive_failures,
            )
            raise VerificationError(f"NLI request failed: {exc}") from exc

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            label = _LABEL_MAP[parsed["label"].lower().strip()]
            confidence = max(0.0, min(1.0, float(parsed["confidence"])))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._consecutive_failures += 1
            log_event(
                self._logger, "nli_response_malformed", level=logging.WARNING,
                error=str(exc), consecutive_failures=self._consecutive_failures,
            )
            raise VerificationError(f"Malformed NLI response: {exc}") from exc

        self._consecutive_failures = 0
        log_event(self._logger, "nli_request_succeeded", label=label.value, model=self._settings.model_name)
        return label, confidence

    async def verify_pair_with_result(self, text_a: str, text_b: str) -> NLIResult:
        return await build_nli_result(
            lambda: self.verify_pair(text_a, text_b),
            provider="llm_chat_completions",
            model_name=self._settings.model_name,
        )

    def health(self) -> EmbedderHealth:
        """Reuses EmbedderHealth (instruction 2) -- same failure-tracking pattern as OpenAIEmbedder."""
        threshold = self._settings.unavailable_after_consecutive_failures
        if self._consecutive_failures == 0:
            return EmbedderHealth(state=EmbedderHealthState.READY)
        if self._consecutive_failures < threshold:
            return EmbedderHealth(
                state=EmbedderHealthState.DEGRADED, detail=f"{self._consecutive_failures} consecutive failure(s)"
            )
        return EmbedderHealth(
            state=EmbedderHealthState.UNAVAILABLE,
            detail=f"{self._consecutive_failures} consecutive failures (threshold={threshold})",
        )
