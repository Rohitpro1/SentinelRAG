from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.base.embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production Gemini Embedding Provider using Google Gemini REST API.
    Model: text-embedding-004.
    Includes retries, exponential backoff, and rate-limit handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        dimensions: int = 768,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise SentinelRAGError(
                "GeminiEmbeddingProvider requires a non-empty api_key. Provide GEMINI_API_KEY."
            )
        self.api_key = api_key.strip()
        # Clean model name if passed with 'models/' prefix
        self.model = model.strip().removeprefix("models/")
        self._dimensions = dimensions
        self.max_retries = max_retries
        self.timeout = timeout
        self._endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> list[float]:
        logger.info("Calling Gemini Embedding API: model='%s', url='%s'", self.model, self._endpoint)
        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
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
                        logger.warning("Gemini rate limit (429) encountered. Retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        return values
            except Exception as exc:
                logger.error("Gemini embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        raise EmbeddingError("Gemini embedding service exhausted retries.")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = await self.embed_query(text)
            results.append(vec)
        return results
