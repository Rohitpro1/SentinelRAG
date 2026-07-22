from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from app.providers.base.embedding_provider import BaseEmbeddingProvider
from app.services.embedding.deterministic import DeterministicEmbedder

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production Gemini Embedding Provider using Google Gemini REST API.
    Model: gemini-embedding-001.
    Includes retries, exponential backoff, rate-limit handling, and fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._dimensions = dimensions
        self.max_retries = max_retries
        self.timeout = timeout
        self._fallback = DeterministicEmbedder(dimensions=dimensions)
        self._endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> list[float]:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Falling back to deterministic embedder.")
            return await self._fallback.embed_query(text)

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

        logger.error("Gemini embedding exhausted retries. Falling back to deterministic embedder.")
        return await self._fallback.embed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vec = await self.embed_query(text)
            results.append(vec)
        return results
