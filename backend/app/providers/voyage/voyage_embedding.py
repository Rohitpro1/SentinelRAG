from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import httpx

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.base.embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


VOYAGE_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-lite-01": 512,
    "voyage-01": 1024,
    "voyage-02": 1024,
}


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production Voyage AI Embedding Provider using Voyage REST API.
    Model: voyage-3-large (default 1024 dims).
    Supports connection pooling, 429 rate limit jitter, and transport retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-3-large",
        max_retries: int = 3,
        timeout: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise SentinelRAGError(
                "VoyageEmbeddingProvider requires a non-empty api_key. Provide VOYAGE_API_KEY."
            )
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client
        self._dimensions = VOYAGE_MODEL_DIMENSIONS.get(self.model, 1024)
        self._endpoint = "https://api.voyageai.com/v1/embeddings"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"voyage ({self.model})"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client and not self._client.is_closed:
            return self._client
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        return httpx.AsyncClient(timeout=self.timeout, limits=limits)

    async def embed_query(self, text: str) -> list[float]:
        logger.info("Calling Voyage Embedding API: model='%s', url='%s'", self.model, self._endpoint)
        payload = {
            "model": self.model,
            "input": [text],
            "input_type": "query",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        should_close = self._client is None
        client = self._get_client()

        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        self._endpoint,
                        json=payload,
                        headers=headers,
                    )
                    
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        backoff = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.uniform(0.1, 1.0)
                        logger.warning("Voyage Rate Limit (HTTP 429) on attempt %d/%d. Backing off for %.2fs...", attempt, self.max_retries, backoff)
                        await asyncio.sleep(backoff)
                        continue

                    if response.status_code in (401, 403):
                        raise SentinelRAGError(f"Voyage Authentication failed (HTTP {response.status_code}): {response.text}")

                    response.raise_for_status()
                    data = response.json()
                    data_items = data.get("data", [])
                    if data_items and "embedding" in data_items[0]:
                        return data_items[0]["embedding"]
                    
                    raise EmbeddingError(f"Voyage API returned HTTP 200 OK but embedding payload was empty: {data}")

                except httpx.TimeoutException:
                    backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                    logger.warning("Voyage Transport Timeout (%.1fs) on attempt %d/%d. Retrying in %.2fs...", self.timeout, attempt, self.max_retries, backoff)
                    if attempt < self.max_retries:
                        await asyncio.sleep(backoff)

                except httpx.HTTPStatusError as exc:
                    logger.error("Voyage Embedding HTTP error %d on attempt %d/%d: %s", exc.response.status_code, attempt, self.max_retries, exc)
                    if attempt < self.max_retries and exc.response.status_code >= 500:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))
                    else:
                        raise EmbeddingError(f"Voyage Embedding HTTP error {exc.response.status_code}: {exc}") from exc

                except Exception as exc:
                    if isinstance(exc, (EmbeddingError, SentinelRAGError)):
                        raise
                    logger.error("Voyage Embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                    if attempt < self.max_retries:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))
        finally:
            if should_close and not client.is_closed:
                await client.aclose()

        raise EmbeddingError("Voyage embedding service exhausted retries.")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Calling Voyage Batch Embedding API (%d items): model='%s'", len(texts), self.model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        chunk_size = 128
        all_embeddings: list[list[float]] = []

        should_close = self._client is None
        client = self._get_client()

        try:
            for i in range(0, len(texts), chunk_size):
                batch_texts = texts[i : i + chunk_size]
                payload = {
                    "model": self.model,
                    "input": batch_texts,
                    "input_type": "document",
                }

                batch_success = False
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = await client.post(
                            self._endpoint,
                            json=payload,
                            headers=headers,
                        )
                        if response.status_code == 429:
                            retry_after = response.headers.get("retry-after")
                            backoff = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.uniform(0.1, 1.0)
                            logger.warning("Voyage Batch Rate Limit (HTTP 429) on attempt %d/%d. Backing off for %.2fs...", attempt, self.max_retries, backoff)
                            await asyncio.sleep(backoff)
                            continue

                        if response.status_code in (401, 403):
                            raise SentinelRAGError(f"Voyage Authentication failed (HTTP {response.status_code}): {response.text}")

                        response.raise_for_status()
                        data = response.json()
                        data_items = data.get("data", [])
                        if data_items and len(data_items) == len(batch_texts):
                            for item in data_items:
                                all_embeddings.append(item.get("embedding", []))
                            batch_success = True
                            break

                        logger.warning("Voyage batch payload missing expected items. Falling back to embed_query.")
                        break

                    except httpx.TimeoutException:
                        backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                        logger.warning("Voyage Batch Transport Timeout (%.1fs) on attempt %d/%d. Retrying in %.2fs...", self.timeout, attempt, self.max_retries, backoff)
                        if attempt < self.max_retries:
                            await asyncio.sleep(backoff)

                    except Exception as exc:
                        if isinstance(exc, SentinelRAGError):
                            raise
                        logger.error("Voyage Batch Embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                        if attempt < self.max_retries:
                            await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))

                if not batch_success:
                    for t in batch_texts:
                        vec = await self.embed_query(t)
                        all_embeddings.append(vec)

        finally:
            if should_close and not client.is_closed:
                await client.aclose()

        return all_embeddings
