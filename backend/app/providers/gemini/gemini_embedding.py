import asyncio
import logging
import random
from typing import Optional

import httpx

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.base.embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Production Gemini Embedding Provider using Google Gemini REST API.
    Model: text-embedding-004.
    Supports native batchEmbedContents, bounded connection pooling, 429 rate limit jitter,
    and distinct transport timeout handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        dimensions: int = 768,
        max_retries: int = 3,
        timeout: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise SentinelRAGError(
                "GeminiEmbeddingProvider requires a non-empty api_key. Provide GEMINI_API_KEY."
            )
        self.api_key = api_key.strip()
        self.model = model.strip().removeprefix("models/")
        self._dimensions = dimensions
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client
        self._single_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        self._batch_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self) -> httpx.AsyncClient:
        if self._client and not self._client.is_closed:
            return self._client
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        return httpx.AsyncClient(timeout=self.timeout, limits=limits)

    async def embed_query(self, text: str) -> list[float]:
        logger.info("Calling Gemini Single Embedding API: model='%s', url='%s'", self.model, self._single_endpoint)
        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }

        should_close = self._client is None
        client = self._get_client()

        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        f"{self._single_endpoint}?key={self.api_key}",
                        json=payload,
                    )
                    
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        backoff = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.uniform(0.1, 1.0)
                        logger.warning("Gemini Rate Limit (HTTP 429) on attempt %d/%d. Backing off for %.2fs...", attempt, self.max_retries, backoff)
                        await asyncio.sleep(backoff)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        return values
                    
                    raise EmbeddingError(f"Gemini API returned HTTP 200 OK but embedding values payload was empty: {data}")

                except httpx.TimeoutException:
                    backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                    logger.warning("Gemini Transport Timeout (%.1fs) on attempt %d/%d. Retrying in %.2fs...", self.timeout, attempt, self.max_retries, backoff)
                    if attempt < self.max_retries:
                        await asyncio.sleep(backoff)

                except httpx.HTTPStatusError as exc:
                    logger.error("Gemini Embedding HTTP error %d on attempt %d/%d: %s", exc.response.status_code, attempt, self.max_retries, exc)
                    if attempt < self.max_retries and exc.response.status_code >= 500:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))
                    else:
                        raise EmbeddingError(f"Gemini Embedding HTTP error {exc.response.status_code}: {exc}") from exc

                except Exception as exc:
                    if isinstance(exc, EmbeddingError):
                        raise
                    logger.error("Gemini Embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                    if attempt < self.max_retries:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))
        finally:
            if should_close and not client.is_closed:
                await client.aclose()

        raise EmbeddingError("Gemini embedding service exhausted retries.")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Calling Gemini Batch Embedding API (%d items): model='%s'", len(texts), self.model)
        
        # Sub-batch in chunks of 50 for API limits
        chunk_size = 50
        all_embeddings: list[list[float]] = []

        should_close = self._client is None
        client = self._get_client()

        try:
            for i in range(0, len(texts), chunk_size):
                batch_texts = texts[i : i + chunk_size]
                batch_requests = [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]}
                    }
                    for t in batch_texts
                ]
                payload = {"requests": batch_requests}

                batch_success = False
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = await client.post(
                            f"{self._batch_endpoint}?key={self.api_key}",
                            json=payload,
                        )
                        if response.status_code == 429:
                            retry_after = response.headers.get("retry-after")
                            backoff = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.uniform(0.1, 1.0)
                            logger.warning("Gemini Batch Rate Limit (HTTP 429) on attempt %d/%d. Backing off for %.2fs...", attempt, self.max_retries, backoff)
                            await asyncio.sleep(backoff)
                            continue

                        response.raise_for_status()
                        data = response.json()
                        embeddings_list = data.get("embeddings", [])
                        if embeddings_list and len(embeddings_list) == len(batch_texts):
                            for item in embeddings_list:
                                all_embeddings.append(item.get("values", []))
                            batch_success = True
                            break

                        # If batchEmbedContents payload shape unexpected, log and fallback
                        logger.warning("Gemini batchEmbedContents payload missing expected items. Falling back to embed_query.")
                        break

                    except httpx.TimeoutException:
                        backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                        logger.warning("Gemini Batch Transport Timeout (%.1fs) on attempt %d/%d. Retrying in %.2fs...", self.timeout, attempt, self.max_retries, backoff)
                        if attempt < self.max_retries:
                            await asyncio.sleep(backoff)

                    except Exception as exc:
                        logger.error("Gemini Batch Embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                        if attempt < self.max_retries:
                            await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 1.0))

                if not batch_success:
                    # Fallback sequentially for this batch
                    for t in batch_texts:
                        vec = await self.embed_query(t)
                        all_embeddings.append(vec)

        finally:
            if should_close and not client.is_closed:
                await client.aclose()

        return all_embeddings
