from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from app.core.exceptions import IngestionError
from app.core.logging import get_logger, log_event
from app.repositories.interfaces import MetadataRepository, VectorRepository
from app.schemas.retrieval import Chunk
from app.services.embedding.base import BaseEmbedder
from app.services.ingestion.chunker import BaseChunker
from app.services.ingestion.parsers.factory import DocumentParserFactory

logger = get_logger(__name__)


class IngestionService:
    """
    Production Ingestion Pipeline Service.
    Handles document validation, parsing, adaptive chunking, vector embedding,
    Qdrant storage, duplicate detection, and metadata tracking.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_repo: VectorRepository,
        metadata_repo: MetadataRepository,
        max_file_size_mb: int = 50,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._metadata_repo = metadata_repo
        self._max_file_bytes = max_file_size_mb * 1024 * 1024

    async def ingest_document(
        self, filename: str, content: bytes, is_async: bool = False
    ) -> dict[str, Any]:
        """
        Main ingestion entrypoint.
        """
        if not content:
            raise IngestionError(f"File {filename} is empty.")

        if len(content) > self._max_file_bytes:
            raise IngestionError(
                f"File {filename} exceeds maximum allowed size of {self._max_file_bytes // (1024*1024)} MB."
            )

        fingerprint = hashlib.sha256(content).hexdigest()

        # Check for duplicate upload
        existing_doc_id = await self._metadata_repo.find_by_fingerprint(fingerprint)
        if existing_doc_id:
            logger.info("Duplicate document detected for fingerprint %s (Doc ID: %s)", fingerprint, existing_doc_id)
            meta = await self._metadata_repo.get_document_metadata(existing_doc_id)
            return {
                "status": "already_exists",
                "document_id": existing_doc_id,
                "filename": filename,
                "chunks_created": meta.get("chunks_created", 0),
            }

        document_id = f"doc-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.utcnow().isoformat()

        initial_meta = {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": 0,
            "file_size_bytes": len(content),
            "uploaded_at": now_iso,
            "status": "processing" if is_async else "pending",
            "fingerprint": fingerprint,
        }
        await self._metadata_repo.save_document_metadata(document_id, initial_meta)

        if is_async:
            # Fire background processing task
            asyncio.create_task(self._process_ingestion(document_id, filename, content, fingerprint))
            return {
                "status": "processing",
                "document_id": document_id,
                "filename": filename,
                "chunks_created": 0,
            }

        return await self._process_ingestion(document_id, filename, content, fingerprint)

    async def _process_ingestion(
        self, document_id: str, filename: str, content: bytes, fingerprint: str
    ) -> dict[str, Any]:
        start_time = datetime.utcnow()
        try:
            log_event(logger, "ingestion_started", document_id=document_id, filename=filename)

            # 1. Parse Document
            parser = DocumentParserFactory.get_parser(filename)
            parsed_doc = await parser.parse(content, filename)
            log_event(logger, "parsing_completed", document_id=document_id, length=len(parsed_doc.text))

            # 2. Adaptive Chunking
            text_chunks = self._chunker.chunk(parsed_doc.text)
            if not text_chunks:
                raise IngestionError(f"No valid text chunks created from document {filename}.")

            log_event(logger, "chunking_completed", document_id=document_id, chunks=len(text_chunks))

            # Convert to domain Chunks
            domain_chunks: list[Chunk] = []
            for idx, tc in enumerate(text_chunks):
                chunk_id = f"{document_id}-c{idx}"
                domain_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=tc.text,
                        token_count=tc.token_count,
                        source_reliability_score=1.0,
                        metadata={
                            "filename": filename,
                            "page": 1,
                            "sections": parsed_doc.sections[:3],
                        },
                    )
                )

            # 3. Batch Vector Embedding
            texts_to_embed = [c.text for c in domain_chunks]
            embeddings = await self._embedder.embed_batch(texts_to_embed)
            log_event(logger, "embedding_completed", document_id=document_id, count=len(embeddings))

            # 4. Store in Qdrant Vector Repository
            await self._vector_repo.upsert(domain_chunks, embeddings)
            log_event(logger, "qdrant_insertion_completed", document_id=document_id, points=len(domain_chunks))

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # 5. Update Metadata Repository
            final_meta = {
                "document_id": document_id,
                "filename": filename,
                "chunks_created": len(domain_chunks),
                "file_size_bytes": len(content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "status": "completed",
                "fingerprint": fingerprint,
                "ingestion_duration_ms": round(duration_ms, 2),
            }
            await self._metadata_repo.save_document_metadata(document_id, final_meta)

            return {
                "document_id": document_id,
                "filename": filename,
                "chunks_created": len(domain_chunks),
                "status": "completed",
            }

        except Exception as exc:
            logger.error("Ingestion failed for document %s (%s): %s", document_id, filename, exc, exc_info=True)
            failed_meta = {
                "document_id": document_id,
                "filename": filename,
                "chunks_created": 0,
                "file_size_bytes": len(content),
                "uploaded_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error_message": str(exc),
                "fingerprint": fingerprint,
            }
            await self._metadata_repo.save_document_metadata(document_id, failed_meta)
            if not isinstance(exc, IngestionError):
                raise IngestionError(f"Ingestion failed for {filename}: {exc}") from exc
            raise

    async def list_documents(self) -> list[dict[str, Any]]:
        # For in-memory repository or metadata repo that tracks documents
        if hasattr(self._metadata_repo, "_metadata"):
            return list(getattr(self._metadata_repo, "_metadata").values())
        return []

    async def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        meta = await self._metadata_repo.get_document_metadata(document_id)
        return meta if meta else None

    async def delete_document(self, document_id: str) -> bool:
        meta = await self._metadata_repo.get_document_metadata(document_id)
        if not meta:
            return False

        await self._vector_repo.delete(document_id)
        if hasattr(self._metadata_repo, "_metadata"):
            getattr(self._metadata_repo, "_metadata").pop(document_id, None)
            fingerprint = meta.get("fingerprint")
            if fingerprint and hasattr(self._metadata_repo, "_fingerprints"):
                getattr(self._metadata_repo, "_fingerprints").pop(fingerprint, None)
        return True
