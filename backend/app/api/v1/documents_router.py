from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_current_principal, get_ingestion_service
from app.api.v1.document_schemas import (
    DocumentDetailResponseBody,
    DocumentListItem,
    DocumentStatusResponseBody,
    DocumentUploadResponseBody,
)
from app.core.exceptions import IngestionError
from app.services.ingestion.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponseBody, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    is_async: bool = False,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    _principal: str = Depends(get_current_principal),
) -> DocumentUploadResponseBody:
    """
    Upload knowledge document (PDF, DOCX, TXT, Markdown) for processing and vector storage.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in upload.")

    try:
        content = await file.read()
        res = await ingestion_service.ingest_document(
            filename=file.filename, content=content, is_async=is_async
        )
        return DocumentUploadResponseBody(
            document_id=res["document_id"],
            filename=res["filename"],
            chunks_created=res.get("chunks_created", 0),
            status=res["status"],
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    _principal: str = Depends(get_current_principal),
) -> list[DocumentListItem]:
    """
    List all ingested documents.
    """
    docs = await ingestion_service.list_documents()
    items: list[DocumentListItem] = []
    for doc in docs:
        items.append(
            DocumentListItem(
                document_id=doc["document_id"],
                filename=doc["filename"],
                chunks_created=doc.get("chunks_created", 0),
                uploaded_at=doc.get("uploaded_at", ""),
                status=doc.get("status", "unknown"),
            )
        )
    return items


@router.get("/{document_id}", response_model=DocumentDetailResponseBody)
async def get_document_details(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    _principal: str = Depends(get_current_principal),
) -> DocumentDetailResponseBody:
    """
    Get detailed metadata for a specific document.
    """
    doc = await ingestion_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return DocumentDetailResponseBody(
        document_id=doc["document_id"],
        filename=doc["filename"],
        chunks_created=doc.get("chunks_created", 0),
        file_size_bytes=doc.get("file_size_bytes", 0),
        uploaded_at=doc.get("uploaded_at", ""),
        status=doc.get("status", "unknown"),
        fingerprint=doc.get("fingerprint"),
        error_message=doc.get("error_message"),
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponseBody)
async def get_document_status(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    _principal: str = Depends(get_current_principal),
) -> DocumentStatusResponseBody:
    """
    Poll ingestion status for a document.
    """
    doc = await ingestion_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return DocumentStatusResponseBody(
        document_id=doc["document_id"],
        status=doc.get("status", "unknown"),
        error_message=doc.get("error_message"),
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    _principal: str = Depends(get_current_principal),
) -> dict[str, Any]:
    """
    Delete document metadata and remove vector chunks from Qdrant storage.
    """
    success = await ingestion_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"status": "deleted", "document_id": document_id}
