from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DocumentUploadResponseBody(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    chunks: int = Field(alias="chunks_created", default=0)
    uploaded_at: str
    status: str


class DocumentDetailResponseBody(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    file_size_bytes: int
    uploaded_at: str
    status: str
    fingerprint: Optional[str] = None
    error_message: Optional[str] = None


class DocumentStatusResponseBody(BaseModel):
    document_id: str
    status: str
    error_message: Optional[str] = None
