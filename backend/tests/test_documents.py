import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion.parsers.docx_parser import DOCXDocumentParser
from app.services.ingestion.parsers.factory import DocumentParserFactory
from app.services.ingestion.parsers.md_parser import MarkdownDocumentParser
from app.services.ingestion.parsers.pdf_parser import PDFDocumentParser
from app.services.ingestion.parsers.txt_parser import TextDocumentParser
from app.core.exceptions import IngestionError

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer dev-token"}


@pytest.mark.asyncio
async def test_text_parser():
    parser = TextDocumentParser()
    parsed = await parser.parse(b"Hello world. This is a text file.", "test.txt")
    assert "Hello world" in parsed.text
    assert parsed.metadata["format"] == "txt"


@pytest.mark.asyncio
async def test_markdown_parser():
    parser = MarkdownDocumentParser()
    content = b"# Title\n\nSection 1 text.\n\n## Subtitle\nSection 2 text."
    parsed = await parser.parse(content, "guide.md")
    assert "Section 1 text" in parsed.text
    assert "Title" in parsed.sections


@pytest.mark.asyncio
async def test_pdf_parser_valid():
    parser = PDFDocumentParser()
    content = b"%PDF-1.4 (fake pdf text block (Sample PDF Document Content) ET)"
    parsed = await parser.parse(content, "sample.pdf")
    assert parsed.metadata["format"] == "pdf"


@pytest.mark.asyncio
async def test_pdf_parser_encrypted():
    parser = PDFDocumentParser()
    content = b"%PDF-1.4 /Encrypt 1 0 R"
    with pytest.raises(IngestionError, match="encrypted"):
        await parser.parse(content, "locked.pdf")


@pytest.mark.asyncio
async def test_empty_file_error():
    parser = TextDocumentParser()
    with pytest.raises(IngestionError, match="empty"):
        await parser.parse(b"", "empty.txt")


def test_parser_factory_unsupported():
    with pytest.raises(IngestionError, match="Unsupported file format"):
        DocumentParserFactory.get_parser("image.png")


def test_api_upload_text_document():
    files = {"file": ("policy.txt", b"Refund policy detail number 1. Full refund within 30 days.", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files, headers=AUTH_HEADERS)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "policy.txt"
    assert data["status"] == "completed"
    assert data["chunks_created"] > 0
    doc_id = data["document_id"]

    # Retrieve document detail
    detail_res = client.get(f"/api/v1/documents/{doc_id}", headers=AUTH_HEADERS)
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["document_id"] == doc_id
    assert detail_data["filename"] == "policy.txt"

    # Status check
    status_res = client.get(f"/api/v1/documents/{doc_id}/status", headers=AUTH_HEADERS)
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "completed"

    # List documents
    list_res = client.get("/api/v1/documents", headers=AUTH_HEADERS)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1


def test_api_duplicate_upload_detection():
    content = b"Unique content for duplicate detection test."
    files1 = {"file": ("dup.txt", content, "text/plain")}
    res1 = client.post("/api/v1/documents/upload", files=files1, headers=AUTH_HEADERS)
    assert res1.status_code == 201
    data1 = res1.json()

    # Upload identical content
    files2 = {"file": ("dup.txt", content, "text/plain")}
    res2 = client.post("/api/v1/documents/upload", files=files2, headers=AUTH_HEADERS)
    assert res2.status_code == 201
    data2 = res2.json()

    assert data2["status"] == "already_exists"
    assert data2["document_id"] == data1["document_id"]


def test_api_delete_document():
    files = {"file": ("to_delete.txt", b"Delete me soon.", "text/plain")}
    res = client.post("/api/v1/documents/upload", files=files, headers=AUTH_HEADERS)
    doc_id = res.json()["document_id"]

    del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=AUTH_HEADERS)
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Subsequent GET should return 404
    get_res = client.get(f"/api/v1/documents/{doc_id}", headers=AUTH_HEADERS)
    assert get_res.status_code == 404
