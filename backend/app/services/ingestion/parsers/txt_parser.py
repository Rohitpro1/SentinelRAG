from __future__ import annotations

from app.core.exceptions import IngestionError
from app.services.ingestion.parsers.base import BaseDocumentParser, ParsedDocument


class TextDocumentParser(BaseDocumentParser):
    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        if not content or not content.strip():
            raise IngestionError(f"File {filename} is empty.")

        text = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = content.decode(encoding).strip()
                break
            except UnicodeDecodeError:
                continue

        if not text:
            raise IngestionError(f"Unable to decode text file {filename} with supported encodings.")

        return ParsedDocument(
            text=text,
            page_count=1,
            sections=["main"],
            metadata={"filename": filename, "format": "txt"},
        )
