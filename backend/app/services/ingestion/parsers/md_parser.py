from __future__ import annotations

import re
from app.core.exceptions import IngestionError
from app.services.ingestion.parsers.base import BaseDocumentParser, ParsedDocument


class MarkdownDocumentParser(BaseDocumentParser):
    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        if not content or not content.strip():
            raise IngestionError(f"File {filename} is empty.")

        text = ""
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                decoded = content.decode(encoding).strip()
                cleaned = "".join(ch for ch in decoded if ch.isprintable() or ch in "\n\r\t").strip()
                if cleaned:
                    text = cleaned
                    break
            except UnicodeDecodeError:
                continue

        if not text:
            raise IngestionError(f"Unable to decode markdown file {filename}.")

        # Extract markdown headers as sections
        headers = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
        sections = headers if headers else ["main"]

        return ParsedDocument(
            text=text,
            page_count=1,
            sections=sections,
            metadata={"filename": filename, "format": "markdown", "header_count": len(headers)},
        )
