from __future__ import annotations

import io
import re
from app.core.exceptions import IngestionError
from app.services.ingestion.parsers.base import BaseDocumentParser, ParsedDocument


class PDFDocumentParser(BaseDocumentParser):
    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        if not content or len(content) < 10:
            raise IngestionError(f"File {filename} is empty or corrupted.")

        if not content.startswith(b"%PDF"):
            raise IngestionError(f"File {filename} is not a valid PDF document.")

        if b"/Encrypt" in content:
            raise IngestionError(f"File {filename} is an encrypted PDF.")

        pages_text: list[str] = []

        # Attempt using pypdf if installed
        try:
            import pypdf  # type: ignore[import-not-found]
            reader = pypdf.PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise IngestionError(f"File {filename} is an encrypted PDF.")

            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(txt.strip())
        except ImportError:
            # Native fallback PDF text extraction if pypdf not present
            text_blocks = re.findall(b"BT(.*?)ET", content, re.DOTALL)
            extracted = []
            for block in text_blocks:
                strings = re.findall(b"\\((.*?)\\)", block)
                if strings:
                    extracted.append(b" ".join(strings).decode("latin-1", errors="ignore"))
            if extracted:
                pages_text.append("\n".join(extracted))
        except Exception as exc:
            raise IngestionError(f"Corrupted PDF file {filename}: {exc}") from exc

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            # Extract plain string fallback from PDF stream
            plain_matches = re.findall(r"\(([A-Za-z0-9\s.,!?-]{4,})\)", content.decode("latin-1", errors="ignore"))
            if plain_matches:
                full_text = " ".join(plain_matches).strip()

        if not full_text:
            raise IngestionError(f"PDF file {filename} contains no extractable text.")

        return ParsedDocument(
            text=full_text,
            page_count=max(1, len(pages_text)),
            sections=["main"],
            metadata={"filename": filename, "format": "pdf", "page_count": len(pages_text)},
        )
