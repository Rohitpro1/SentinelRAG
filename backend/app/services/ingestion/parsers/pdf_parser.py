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

        try:
            import pypdf  # type: ignore[import-not-found]
            reader = pypdf.PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise IngestionError(f"File {filename} is an encrypted PDF.")

            for page in reader.pages:
                txt = page.extract_text() or ""
                # Clean printable text only
                cleaned = "".join(ch for ch in txt if ch.isprintable() or ch in "\n\r\t").strip()
                if cleaned:
                    pages_text.append(cleaned)
        except ImportError:
            # Fallback for plain ASCII text streams when pypdf is not available
            text_blocks = re.findall(b"BT(.*?)ET", content, re.DOTALL)
            extracted = []
            for block in text_blocks:
                strings = re.findall(b"\\((.*?)\\)", block)
                if strings:
                    raw_bytes = b" ".join(strings)
                    try:
                        decoded = raw_bytes.decode("utf-8", errors="ignore")
                        cleaned = "".join(ch for ch in decoded if ch.isprintable() or ch in "\n\r\t").strip()
                        if cleaned:
                            extracted.append(cleaned)
                    except Exception:
                        continue
            if extracted:
                pages_text.append("\n".join(extracted))
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(f"Corrupted PDF file {filename}: {exc}") from exc

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise IngestionError(f"PDF file {filename} contains no extractable text.")

        return ParsedDocument(
            text=full_text,
            page_count=max(1, len(pages_text)),
            sections=["main"],
            metadata={"filename": filename, "format": "pdf", "page_count": len(pages_text)},
        )
