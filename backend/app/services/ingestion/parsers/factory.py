from __future__ import annotations

import os
from app.core.exceptions import IngestionError
from app.services.ingestion.parsers.base import BaseDocumentParser
from app.services.ingestion.parsers.docx_parser import DOCXDocumentParser
from app.services.ingestion.parsers.md_parser import MarkdownDocumentParser
from app.services.ingestion.parsers.pdf_parser import PDFDocumentParser
from app.services.ingestion.parsers.txt_parser import TextDocumentParser


class DocumentParserFactory:
    _parsers: dict[str, BaseDocumentParser] = {
        ".pdf": PDFDocumentParser(),
        ".docx": DOCXDocumentParser(),
        ".txt": TextDocumentParser(),
        ".md": MarkdownDocumentParser(),
        ".markdown": MarkdownDocumentParser(),
    }

    @classmethod
    def get_parser(cls, filename: str) -> BaseDocumentParser:
        ext = os.path.splitext(filename)[1].lower()
        if not ext or ext not in cls._parsers:
            raise IngestionError(
                f"Unsupported file format '{ext}'. Supported formats: PDF (.pdf), DOCX (.docx), TXT (.txt), Markdown (.md)."
            )
        return cls._parsers[ext]
