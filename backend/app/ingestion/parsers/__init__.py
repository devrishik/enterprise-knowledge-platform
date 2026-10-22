"""Format-specific parsers. Each returns a ParsedDocument with structural blocks."""

from __future__ import annotations

from app.ingestion.parsers.base import ParseError, detect_doc_type, safe_filename
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser

PARSERS: dict[str, type] = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "txt": TxtParser,
    "md": MarkdownParser,
    "html": HtmlParser,
}


def get_parser(doc_type: str):
    try:
        return PARSERS[doc_type]()
    except KeyError:
        raise ParseError(f"No parser available for document type '{doc_type}'") from None


__all__ = [
    "PARSERS",
    "get_parser",
    "ParseError",
    "detect_doc_type",
    "safe_filename",
    "PdfParser",
    "DocxParser",
    "TxtParser",
    "MarkdownParser",
    "HtmlParser",
]
