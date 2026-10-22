"""PDF parser preserving page numbers via pypdf."""

from __future__ import annotations

import io

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.parsers.base import ParseError


class PdfParser:
    def parse(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ParseError("pypdf is not installed") from exc

        try:
            reader = PdfReader(io.BytesIO(payload))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ParseError("PDF is password-protected") from exc

            blocks: list[ParsedBlock] = []
            pages_with_text = 0
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                text = text.strip()
                if not text:
                    continue
                pages_with_text += 1
                # Split page text into paragraphs; keep the page number on every
                # block so citations can point at pages.
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [text]
                for para in paragraphs:
                    section = self._heading_like(para)
                    blocks.append(
                        ParsedBlock(text=para, kind="heading" if section else "paragraph",
                                    page=page_number, section=section)
                    )
            if not blocks or pages_with_text == 0:
                raise ParseError("PDF contains no extractable text (possibly a scanned image)")
            metadata = {"page_count": len(reader.pages), "pages_with_text": pages_with_text}
            title = (reader.metadata and reader.metadata.title) or filename
            return ParsedDocument(
                filename=filename, doc_type="pdf", title=str(title)[:500],
                blocks=blocks, metadata=metadata,
            )
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Failed to parse PDF: {exc}") from exc

    @staticmethod
    def _heading_like(paragraph: str) -> str | None:
        """Best-effort heading detection for PDF lines (no structural markup)."""
        first_line = paragraph.strip().splitlines()[0] if paragraph.strip() else ""
        cleaned = first_line.strip()
        if 3 <= len(cleaned) <= 80:
            letters = sum(c.isalpha() for c in cleaned)
            upper_ratio = sum(c.isupper() for c in cleaned) / letters if letters else 0
            words = cleaned.split()
            if upper_ratio > 0.7 and 1 <= len(words) <= 12 and not cleaned.endswith((".", ";", ",")):
                return cleaned
        return None
