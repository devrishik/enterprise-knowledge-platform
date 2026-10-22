"""DOCX parser preserving heading structure via python-docx."""

from __future__ import annotations

import io

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.parsers.base import ParseError


class DocxParser:
    def parse(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            import docx  # python-docx

            document = docx.Document(io.BytesIO(payload))
        except Exception as exc:
            raise ParseError(f"Failed to parse DOCX: {exc}") from exc

        blocks: list[ParsedBlock] = []
        section: str | None = None
        title = filename
        for para in document.paragraphs:
            text = (para.text or "").strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading") or style == "title":
                section = text
                if not blocks:
                    title = text
                blocks.append(ParsedBlock(text=text, kind="heading", section=section))
            else:
                blocks.append(ParsedBlock(text=text, kind="paragraph", section=section))

        for table in document.tables:
            for row in table.rows:
                cells = [(c.text or "").strip() for c in row.cells]
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    blocks.append(
                        ParsedBlock(text=row_text, kind="table_row", section=section or "Table")
                    )

        if not blocks:
            raise ParseError("DOCX contains no extractable text")
        return ParsedDocument(
            filename=filename, doc_type="docx", title=title[:500], blocks=blocks,
            metadata={"paragraph_count": len(blocks)},
        )
