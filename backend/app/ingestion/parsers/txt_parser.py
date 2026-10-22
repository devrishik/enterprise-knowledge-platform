"""Plain-text parser: split on blank lines into paragraphs."""

from __future__ import annotations

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.parsers.base import ParseError


class TxtParser:
    def parse(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = payload.decode("utf-8", errors="replace")
            except Exception as exc:  # pragma: no cover
                raise ParseError(f"Failed to decode text file: {exc}") from exc
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        blocks = [ParsedBlock(text=p, kind="paragraph") for p in paragraphs]
        if not blocks:
            raise ParseError("Text file is empty")
        first_line = paragraphs[0].splitlines()[0]
        title = first_line[:80] if len(first_line) <= 80 else filename
        return ParsedDocument(
            filename=filename, doc_type="txt", title=title, blocks=blocks,
            metadata={"paragraph_count": len(blocks)},
        )
