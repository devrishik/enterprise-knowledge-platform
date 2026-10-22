"""Markdown parser preserving heading sections, lists, tables and paragraphs.

Uses markdown-it-py tokens (CommonMark + table plugin) rather than regex
splitting, so nested structures and code fences are handled correctly.
"""

from __future__ import annotations

from markdown_it import MarkdownIt

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.parsers.base import ParseError


class MarkdownParser:
    def parse(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
        if not text.strip():
            raise ParseError("Markdown file is empty")

        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(text)

        blocks: list[ParsedBlock] = []
        section: str | None = None
        title: str | None = None
        in_list = False
        i = 0

        while i < len(tokens):
            tok = tokens[i]
            ttype = tok.type

            if ttype == "heading_open":
                inline = tokens[i + 1] if i + 1 < len(tokens) else None
                heading_text = (inline.content if inline and inline.type == "inline" else "").strip()
                if heading_text:
                    blocks.append(ParsedBlock(text=heading_text, kind="heading", section=heading_text))
                    section = heading_text
                    if title is None:
                        title = heading_text
                i += 3  # heading_open, inline, heading_close
                continue

            if ttype == "paragraph_open":
                j = i + 1
                parts: list[str] = []
                while j < len(tokens) and tokens[j].type != "paragraph_close":
                    if tokens[j].type == "inline":
                        parts.append(tokens[j].content.strip())
                    j += 1
                paragraph_text = "\n".join(p for p in parts if p)
                if paragraph_text.strip():
                    kind = "list_item" if in_list else "paragraph"
                    blocks.append(ParsedBlock(text=paragraph_text, kind=kind, section=section))
                i = j + 1
                continue

            if ttype == "tr_open":
                j = i + 1
                cells: list[str] = []
                while j < len(tokens) and tokens[j].type != "tr_close":
                    if tokens[j].type == "inline" and tokens[j].content.strip():
                        cells.append(tokens[j].content.strip())
                    j += 1
                if cells:
                    blocks.append(
                        ParsedBlock(
                            text=" | ".join(cells), kind="table_row",
                            section=section or "Table",
                        )
                    )
                i = j + 1
                continue

            if ttype in ("bullet_list_open", "ordered_list_open"):
                in_list = True
            elif ttype in ("bullet_list_close", "ordered_list_close"):
                in_list = False
            elif ttype == "fence" and tok.content.strip():
                blocks.append(ParsedBlock(text=tok.content.strip(), kind="paragraph", section=section))

            i += 1

        if not blocks:
            raise ParseError("Markdown contains no extractable text")
        return ParsedDocument(
            filename=filename,
            doc_type="md",
            title=(title or filename)[:500],
            blocks=blocks,
            metadata={"block_count": len(blocks)},
        )
