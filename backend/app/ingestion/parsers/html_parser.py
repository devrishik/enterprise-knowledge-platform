"""HTML parser preserving headings, paragraphs, lists and tables.

BeautifulSoup with Python's built-in parser (no external HTML engine needed);
script/style content is discarded and all markup is stripped — only text nodes
are kept, so no untrusted HTML ever reaches the UI or the LLM.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.ingestion import ParsedBlock, ParsedDocument
from app.ingestion.parsers.base import ParseError

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HtmlParser:
    def parse(self, filename: str, payload: bytes) -> ParsedDocument:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
        if not text.strip():
            raise ParseError("HTML file is empty")

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = (soup.title.string.strip() if soup.title and soup.title.string else "") or filename

        blocks: list[ParsedBlock] = []
        section: str | None = None

        for element in soup.body.descendants if soup.body else soup.descendants:
            name = getattr(element, "name", None)
            if not name:
                continue
            if name in _HEADING_TAGS:
                heading_text = element.get_text(" ", strip=True)
                if heading_text:
                    blocks.append(ParsedBlock(text=heading_text, kind="heading", section=heading_text))
                    section = heading_text
            elif name == "p":
                p_text = element.get_text(" ", strip=True)
                if p_text:
                    blocks.append(ParsedBlock(text=p_text, kind="paragraph", section=section))
            elif name == "li":
                li_text = element.get_text(" ", strip=True)
                if li_text:
                    blocks.append(ParsedBlock(text=li_text, kind="list_item", section=section))
            elif name == "tr":
                cells = [
                    c.get_text(" ", strip=True)
                    for c in element.find_all(["td", "th"])
                    if c.get_text(" ", strip=True)
                ]
                if cells:
                    blocks.append(
                        ParsedBlock(text=" | ".join(cells), kind="table_row", section=section or "Table")
                    )

        if not blocks:
            raise ParseError("HTML contains no extractable text")
        return ParsedDocument(
            filename=filename,
            doc_type="html",
            title=title[:500],
            blocks=blocks,
            metadata={"block_count": len(blocks)},
        )
