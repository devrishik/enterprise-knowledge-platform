"""Ingestion pipeline: parsing -> structure-aware chunking -> embeddings -> storage.

Each supported file type has a dedicated parser producing a uniform stream of
``ParsedBlock`` objects (text + optional page/section metadata), so chunking,
embedding and indexing are format-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedBlock:
    """A structural unit of a document (heading, paragraph, list item, ...)."""

    text: str
    kind: str = "paragraph"  # heading | paragraph | list_item | table_row
    page: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    doc_type: str
    title: str = ""
    blocks: list[ParsedBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)
