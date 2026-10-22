"""Structure-aware chunking.

Strategy (``structure`` — default):

1. Blocks are grouped under their current heading section; the heading text is
   prepended to the section's chunks so every chunk carries its context.
2. Consecutive blocks of the same section (and, for PDFs, the same page) are
   merged while the result stays under ``chunk_size`` characters — chunks follow
   the document's semantic structure instead of cutting mid-thought.
3. Oversized blocks are split at sentence boundaries with ``chunk_overlap``
   characters of overlap; page/section metadata is preserved on every piece.
4. PDF page boundaries are never merged across pages, so citations can point at
   an exact page.

A ``fixed`` strategy (naive fixed-size character split with overlap) is kept
selectable for A/B comparison and as a fallback for pathological inputs.

Defaults (1200 chars / 200 overlap ≈ 300/50 tokens of English prose) were
chosen to fit the all-MiniLM-L6-v2 context window (256 word pieces) with head
room for the heading prefix, while keeping enough context per chunk for the
reranker and the LLM. Both values are configuration, not code constants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.ingestion import ParsedBlock, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)


def _token_count(text: str) -> int:
    return len(text.split())


def _split_oversized(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a too-long block at sentence boundaries with overlap."""
    if len(text) <= chunk_size:
        return [text]
    sentences = _SENTENCE_SPLIT.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            # Pathologically long sentence: hard character split.
            if current:
                pieces.append(current)
                current = ""
            step = max(chunk_size - overlap, 1)
            for start in range(0, len(sentence), step):
                piece = sentence[start : start + chunk_size]
                if piece.strip():
                    pieces.append(piece)
            continue
        if current and len(current) + 1 + len(sentence) > chunk_size:
            pieces.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + sentence).strip() if tail else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


class StructureAwareChunker:
    name = "structure"

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        settings = get_settings()
        chunks: list[Chunk] = []

        # group blocks by (page, section) boundaries; PDF pages never merge
        groups: list[dict] = []
        for block in parsed.blocks:
            boundary = self._boundary(block, parsed.doc_type)
            if groups and groups[-1]["boundary"] == boundary and block.kind != "heading":
                groups[-1]["blocks"].append(block)
            else:
                groups.append({"boundary": boundary, "blocks": [block]})

        for group in groups:
            blocks = group["blocks"]
            section = next((b.section for b in blocks if b.section), None)
            heading_prefix = section if section else ""
            page = next((b.page for b in blocks if b.page is not None), None)

            merged = ""
            for block in blocks:
                block_text = block.text
                if not merged:
                    merged = block_text
                    continue
                candidate = f"{merged}\n{block_text}"
                prefix_len = len(heading_prefix) + 1 if heading_prefix else 0
                if len(candidate) + prefix_len <= self.chunk_size:
                    merged = candidate
                else:
                    self._emit(chunks, merged, heading_prefix, section, page)
                    tail = merged[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                    merged = (tail + "\n" + block_text).strip() if tail else block_text
            if merged.strip():
                self._emit(chunks, merged, heading_prefix, section, page)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        return chunks

    def _boundary(self, block: ParsedBlock, doc_type: str) -> tuple:
        if doc_type == "pdf":
            return ("page", block.page, block.section)
        return ("section", block.section)

    def _emit(
        self,
        chunks: list[Chunk],
        text: str,
        heading_prefix: str,
        section: str | None,
        page: int | None,
    ) -> None:
        full = f"{heading_prefix}\n{text}" if heading_prefix else text
        for piece in _split_oversized(full, self.chunk_size, self.chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_index=-1,  # assigned after all chunks exist
                    text=piece,
                    token_count=_token_count(piece),
                    page_number=page,
                    section=section,
                    metadata={"strategy": self.name, **({"heading_prefix": heading_prefix} if heading_prefix else {})},
                )
            )


class FixedSizeChunker(StructureAwareChunker):
    """Naive fixed-size character split; selectable for A/B comparison."""

    name = "fixed"

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        text = parsed.full_text
        pieces = _split_oversized(text, self.chunk_size, self.chunk_overlap)
        return [
            Chunk(
                chunk_index=i,
                text=piece,
                token_count=_token_count(piece),
                page_number=None,
                section=None,
                metadata={"strategy": self.name},
            )
            for i, piece in enumerate(pieces)
        ]


def get_chunker(strategy: str | None = None) -> StructureAwareChunker:
    settings = get_settings()
    chosen = strategy or settings.chunking_strategy
    if chosen == "fixed":
        return FixedSizeChunker()
    return StructureAwareChunker()
