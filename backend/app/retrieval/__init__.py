"""Retrieval layer: dense (pgvector), lexical (PostgreSQL FTS), RRF fusion,
cross-encoder reranking, and the hybrid engine that orchestrates them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetrievalStrategy = Literal["dense", "lexical", "hybrid", "hybrid_rerank"]


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    document_filename: str
    text: str
    score: float
    source: str  # dense | lexical | fused | reranked
    page_number: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_filename": self.document_filename,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
        }


@dataclass
class RetrievalResult:
    strategy: str
    query: str
    dense: list[ScoredChunk] = field(default_factory=list)
    lexical: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    reranked: list[ScoredChunk] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

    def final_chunks(self) -> list[ScoredChunk]:
        """The selected context chunks for the given strategy."""
        if self.strategy == "dense":
            return self.dense
        if self.strategy == "lexical":
            return self.lexical
        if self.strategy == "hybrid":
            return self.fused
        return self.reranked
