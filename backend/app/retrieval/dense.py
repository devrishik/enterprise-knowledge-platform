"""Dense vector retrieval.

Production mode uses pgvector's cosine distance operator (``<=>``) with an
HNSW index.  Compatibility mode (``EKIP_VECTOR_BACKEND=array``) fetches the
tenant's chunk vectors and computes exact cosine similarity in Python — same
interface, no pgvector extension required.  See docs/design_decisions.md.
"""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings import get_embedding_provider
from app.models import Document, DocumentChunk
from app.retrieval import ScoredChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # vectors are L2-normalised at embed time


class DenseRetriever:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self.log = get_logger("retrieval.dense")

    def search(self, query: str, tenant_id, top_k: int | None = None) -> tuple[list[ScoredChunk], float]:
        """Return (chunks, latency_ms) for the tenant's own documents only."""
        top_k = top_k or self.settings.dense_top_k
        started = time.perf_counter()
        provider = get_embedding_provider()
        query_vector = provider.embed_query(query)

        if self.settings.vector_backend == "pgvector":
            chunks = self._pgvector_rows(query_vector, tenant_id, top_k)
        else:
            chunks = self._array_rows(query_vector, tenant_id, top_k)

        latency = round((time.perf_counter() - started) * 1000, 2)
        return chunks, latency

    def _pgvector_rows(self, query_vector: list[float], tenant_id, top_k: int) -> list[ScoredChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        rows = self.session.execute(
            select(DocumentChunk, Document.filename, distance.label("dist"))
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.tenant_id == tenant_id, DocumentChunk.embedding.is_not(None))
            .order_by(distance, DocumentChunk.id)  # stable tie-break
            .limit(top_k)
        ).all()
        return [
            ScoredChunk(
                chunk_id=str(row[0].id),
                document_id=str(row[0].document_id),
                document_filename=row[1],
                text=row[0].text,
                score=1.0 - float(row[2]),
                source="dense",
                page_number=row[0].page_number,
                section=row[0].section,
                metadata=row[0].metadata_json or {},
            )
            for row in rows
        ]

    def _array_rows(self, query_vector: list[float], tenant_id, top_k: int) -> list[ScoredChunk]:
        rows = self.session.execute(
            select(DocumentChunk, Document.filename)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.tenant_id == tenant_id, DocumentChunk.embedding.is_not(None))
            .order_by(DocumentChunk.id)  # deterministic base order for score ties
        ).all()
        scored: list[tuple[float, DocumentChunk, str]] = []
        for chunk, filename in rows:
            vec = chunk.embedding
            if not vec:
                continue
            scored.append((cosine_similarity(query_vector, list(vec)), chunk, filename))
        # sort by score desc with a stable tie-break on chunk id
        scored.sort(key=lambda item: (-item[0], str(item[1].id)))
        return [
            ScoredChunk(
                chunk_id=str(chunk.id),
                document_id=str(chunk.document_id),
                document_filename=filename,
                text=chunk.text,
                score=float(score),
                source="dense",
                page_number=chunk.page_number,
                section=chunk.section,
                metadata=chunk.metadata_json or {},
            )
            for score, chunk, filename in scored[:top_k]
        ]
