"""Ingestion pipeline: validate -> parse -> chunk -> embed -> store -> index.

File-type validation happens before any database write, so rejected uploads
never leave partial rows.  Parse-level failures (corrupt file, empty document)
are recorded as ``failed`` documents/jobs and returned as a result — not raised
— so the failure record itself is committed and inspectable from the UI.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.embeddings import get_embedding_provider
from app.ingestion.chunking import Chunk, get_chunker
from app.ingestion.parsers import detect_doc_type, get_parser
from app.models import Document, DocumentChunk, IngestionJob


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    doc_type: str
    status: str
    chunk_count: int
    duration_ms: float
    timing_ms: dict = field(default_factory=dict)
    error: str | None = None


class IngestionPipeline:
    def __init__(self, session: Session):
        self.session = session
        self.log = get_logger("ingestion")

    def ingest(
        self,
        *,
        filename: str,
        payload: bytes,
        tenant_id,
        user_id=None,
    ) -> IngestionResult:
        # Client errors (unsupported type) surface before any DB write.
        doc_type = detect_doc_type(filename, payload)

        started = time.perf_counter()
        timing: dict[str, float] = {}

        document = Document(
            tenant_id=tenant_id,
            uploader_user_id=user_id,
            filename=filename,
            doc_type=doc_type,
            status="processing",
            content_hash=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        self.session.add(document)
        self.session.flush()  # assign document.id before referencing it
        job = IngestionJob(document_id=document.id, tenant_id=tenant_id, status="running", stage="parsing")
        self.session.add(job)
        self.session.flush()

        def fail(message: str) -> IngestionResult:
            document.status = "failed"
            document.error_message = message
            job.status = "failed"
            job.error_message = message
            job.finished_at = datetime.now(timezone.utc)
            job.timing_ms = timing
            self.session.flush()
            self.log.warning("ingestion_failed", extra={"context": {"filename": filename, "error": message}})
            return IngestionResult(
                document_id=str(document.id),
                filename=filename,
                doc_type=document.doc_type,
                status="failed",
                chunk_count=0,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                timing_ms=timing,
                error=message,
            )

        try:
            # ---- parsing ---------------------------------------------------
            parser = get_parser(doc_type)
            t0 = time.perf_counter()
            parsed = parser.parse(filename, payload)
            timing["parse_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            document.title = parsed.title

            # ---- chunking --------------------------------------------------
            job.stage = "chunking"
            t0 = time.perf_counter()
            chunker = get_chunker()
            chunks: list[Chunk] = chunker.chunk(parsed)
            timing["chunking_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            if not chunks:
                return fail("Document produced no chunks")

            # ---- embedding (batched) ---------------------------------------
            job.stage = "embedding"
            t0 = time.perf_counter()
            provider = get_embedding_provider()
            vectors = provider.embed_texts([c.text for c in chunks])
            timing["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            # ---- storage ---------------------------------------------------
            job.stage = "indexing"
            t0 = time.perf_counter()
            for chunk, vector in zip(chunks, vectors):
                self.session.add(
                    DocumentChunk(
                        document_id=document.id,
                        tenant_id=tenant_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        embedding=vector,
                        metadata_json=chunk.metadata,
                    )
                )
            timing["indexing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            document.status = "completed"
            document.chunk_count = len(chunks)
            document.metadata_json = {
                **parsed.metadata,
                "parser": type(parser).__name__,
                "chunking_strategy": chunker.name,
                "embedding_model": provider.name,
            }
            job.status = "completed"
            job.stage = "done"
            job.finished_at = datetime.now(timezone.utc)
            job.timing_ms = timing
            self.session.flush()

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.log.info(
                "ingestion_completed",
                extra={"context": {"filename": filename, "chunks": len(chunks), "ms": duration_ms}},
            )
            return IngestionResult(
                document_id=str(document.id),
                filename=filename,
                doc_type=document.doc_type,
                status="completed",
                chunk_count=len(chunks),
                duration_ms=duration_ms,
                timing_ms=timing,
            )
        except Exception as exc:
            # Parse errors, empty documents, embedding failures... The failure
            # is recorded and committed (no raise) so users can inspect it.
            message = f"{type(exc).__name__}: {exc}"[:500]
            return fail(message)


def run_ingestion(
    session: Session, *, filename: str, payload: bytes, tenant_id, user_id=None
) -> IngestionResult:
    """Convenience wrapper used by API routes, scripts and tests."""
    return IngestionPipeline(session).ingest(
        filename=filename, payload=payload, tenant_id=tenant_id, user_id=user_id
    )
