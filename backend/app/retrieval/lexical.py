"""Lexical (BM25-style) retrieval using PostgreSQL full-text search.

Query terms are OR-expanded (each content word stems to a lexeme joined with
``|``) and ranked with ``ts_rank_cd`` cover-density, normalised by document
length (flag 32).  Chunks matching more query terms rank higher — behaviour
similar in spirit to BM25's term-frequency saturation, minus IDF weighting
(which PostgreSQL core does not expose; that would need ParadeDB/pg_search or
an external engine — see docs/design_decisions.md).

Why OR-expansion rather than ``websearch_to_tsquery``'s strict AND: real user
questions routinely contain one word no chunk uses verbatim ("How many days
are employees entitled to..."), and AND semantics then return nothing at all.
OR-expansion keeps recall sane; precision is restored by RRF fusion and the
cross-encoder reranker.  Quoted-phrase queries are not specially handled — a
documented limitation for this corpus scale.
"""

from __future__ import annotations

import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval import ScoredChunk

_SQL = text(
    """
    WITH terms AS (
        SELECT DISTINCT lex FROM (
            SELECT unnest(ts_lexize('english_stem', word)) AS lex
            FROM regexp_split_to_table(
                     lower(regexp_replace(:query, '[^a-zA-Z0-9\\s]+', ' ', 'g')), '\\s+'
                 ) AS word
            WHERE length(word) >= 2
        ) s
        WHERE lex IS NOT NULL
    ),
    q AS (
        SELECT to_tsquery('english', string_agg(lex, ' | ')) AS query FROM terms
    )
    SELECT c.id::text          AS chunk_id,
           c.document_id::text AS document_id,
           d.filename          AS document_filename,
           c.text,
           c.page_number,
           c.section,
           ts_rank_cd(c.tsv, q.query, 32) AS score
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN q
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND q.query IS NOT NULL
      AND c.tsv @@ q.query
    ORDER BY score DESC, c.id
    LIMIT :top_k
    """
)


class LexicalRetriever:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self.log = get_logger("retrieval.lexical")

    def search(self, query: str, tenant_id, top_k: int | None = None) -> tuple[list[ScoredChunk], float]:
        """Return (chunks, latency_ms), tenant-scoped via the WHERE clause."""
        top_k = top_k or self.settings.lexical_top_k
        started = time.perf_counter()
        rows = self.session.execute(_SQL, {"query": query, "tenant_id": str(tenant_id), "top_k": top_k}).mappings().all()
        latency = round((time.perf_counter() - started) * 1000, 2)

        max_score = max((r["score"] for r in rows), default=0.0)
        chunks = [
            ScoredChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_filename=r["document_filename"],
                text=r["text"],
                # normalise per-query to [0,1] for the heuristic relevance gate
                score=float(r["score"]) / max_score if max_score > 0 else 0.0,
                source="lexical",
                page_number=r["page_number"],
                section=r["section"],
            )
            for r in rows
        ]
        return chunks, latency
