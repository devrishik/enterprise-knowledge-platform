"""Single source of truth for the embedding column type.

The production backend stores embeddings in a native ``pgvector`` column and
performs cosine ANN search in PostgreSQL.  A compatibility backend stores the
same vectors as ``double precision[]`` and ranks in Python; it exists so the
full pipeline can run and be tested on PostgreSQL installations where the
pgvector extension is unavailable.  The ORM model and the Alembic migration
both call :func:`embedding_sql_type`, so the schema never diverges from the
runtime configuration.
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION
from sqlalchemy.types import TypeEngine

from app.core.config import get_settings


def vector_backend() -> str:
    return get_settings().vector_backend


def embedding_sql_type() -> TypeEngine:
    """Return the SQL type for the ``document_chunks.embedding`` column."""
    settings = get_settings()
    if settings.vector_backend == "pgvector":
        from pgvector.sqlalchemy import Vector

        return Vector(settings.embedding_dim)
    # Compatibility backend: plain float array, cosine similarity computed in
    # Python (see app/retrieval/dense.py).
    return ARRAY(DOUBLE_PRECISION)
