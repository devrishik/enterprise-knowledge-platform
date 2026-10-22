"""Health endpoint: reports database connectivity and provider configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import schemas
from app.core.config import get_settings
from app.db.session import get_db
from app.embeddings import get_embedding_provider
from app.llm import get_llm_provider
from app.retrieval.rerank import get_reranker

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=schemas.HealthOut)
def health(session: Session = Depends(get_db)) -> schemas.HealthOut:
    settings = get_settings()

    database = "up"
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        database = f"down: {type(exc).__name__}"

    llm = get_llm_provider()
    try:
        llm.ensure_configured()
        llm_configured = True
        llm_message = f"provider '{llm.name}' ready"
    except Exception as exc:
        llm_configured = False
        llm_message = str(exc)

    try:
        embedder = get_embedding_provider()
        embedding_message = f"{embedder.name} (dim={embedder.dimension})"
    except Exception as exc:
        embedding_message = f"unavailable: {exc}"

    reranker = get_reranker()
    reranker_name = getattr(reranker, "name", "unknown")

    return schemas.HealthOut(
        status="ok" if database == "up" else "degraded",
        database=database,
        vector_backend=settings.vector_backend,
        llm_provider=llm.name,
        llm_configured=llm_configured,
        embedding_provider=embedding_message,
        reranker_provider=reranker_name,
        details={"llm_message": llm_message},
    )
