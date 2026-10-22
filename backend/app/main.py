"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.errors import install_error_handlers
from app.core.logging import configure_logging, get_logger
from app.db.session import get_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("startup")
    log.info(
        "startup",
        extra={
            "context": {
                "environment": settings.environment,
                "vector_backend": settings.vector_backend,
                "llm_provider": settings.llm_provider,
                "embedding_provider": settings.embedding_provider,
            }
        },
    )
    # Seed demo tenants/users + evaluation questions (idempotent). The schema
    # itself is created by Alembic (`alembic upgrade head` runs at container
    # start); seeding tolerates a not-yet-migrated database.
    try:
        factory = get_session_factory()
        with factory() as session:
            from app.services.seed import seed_demo_data, seed_evaluation_questions

            seed_demo_data(session)
            seeded = seed_evaluation_questions(session)
            session.commit()
            if seeded:
                log.info("seeded_evaluation_questions", extra={"context": {"count": seeded}})
    except Exception as exc:
        log.warning(
            "seed_skipped",
            extra={"context": {"error": f"{type(exc).__name__}: {exc}"}},
        )
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Agentic RAG platform: hybrid retrieval (pgvector + PostgreSQL FTS), "
            "RRF fusion, cross-encoder reranking, LangGraph workflow, citation-grounded "
            "answers, evaluation and observability."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    from app.api.routes import router as api_router
    from app.api.routes.health import router as health_router

    app.include_router(api_router)
    app.include_router(health_router)
    return app


app = create_app()
