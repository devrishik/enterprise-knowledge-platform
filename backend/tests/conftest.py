"""Pytest configuration.

Test environment is pinned BEFORE any app import: hashing embeddings, lexical
reranker, extractive LLM, array vector backend, and a dedicated test database
(ekip_test).  Integration tests require a reachable PostgreSQL and are skipped
with an explicit message when it is not available.

Set EKIP_TEST_DATABASE_URL to override the test database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

TEST_DB_URL = os.environ.get(
    "EKIP_TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/ekip_test"
)

os.environ["EKIP_ENVIRONMENT"] = "test"
os.environ["EKIP_VECTOR_BACKEND"] = os.environ.get("EKIP_TEST_VECTOR_BACKEND", "array")
os.environ["EKIP_EMBEDDING_PROVIDER"] = "hashing"
os.environ["EKIP_EMBEDDING_DIM"] = "64"
os.environ["EKIP_RERANKER_PROVIDER"] = "lexical"
os.environ["EKIP_LLM_PROVIDER"] = "extractive"
os.environ["EKIP_DATABASE_URL"] = TEST_DB_URL
os.environ["EKIP_LOG_LEVEL"] = "ERROR"
os.environ["EKIP_SEED_DEMO_DATA"] = "true"
os.environ["EKIP_MAX_QUERY_REWRITES"] = "1"

import pytest  # noqa: E402


def _database_reachable() -> bool:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


DB_AVAILABLE = _database_reachable()
requires_db = pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL test database not reachable")


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def alembic_config():
    from alembic.config import Config

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    return cfg


@pytest.fixture(scope="session")
def database(alembic_config):
    """Ensure the schema exists (migrate from scratch if needed)."""
    from alembic import command

    command.upgrade(alembic_config, "head")
    yield


def _truncate_all() -> None:
    from sqlalchemy import text

    from app.db.session import get_engine

    tables = [
        "messages",
        "conversations",
        "query_traces",
        "ingestion_jobs",
        "document_chunks",
        "documents",
        "evaluation_runs",
        "evaluation_questions",
        "users",
        "tenants",
    ]
    with get_engine().begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {', '.join(tables)} CASCADE"))


@pytest.fixture()
def db_session(database):
    """Fresh database session with a clean schema per test."""
    from app.db.session import get_session_factory
    from app.services.seed import seed_demo_data, seed_evaluation_questions

    _truncate_all()
    factory = get_session_factory()
    session = factory()
    seed_demo_data(session)
    seed_evaluation_questions(session)
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def acme_user(db_session):
    from sqlalchemy import select

    from app.models import User

    user = db_session.scalar(select(User).where(User.email == "alice@acme.test"))
    assert user is not None
    return user


@pytest.fixture()
def globex_user(db_session):
    from sqlalchemy import select

    from app.models import User

    user = db_session.scalar(select(User).where(User.email == "gina@globex.test"))
    assert user is not None
    return user


@pytest.fixture()
def client(db_session):
    """FastAPI test client (runs startup: seeding, then serves requests)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def ingested_corpus(db_session, acme_user):
    """The full sample corpus ingested for the acme tenant."""
    from pathlib import Path

    from app.ingestion.pipeline import run_ingestion

    directory = BACKEND.parent / "data" / "sample_documents"
    results = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        result = run_ingestion(
            db_session, filename=path.name, payload=path.read_bytes(),
            tenant_id=acme_user.tenant_id, user_id=acme_user.id,
        )
        results.append(result)
    db_session.commit()
    return results
