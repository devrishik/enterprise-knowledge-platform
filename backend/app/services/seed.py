"""Startup seeding: demo tenants/users and the evaluation question set."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import EvaluationQuestion, Tenant, User

DEMO_TENANTS = [
    {
        "name": "Acme Corp",
        "slug": "acme",
        "users": [
            {"email": "alice@acme.test", "display_name": "Alice (Acme)", "api_key": "demo-acme-alice-key"},
            {"email": "bob@acme.test", "display_name": "Bob (Acme)", "api_key": "demo-acme-bob-key"},
        ],
    },
    {
        "name": "Globex Corp",
        "slug": "globex",
        "users": [
            {"email": "gina@globex.test", "display_name": "Gina (Globex)", "api_key": "demo-globex-gina-key"},
        ],
    },
]


def seed_demo_data(session: Session) -> None:
    log = get_logger("seed")
    settings = get_settings()
    if not settings.seed_demo_data:
        return

    for tenant_def in DEMO_TENANTS:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == tenant_def["slug"]))
        if tenant is None:
            tenant = Tenant(name=tenant_def["name"], slug=tenant_def["slug"])
            session.add(tenant)
            session.flush()
            log.info("seeded_tenant", extra={"context": {"slug": tenant.slug}})
        for user_def in tenant_def["users"]:
            if session.scalar(select(User).where(User.email == user_def["email"])) is None:
                session.add(
                    User(
                        tenant_id=tenant.id,
                        email=user_def["email"],
                        display_name=user_def["display_name"],
                        api_key=user_def["api_key"],
                        is_demo=True,
                    )
                )
    session.flush()


def seed_evaluation_questions(session: Session) -> int:
    """Load the evaluation dataset (data/evaluation/questions.json) once."""
    path = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "questions.json"
    if not path.exists():
        return 0
    existing = session.scalar(select(EvaluationQuestion).limit(1))
    if existing is not None:
        return 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for item in payload:
        session.add(
            EvaluationQuestion(
                question=item["question"],
                question_type=item["question_type"],
                expected_doc_filenames=item.get("expected_doc_filenames", []),
                expected_answer_keywords=item.get("expected_answer_keywords", []),
                expected_answerable=item.get("expected_answerable", True),
                notes=item.get("notes"),
            )
        )
        count += 1
    session.flush()
    return count
