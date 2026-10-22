"""FastAPI dependencies: DB session + lightweight API-key authentication."""

from __future__ import annotations

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AuthError
from app.db.session import get_db
from app.models import User


def get_current_user(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: Session = Depends(get_db),
) -> User:
    if not x_api_key:
        raise AuthError("Missing X-API-Key header. Use a demo key from GET /api/auth/demo-users.")
    user = session.scalar(select(User).where(User.api_key == x_api_key))
    if user is None:
        raise AuthError("Invalid API key.")
    return user
