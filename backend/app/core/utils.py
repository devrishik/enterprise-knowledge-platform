"""Small shared utilities: safe UUID parsing for path parameters."""

from __future__ import annotations

from uuid import UUID

from app.core.errors import NotFoundError


def parse_uuid(value: str, what: str = "resource") -> UUID:
    """Parse a path parameter as UUID; malformed ids become 404s, never 500s."""
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise NotFoundError(f"Invalid {what} id") from None
