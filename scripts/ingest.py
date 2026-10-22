"""Ingest documents into a tenant's knowledge base via the real pipeline.

Usage (from backend/):
    python ../scripts/ingest.py --tenant acme [--dir ../data/sample_documents]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.ingestion.pipeline import run_ingestion  # noqa: E402
from app.models import Tenant, User  # noqa: E402
from sqlalchemy import select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents for a tenant")
    parser.add_argument("--tenant", default="acme", help="tenant slug (default: acme)")
    parser.add_argument("--dir", default=str(ROOT / "data" / "sample_documents"))
    parser.add_argument("--user", default=None, help="user email to attribute uploads to")
    args = parser.parse_args()

    settings = get_settings()
    print(f"database : {settings.database_url.split('@')[-1]}")
    print(f"backend  : vector={settings.vector_backend} embed={settings.embedding_provider}")

    factory = get_session_factory()
    with factory() as session:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == args.tenant))
        if tenant is None:
            print(f"tenant '{args.tenant}' not found — start the API once to seed demo tenants")
            return 2
        user = None
        if args.user:
            user = session.scalar(select(User).where(User.email == args.user))
        else:
            user = session.scalar(select(User).where(User.tenant_id == tenant.id).limit(1))

        directory = Path(args.dir)
        files = sorted(p for p in directory.iterdir() if p.is_file())
        ok, failed = 0, 0
        for path in files:
            payload = path.read_bytes()
            result = run_ingestion(
                session,
                filename=path.name,
                payload=payload,
                tenant_id=tenant.id,
                user_id=user.id if user else None,
            )
            session.commit()
            status = "OK " if result.status == "completed" else "FAIL"
            print(
                f"{status} {path.name:<28} chunks={result.chunk_count:<3} "
                f"{result.duration_ms:>8.1f} ms {result.error or ''}"
            )
            ok += result.status == "completed"
            failed += result.status != "completed"
        print(f"\n{ok} ingested, {failed} failed")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
