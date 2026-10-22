#!/bin/sh
# Backend container entrypoint: wait for PostgreSQL, migrate, serve.
set -e

echo "waiting for database..."
python - <<'EOF'
import os, sys, time
import psycopg

url = os.environ["EKIP_DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
for attempt in range(60):
    try:
        with psycopg.connect(url, connect_timeout=3):
            print("database is up")
            sys.exit(0)
    except Exception as exc:
        print(f"  not ready ({exc.__class__.__name__}), retrying...")
        time.sleep(2)
print("database never became reachable", file=sys.stderr)
sys.exit(1)
EOF

echo "applying migrations..."
alembic upgrade head

echo "starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
