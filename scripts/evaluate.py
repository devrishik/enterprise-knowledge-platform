"""Run the evaluation suite and print a report from the actual results.

Usage (from backend/):
    python ../scripts/evaluate.py --tenant acme [--generation] [--max 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import get_session_factory  # noqa: E402
from app.evaluation.runner import EvaluationRunner  # noqa: E402
from app.models import Tenant  # noqa: E402
from sqlalchemy import select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--strategies", default="dense,lexical,hybrid,hybrid_rerank")
    parser.add_argument("--generation", action="store_true", help="also evaluate generation")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    factory = get_session_factory()
    with factory() as session:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == args.tenant))
        if tenant is None:
            print(f"tenant '{args.tenant}' not found")
            return 2

        runner = EvaluationRunner(session)
        run_ids = runner.run(
            tenant_id=tenant.id,
            strategies=[s.strip() for s in args.strategies.split(",")],
            include_generation=args.generation,
            max_questions=args.max,
        )
        session.commit()

        from app.models import EvaluationRun

        for run_id in run_ids:
            run = session.get(EvaluationRun, run_id)
            print(f"\n=== {run.strategy.upper()} ({run.question_count} questions) ===")
            print(json.dumps(run.summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
