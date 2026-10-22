"""Latency benchmark for the retrieval strategies over the evaluation questions.

Usage (from backend/):
    python ../scripts/benchmark.py --tenant acme [--repeat 3]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import get_session_factory  # noqa: E402
from app.evaluation.runner import session_questions  # noqa: E402
from app.models import Tenant  # noqa: E402
from app.retrieval.engine import HybridRetrievalEngine  # noqa: E402
from sqlalchemy import select  # noqa: E402

STRATEGIES = ["dense", "lexical", "hybrid", "hybrid_rerank"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies")
    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    factory = get_session_factory()
    with factory() as session:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == args.tenant))
        if tenant is None:
            print(f"tenant '{args.tenant}' not found")
            return 2
        questions = [q.question for q in session_questions(session) if q.expected_answerable]
        engine = HybridRetrievalEngine(session)

        print(f"{'strategy':<14} {'median_total_ms':>15} {'p95_ms':>9}")
        for strategy in STRATEGIES:
            totals: list[float] = []
            for _ in range(args.repeat):
                for question in questions:
                    started = time.perf_counter()
                    engine.retrieve(question, tenant.id, strategy=strategy)
                    totals.append((time.perf_counter() - started) * 1000)
            totals.sort()
            median = statistics.median(totals)
            p95 = totals[int(len(totals) * 0.95) - 1] if totals else 0.0
            print(f"{strategy:<14} {median:>15.1f} {p95:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
