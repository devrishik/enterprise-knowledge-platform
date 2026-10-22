"""Run the generation evaluation with a real LLM (env-configured provider).

Usage (local, Ollama example):
  EKIP_LLM_PROVIDER=openai \
  EKIP_OPENAI_BASE_URL=http://127.0.0.1:11434/v1 \
  EKIP_OPENAI_API_KEY=ollama-local \
  EKIP_LLM_MODEL=qwen2.5:3b \
  EKIP_LLM_TIMEOUT_SECONDS=180 \
  python scripts/run_llm_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP_PATH = ROOT / "data" / "evaluation" / "last_generation_run.json"
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import get_session_factory  # noqa: E402
from app.evaluation.runner import EvaluationRunner  # noqa: E402
from app.models import EvaluationRun, Tenant  # noqa: E402
from sqlalchemy import select  # noqa: E402


def main() -> int:
    factory = get_session_factory()
    with factory() as session:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == "acme"))
        if tenant is None:
            print("tenant 'acme' not found", file=sys.stderr)
            return 2
        runner = EvaluationRunner(session)
        run_ids = runner.run(
            tenant_id=tenant.id,
            strategies=["hybrid_rerank"],
            include_generation=True,
        )
        session.commit()
        generation = None
        for rid in run_ids:
            run = session.get(EvaluationRun, rid)
            print(f"=== RUN {run.strategy} ({run.question_count} questions) ===")
            print(json.dumps(run.summary, indent=2))
            if run.strategy == "generation":
                generation = run
        if generation is None:
            print("no generation run to dump", file=sys.stderr)
            return 1
        payload = {
            "run_id": str(generation.id),
            "strategy": generation.strategy,
            "finished_at": generation.finished_at.isoformat() if generation.finished_at else None,
            "summary": generation.summary,
            "per_question": generation.per_question,
        }
        DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        DUMP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {DUMP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
