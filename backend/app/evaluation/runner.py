"""Evaluation runner: retrieval metrics per strategy + generation metrics +
latency benchmarks, all produced by actual execution and stored per run."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation import generation_eval
from app.evaluation.metrics import (
    average,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from app.models import EvaluationQuestion, EvaluationRun, User
from app.retrieval.engine import HybridRetrievalEngine
from app.services.query_service import run_query

_STRATEGIES = ["dense", "lexical", "hybrid", "hybrid_rerank"]


class EvaluationRunner:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self.log = get_logger("evaluation")

    def _questions(self, max_questions: int | None) -> list[EvaluationQuestion]:
        rows = list(session_questions(self.session))
        limit = max_questions or self.settings.evaluation_max_questions
        return rows[:limit]

    def run(
        self,
        *,
        tenant_id,
        strategies: list[str],
        include_generation: bool = False,
        max_questions: int | None = None,
    ) -> list:
        questions = self._questions(max_questions)
        if not questions:
            self.log.warning("evaluation_no_questions")
            return []

        run_ids = []
        for strategy in strategies:
            if strategy not in _STRATEGIES:
                continue
            run = self._run_retrieval_strategy(tenant_id, strategy, questions)
            run_ids.append(run.id)
        if include_generation:
            run = self._run_generation(tenant_id, questions)
            if run is not None:
                run_ids.append(run.id)
        return run_ids

    # ------------------------------------------------------------- retrieval
    def _run_retrieval_strategy(
        self, tenant_id, strategy: str, questions: list[EvaluationQuestion]
    ) -> EvaluationRun:
        engine = HybridRetrievalEngine(self.session)
        run = EvaluationRun(tenant_id=tenant_id, strategy=strategy, status="running",
                            question_count=len(questions))
        self.session.add(run)
        self.session.flush()

        recalls, precisions, mrrs, ndcgs = [], [], [], []
        latency_lists: dict[str, list[float]] = {}
        per_question: list[dict] = []

        for question in questions:
            result = engine.retrieve(question.question, tenant_id, strategy=strategy)
            retrieved_docs = [c.document_filename for c in result.final_chunks()]
            relevant = set(question.expected_doc_filenames or [])

            recall = recall_at_k(retrieved_docs, relevant, 5)
            precision = precision_at_k(retrieved_docs, relevant, 5)
            reciprocal = mean_reciprocal_rank(retrieved_docs, relevant)
            ndcg = ndcg_at_k(retrieved_docs, relevant, 5)
            recalls.append(recall)
            precisions.append(precision)
            mrrs.append(reciprocal)
            ndcgs.append(ndcg)

            for stage, latency in (result.latency_ms or {}).items():
                latency_lists.setdefault(stage, []).append(latency)

            per_question.append(
                {
                    "question_id": str(question.id),
                    "question": question.question,
                    "question_type": question.question_type,
                    "retrieved_docs": retrieved_docs[:10],
                    "relevant_docs": sorted(relevant),
                    "recall@5": recall,
                    "precision@5": precision,
                    "mrr": reciprocal,
                    "ndcg@5": ndcg,
                    "latency_ms": result.latency_ms,
                }
            )

        benchmarks = {
            stage: {"p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}
            for stage, values in latency_lists.items()
        }
        summary = {
            "recall@5": average(recalls),
            "precision@5": average(precisions),
            "mrr": average(mrrs),
            "ndcg@5": average(ndcgs),
            "question_count": len(questions),
            "benchmarks": benchmarks,
        }
        run.status = "completed"
        run.summary = summary
        run.per_question = per_question
        run.finished_at = datetime.now(timezone.utc)
        self.session.flush()
        self.log.info("evaluation_run_completed",
                      extra={"context": {"strategy": strategy, "summary": summary}})
        return run

    # ------------------------------------------------------------- generation
    def _run_generation(self, tenant_id, questions: list[EvaluationQuestion]) -> EvaluationRun | None:
        user = self.session.scalar(select(User).where(User.tenant_id == tenant_id).limit(1))
        if user is None:
            return None
        run = EvaluationRun(tenant_id=tenant_id, strategy="generation", status="running",
                            question_count=len(questions))
        self.session.add(run)
        self.session.flush()

        per_question: list[dict] = []
        faithful_count = 0
        relevancy_scores: list[float] = []
        answerable_faithful: list[float] = []
        answerable_relevancy: list[float] = []
        citation_pass_count = 0
        latencies: list[float] = []

        for question in questions:
            started = time.perf_counter()
            payload = run_query(self.session, question=question.question, user=user)
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            latencies.append(total_ms)

            context_chunks = payload["context_chunks"]
            retrieved_ids = {c["chunk_id"] for c in context_chunks}
            context_text = "\n\n".join(c["text"] for c in context_chunks)

            faith = generation_eval.faithfulness(
                payload["answer"],
                context_text,
                question=question.question,
                expected_answerable=question.expected_answerable,
            )
            relev = generation_eval.answer_relevancy(question.question, payload["answer"])
            citation = generation_eval.citation_correctness(
                payload["answer"], payload["citations"], retrieved_ids,
                expected_answerable=question.expected_answerable,
            )
            faithful = bool(faith.get("faithful"))
            relevancy = float(relev.get("score") or 0)
            faithful_count += 1 if faithful else 0
            relevancy_scores.append(relevancy)
            citation_pass_count += 1 if citation["pass"] else 0
            if question.expected_answerable:
                answerable_faithful.append(1.0 if faithful else 0.0)
                answerable_relevancy.append(relevancy)

            per_question.append(
                {
                    "question_id": str(question.id),
                    "question": question.question,
                    "question_type": question.question_type,
                    "expected_answerable": question.expected_answerable,
                    "expected_doc_filenames": list(question.expected_doc_filenames or []),
                    "expected_answer_keywords": list(question.expected_answer_keywords or []),
                    "answer": payload["answer"],
                    "insufficient_evidence": payload["insufficient_evidence"],
                    "model": payload.get("model", ""),
                    "context": [
                        {
                            "chunk_id": c["chunk_id"],
                            "document_filename": c["document_filename"],
                            "section": c.get("section"),
                            "page_number": c.get("page_number"),
                            "score": c.get("score"),
                            "text": c["text"],
                        }
                        for c in context_chunks
                    ],
                    "faithfulness": faith,
                    "relevancy": relev,
                    "citation_correctness": citation,
                    "total_latency_ms": total_ms,
                    "trace_id": payload["trace_id"],
                }
            )

        summary = {
            "faithfulness_rate": round(faithful_count / len(questions), 4) if questions else 0,
            "avg_relevancy": average(relevancy_scores),
            "faithfulness_rate_answerable": average(answerable_faithful),
            "avg_relevancy_answerable": average(answerable_relevancy),
            "answerable_count": len(answerable_faithful),
            "citation_correctness_rate": round(citation_pass_count / len(questions), 4) if questions else 0,
            "question_count": len(questions),
            "evaluator": "llm_judge+rules" if self._generative() else "heuristic+rules",
            "benchmarks": {
                "total_query": {"p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95)}
            },
        }
        run.status = "completed"
        run.summary = summary
        run.per_question = per_question
        run.finished_at = datetime.now(timezone.utc)
        self.session.flush()
        return run

    def _generative(self) -> bool:
        from app.llm import get_llm_provider

        return get_llm_provider().is_generative_llm


def session_questions(session: Session):
    return session.scalars(
        select(EvaluationQuestion).order_by(EvaluationQuestion.created_at, EvaluationQuestion.id)
    )
