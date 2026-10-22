"""Generation evaluation: faithfulness, answer relevancy, citation correctness.

Two modes, clearly labelled in every result row:
- ``llm_judge``   — an LLM (the configured generative provider) judges
                    faithfulness/relevancy. LLM-based evaluation is itself
                    imperfect; treat scores as estimates, not ground truth.
- ``heuristic``   — deterministic keyword-coverage measures used when no
                    generative LLM is configured. Mechanically reproducible,
                    semantically shallow; they measure pipeline mechanics, not
                    answer quality.

Citation correctness is always deterministic (no judge involved).

A canonical refusal on a question marked unanswerable is scored faithful by
rule (``mode="rule"``) and is not sent to the judge. A refusal has no claims
to check, and the previous judge call treated it as unfaithful. The same
refusal on a question marked answerable is still judged: refusing when the
corpus should answer is a generation failure.
"""

from __future__ import annotations

import json
import re

from app.core.logging import get_logger
from app.llm import get_llm_provider
from app.llm.base import LLMMessage
from app.llm.prompts import (
    FAITHFULNESS_JUDGE_SYSTEM,
    RELEVANCY_JUDGE_SYSTEM,
    build_judge_prompt,
)
from app.retrieval.rerank import content_words

REFUSAL_MARKER = "couldn't find sufficient evidence"
_CANONICAL_REFUSAL = (
    "i couldn't find sufficient evidence in the available documents to answer this question."
)


def citation_correctness(
    answer: str,
    citations: list[dict],
    retrieved_chunk_ids: set[str],
    expected_answerable: bool,
) -> dict:
    """Deterministic checks: citations must reference actually-retrieved chunks;
    unanswerable questions must produce an explicit refusal."""
    valid_citations = all(c.get("chunk_id") in retrieved_chunk_ids for c in citations)
    refused = REFUSAL_MARKER in answer.lower()
    correct_behavior = refused if not expected_answerable else (not refused)
    return {
        "citations_reference_retrieved_chunks": valid_citations,
        "refusal_behavior_correct": correct_behavior,
        "citation_count": len(citations),
        "pass": valid_citations and correct_behavior,
    }


def _judge(prompt_system: str, user_prompt: str) -> dict | None:
    provider = get_llm_provider()
    if not provider.is_generative_llm:
        return None
    try:
        response = provider.complete(
            [LLMMessage("system", prompt_system), LLMMessage("user", user_prompt)], json_mode=True
        )
        return json.loads(response.content)
    except Exception as exc:
        get_logger("evaluation.judge").warning(
            "judge_failed", extra={"context": {"error": str(exc)}}
        )
        return None


def is_canonical_refusal(answer: str) -> bool:
    """True when the answer is the pipeline's refusal sentence and nothing else."""
    normalized = re.sub(r"\s+", " ", answer.strip().lower()).rstrip(".")
    return normalized == _CANONICAL_REFUSAL.rstrip(".")


def faithfulness(
    answer: str,
    context_text: str,
    question: str = "",
    expected_answerable: bool | None = None,
) -> dict:
    if expected_answerable is False and is_canonical_refusal(answer):
        return {
            "mode": "rule",
            "faithful": True,
            "reason": "expected_refusal",
            "explanation": "Canonical refusal on a question marked unanswerable.",
        }
    judged = _judge(
        FAITHFULNESS_JUDGE_SYSTEM, build_judge_prompt(question, answer, context_text)
    )
    if judged is not None:
        return {"mode": "llm_judge", "faithful": bool(judged.get("faithful", False)),
                "supported_claims": judged.get("supported_claims"),
                "unsupported_claims": judged.get("unsupported_claims"),
                "explanation": judged.get("explanation", "")[:300]}
    # Heuristic: share of answer sentences with keyword coverage in the context.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s) > 20]
    if not sentences:
        return {"mode": "heuristic", "faithful": True, "supported_fraction": 1.0}
    context_words = content_words(context_text)
    supported = sum(
        1
        for s in sentences
        if len(content_words(s) & context_words) / max(len(content_words(s)), 1) >= 0.6
    )
    return {
        "mode": "heuristic",
        "faithful": supported == len(sentences),
        "supported_fraction": round(supported / len(sentences), 3),
    }


def answer_relevancy(question: str, answer: str) -> dict:
    judged = _judge(RELEVANCY_JUDGE_SYSTEM, build_judge_prompt(question, answer))
    if judged is not None:
        return {"mode": "llm_judge", "score": judged.get("relevancy_score"),
                "explanation": judged.get("explanation", "")[:300]}
    q_words = content_words(question)
    a_words = content_words(answer)
    coverage = len(q_words & a_words) / len(q_words) if q_words else 0.0
    return {"mode": "heuristic", "score": round(coverage * 5), "keyword_coverage": round(coverage, 3)}
