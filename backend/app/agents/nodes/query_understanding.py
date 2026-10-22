"""Query understanding: analysis + rewriting.

With a generative LLM the analysis is prompt-driven.  With the deterministic
extractive provider (offline mode) it falls back to rule-based heuristics that
are unit-tested: pronoun/ellipsis follow-up detection, compound-question
splitting, and out-of-scope detection.  Clear queries are never rewritten.
"""

from __future__ import annotations

import json
import re
import time

from app.agents.state import RAGState
from app.llm import get_llm_provider
from app.llm.base import LLMMessage
from app.llm.prompts import QUERY_ANALYSIS_SYSTEM, build_analysis_prompt

_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those|they|them|he|she|such)\b", re.I)
_FOLLOWUP_RE = re.compile(r"^(what|how)\s+about\b|\b(the previous|the earlier|the above|that policy|this policy)\b", re.I)
_SHORT_QUERY_WORDS = 5
_CHITCHAT_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|good)\b", re.I)
_AND_SPLIT_RE = re.compile(r"\b(?:and also|and then|plus|;)\b", re.I)


def heuristic_analysis(question: str, history: list[dict] | None) -> dict:
    """Rule-based analysis used when no generative LLM is configured."""
    history = history or []
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")

    stripped = question.strip().rstrip("?.").strip()
    words = stripped.split()

    answerable = not _CHITCHAT_RE.match(stripped)

    needs_rewrite = False
    rewritten = ""
    if len(words) <= _SHORT_QUERY_WORDS and _PRONOUN_RE.search(question):
        needs_rewrite = True
    elif _PRONOUN_RE.search(question) and last_assistant:
        needs_rewrite = True
    elif last_user and _FOLLOWUP_RE.search(question):
        needs_rewrite = True

    if needs_rewrite and last_user:
        # stand-alone reformulation: replace bare pronoun references with the
        # topic of the previous user turn
        topic_words = [w for w in re.findall(r"[A-Za-z0-9-]+", last_user)]
        topic = " ".join(topic_words[-6:])
        rewritten = f"{question.strip()} (referring to: {topic})" if topic else question

    sub_queries: list[str] = []
    if _AND_SPLIT_RE.search(question):
        parts = [p.strip(" ,;") for p in _AND_SPLIT_RE.split(question) if len(p.strip()) > 8]
        if 2 <= len(parts) <= 3:
            sub_queries = parts

    return {
        "needs_rewrite": needs_rewrite,
        "rewritten_query": rewritten,
        "sub_queries": sub_queries,
        "answerable_from_documents": answerable,
        "reason": "heuristic analysis (offline provider)",
    }


def llm_analysis(question: str, history: list[dict] | None) -> dict:
    provider = get_llm_provider()
    response = provider.complete(
        [
            LLMMessage("system", QUERY_ANALYSIS_SYSTEM),
            LLMMessage("user", build_analysis_prompt(question, history)),
        ],
        json_mode=True,
    )
    try:
        parsed = json.loads(_strip_fences(response.content))
    except json.JSONDecodeError:
        return {
            "needs_rewrite": False,
            "rewritten_query": "",
            "sub_queries": [],
            "answerable_from_documents": True,
            "reason": "analysis parse failure; treating question as clear",
        }
    return {
        "needs_rewrite": bool(parsed.get("needs_rewrite", False)),
        "rewritten_query": str(parsed.get("rewritten_query", "") or ""),
        "sub_queries": [str(q) for q in parsed.get("sub_queries", [])][:3],
        "answerable_from_documents": bool(parsed.get("answerable_from_documents", True)),
        "reason": str(parsed.get("reason", "")),
        "_usage": response.usage.to_dict(),
        "_model": response.model,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    return text.strip()


class QueryAnalyzerNode:
    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        question = state["question"]
        history = state.get("conversation_history") or []
        provider = get_llm_provider()

        analysis = (
            llm_analysis(question, history)
            if provider.is_generative_llm
            else heuristic_analysis(question, history)
        )
        usage = analysis.pop("_usage", None)
        model = analysis.pop("_model", None)

        effective = analysis.get("rewritten_query") or question
        update: RAGState = {
            "analysis": analysis,
            "effective_query": effective,
            "stages": [
                {
                    "stage": "query_analyzer",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": analysis,
                }
            ],
        }
        if usage:
            update["token_usage"] = usage
        if model:
            update["model"] = model
        return update


class QueryRewriterNode:
    """Applies the rewrite decided by the analyzer (no rewrite for clear queries)."""

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        analysis = state.get("analysis", {})
        question = state["question"]
        rewritten = analysis.get("rewritten_query") if analysis.get("needs_rewrite") else ""
        effective = rewritten or question
        return {
            "effective_query": effective,
            "stages": [
                {
                    "stage": "query_rewriter",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {"rewritten": bool(rewritten), "rewritten_query": rewritten},
                }
            ],
        }
