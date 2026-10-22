"""Grounded generation + citation validation + final response nodes."""

from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, Field, ValidationError

from app.agents.state import RAGState
from app.llm import get_llm_provider
from app.llm.base import LLMMessage
from app.llm.prompts import GROUNDED_ANSWER_SYSTEM, build_answer_prompt

_NO_EVIDENCE_ANSWER = (
    "I couldn't find sufficient evidence in the available documents to answer this question."
)


class Citation(BaseModel):
    ref: int = Field(ge=1)
    quote: str = ""


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    conflicting_evidence: bool = False


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    return text.strip()


def _normalize_citation(item: object) -> dict | None:
    """Keep ``quote``. A bare int becomes ``{"ref": n, "quote": ""}``."""
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return {"ref": item, "quote": ""}
    if isinstance(item, dict) and item.get("ref") is not None:
        try:
            ref = int(item["ref"])
        except (TypeError, ValueError):
            return None
        quote = item.get("quote") or ""
        if not isinstance(quote, str):
            quote = str(quote)
        return {"ref": ref, "quote": quote}
    return None


def parse_generated_answer(raw: str) -> GeneratedAnswer:
    """Parse + validate the model's JSON answer; raise ValueError on garbage."""
    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw[:200]}") from exc
    if isinstance(payload.get("citations"), list):
        payload["citations"] = [
            citation
            for item in payload["citations"]
            if (citation := _normalize_citation(item)) is not None
        ]
    return GeneratedAnswer.model_validate(payload)


class GenerateAnswerNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        provider = get_llm_provider()
        context = state.get("reranked_results") or []
        history = state.get("conversation_history") or []
        # Retrieval uses effective_query. Generation answers the user's question.
        question = state["question"]
        conflict_hint = (state.get("context_validation") or {}).get("conflict_hint", False)

        prompt = build_answer_prompt(question, _chunks(context), history, conflict_hint=conflict_hint)
        feedback = state.get("citation_feedback")
        if feedback:
            prompt = f"{prompt}\n\n[VALIDATOR FEEDBACK — fix this]\n{feedback}"
        response = provider.complete(
            [LLMMessage("system", GROUNDED_ANSWER_SYSTEM), LLMMessage("user", prompt)],
            json_mode=True,
        )
        try:
            parsed = parse_generated_answer(response.content)
        except (ValueError, ValidationError):
            parsed = GeneratedAnswer(answer=_NO_EVIDENCE_ANSWER, confidence=0.0, insufficient_evidence=True)

        usage = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) + response.usage.prompt_tokens,
            "completion_tokens": usage.get("completion_tokens", 0) + response.usage.completion_tokens,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {
            "generated": parsed.model_dump(),
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "model": response.model or provider.name,
            "token_usage": usage,
            "stages": [
                {
                    "stage": "generate_answer",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {
                        "model": response.model or provider.name,
                        "insufficient_evidence": parsed.insufficient_evidence,
                        "confidence": parsed.confidence,
                        "context_chunks": len(context),
                    },
                }
            ],
        }


def _chunks(payload: list[dict]):
    from app.retrieval import ScoredChunk

    return [
        ScoredChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            document_filename=c["document_filename"],
            text=c["text"],
            score=c["score"],
            source=c.get("source", "context"),
            page_number=c.get("page_number"),
            section=c.get("section"),
        )
        for c in payload
    ]


def quote_supported(quote: str, chunk_text: str) -> bool:
    """True when ``quote`` is a verbatim span of the chunk.

    Whitespace is collapsed before the check. The span must be at least 20
    characters. A paraphrase, a missing quote, or a short fragment fails.
    """
    normalized_quote = re.sub(r"\s+", " ", quote or "").strip()
    if len(normalized_quote) < 20:
        return False
    normalized_chunk = re.sub(r"\s+", " ", chunk_text or "")
    return normalized_quote in normalized_chunk


class CitationValidatorNode:
    """Validates citations against the retrieved context.

    Document name, page, section, chunk id, and excerpt are rebuilt from the
    retrieved chunk. A citation is kept only when its quote is an exact span
    of that chunk (see ``quote_supported``). Invalid refs or quotes trigger
    one regeneration; anything still invalid is dropped.
    """

    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        context = state.get("reranked_results") or []
        generated = state.get("generated") or {}
        attempts = state.get("generation_attempts", 0)

        invalid_refs: list[int] = []
        invalid_quotes: list[dict] = []
        citations = []
        for citation in generated.get("citations", []):
            ref = citation.get("ref")
            quote = citation.get("quote") or ""
            if not isinstance(quote, str):
                quote = str(quote)
            if not (isinstance(ref, int) and not isinstance(ref, bool) and 1 <= ref <= len(context)):
                invalid_refs.append(ref if isinstance(ref, int) and not isinstance(ref, bool) else -1)
                continue
            chunk = context[ref - 1]
            if not quote_supported(quote, chunk["text"]):
                invalid_quotes.append({"ref": ref, "quote": quote})
                continue
            citations.append(
                {
                    "ref": ref,
                    "quote": re.sub(r"\s+", " ", quote).strip(),
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_filename"],
                    "page": chunk.get("page_number"),
                    "section": chunk.get("section"),
                    "excerpt": chunk["text"][:300],
                }
            )

        # One bounded regeneration when a ref or a quote does not check out.
        regenerate = bool(invalid_refs or invalid_quotes) and attempts < 2
        validation = {
            "valid": not invalid_refs and not invalid_quotes,
            "invalid_refs": invalid_refs,
            "invalid_quotes": invalid_quotes,
            "citations": citations,
            "regenerate": regenerate,
        }
        update: RAGState = {
            "citation_validation": validation,
            "stages": [
                {
                    "stage": "citation_validator",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {
                        "valid_refs": [c["ref"] for c in citations],
                        "invalid_refs": invalid_refs,
                        "invalid_quotes": invalid_quotes,
                        "will_regenerate": regenerate,
                    },
                }
            ],
        }
        if regenerate:
            update["citation_feedback"] = _citation_feedback(invalid_refs, invalid_quotes, len(context))
        return update


def _citation_feedback(invalid_refs: list[int], invalid_quotes: list[dict], block_count: int) -> str:
    parts: list[str] = []
    if invalid_refs:
        parts.append(
            f"In your previous answer you cited block number(s) {invalid_refs} which do not "
            f"exist. Only blocks 1..{block_count} exist."
        )
    if invalid_quotes:
        named = ", ".join(f"ref {item['ref']} quote {item['quote']!r}" for item in invalid_quotes)
        parts.append(
            "These quotes are missing or are not verbatim spans of the cited block "
            f"(need at least 20 characters copied exactly): {named}."
        )
    parts.append(
        "Answer again. Cite only valid blocks, and copy each quote as exact words from that block."
    )
    return " ".join(parts)


class FinalResponseNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        validation = state.get("context_validation") or {}
        citation_validation = state.get("citation_validation") or {}
        generated = state.get("generated") or {}

        insufficient = validation.get("status") != "sufficient" or generated.get(
            "insufficient_evidence", False
        )
        if insufficient and not generated.get("answer"):
            answer = _NO_EVIDENCE_ANSWER
        else:
            answer = generated.get("answer") or _NO_EVIDENCE_ANSWER

        if validation.get("status") != "sufficient":
            answer = _NO_EVIDENCE_ANSWER

        confidence = 0.0 if validation.get("status") != "sufficient" else float(
            generated.get("confidence", 0.0)
        )

        final = {
            "answer": answer,
            "citations": [] if validation.get("status") != "sufficient" else citation_validation.get("citations", []),
            "confidence": round(confidence, 3),
            "insufficient_evidence": insufficient,
            "conflicting_evidence": bool(generated.get("conflicting_evidence", False)),
            "retrieval_confidence": validation.get("retrieval_confidence", 0.0),
            "context_chunks": validation.get("chunk_count", 0),
            "model": state.get("model", ""),
            "token_usage": state.get("token_usage", {}),
            "cost_usd": self._cost(state),
            "rewritten_question": (state.get("analysis") or {}).get("rewritten_query") or None,
            "sub_queries": (state.get("analysis") or {}).get("sub_queries", []),
        }
        return {
            "final": final,
            "stages": [
                {
                    "stage": "final_response",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {
                        "insufficient_evidence": final["insufficient_evidence"],
                        "citations": len(final["citations"]),
                    },
                }
            ],
        }

    def _cost(self, state: RAGState) -> float | None:
        usage = state.get("token_usage") or {}
        model = state.get("model", "")
        if not usage or not model:
            return None
        return self.settings.estimate_cost_usd(
            model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
