"""Deterministic extractive provider (offline, no external calls).

This provider is *not* an LLM.  It parses the same grounded-answer prompt a
real model receives (labelled [n] context blocks + question), scores context
sentences by query-term coverage, and composes an answer exclusively from
extracted sentences with citations referring to the supplied blocks.  When no
sentence clears the coverage floor, it returns ``insufficient_evidence``.

Purpose: an honest, reproducible fallback for demos and pipeline tests when no
API key is configured — it demonstrates the grounding/citation contract
mechanically.  Answer *quality* claims must be measured with a real LLM.
"""

from __future__ import annotations

import json
import re

from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.retrieval.rerank import content_words

_BLOCK_RE = re.compile(r"\[(\d+)\]\s*\(([^)]*)\)\n(.*?)(?=\n\n\[\d+\]|\n*\[QUESTION\]|\Z)", re.S)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_QUESTION_RE = re.compile(r"\[QUESTION\]\n(.*)", re.S)
_MIN_COVERAGE = 0.25  # at least a quarter of query content words in the sentence


def _is_heading_echo(sentence: str) -> bool:
    """Short all-title-case fragments are structural headings, not content."""
    words = sentence.split()
    if not (2 <= len(words) <= 6):
        return False
    return all(w[:1].isupper() or not w[:1].isalpha() for w in words)


class ExtractiveProvider(LLMProvider):
    name = "extractive"
    is_generative_llm = False

    def complete(self, messages: list[LLMMessage], *, json_mode: bool = False) -> LLMResponse:
        prompt = messages[-1].content if messages else ""
        result = self.answer_grounded(prompt)
        return LLMResponse(content=json.dumps(result), model=self.name)

    # ------------------------------------------------------------------ core
    def answer_grounded(self, prompt: str) -> dict:
        question_match = _QUESTION_RE.search(prompt)
        question = (question_match.group(1).strip() if question_match else "").splitlines()[0] if question_match else ""
        q_words = content_words(question)

        blocks: list[tuple[int, str, str]] = []  # (index, header, text)
        for match in _BLOCK_RE.finditer(prompt):
            blocks.append((int(match.group(1)), match.group(2), match.group(3)))

        if not blocks or not question:
            return {
                "answer": "I couldn't find sufficient evidence in the available documents to answer this question.",
                "citations": [],
                "confidence": 0.0,
                "insufficient_evidence": True,
                "conflicting_evidence": False,
            }

        scored: list[tuple[float, int, str]] = []  # (coverage, block_index, sentence)
        for index, _header, text in blocks:
            for sentence in _SENTENCE_RE.split(text.replace("\n", " ")):
                sentence = sentence.strip()
                if len(sentence) < 25 or _is_heading_echo(sentence):
                    continue
                s_words = content_words(sentence)
                coverage = len(q_words & s_words) / len(q_words) if q_words else 0.0
                if coverage >= _MIN_COVERAGE:
                    scored.append((coverage, index, sentence))

        if not scored:
            return {
                "answer": "I couldn't find sufficient evidence in the available documents to answer this question.",
                "citations": [],
                "confidence": 0.0,
                "insufficient_evidence": True,
                "conflicting_evidence": False,
            }

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = scored[:4]
        seen_sentences: set[str] = set()
        parts: list[str] = []
        citations: list[dict] = []
        for _coverage, index, sentence in chosen:
            key = sentence[:80]
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            parts.append(sentence)
            citations.append({"ref": index, "quote": sentence})
        confidence = round(sum(c for c, _, _ in chosen) / len(chosen), 3)
        answer = " ".join(parts)
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "insufficient_evidence": False,
            "conflicting_evidence": False,
        }
