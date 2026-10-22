"""Context validation before generation.

Detects: zero results, uniformly weak relevance, and *possible* conflicting
evidence via a deliberately simple, documented heuristic.  The relevance gate
is a heuristic threshold on normalised retrieval scores — not a calibrated
probability.  Its limitation is documented in docs/rag_pipeline.md.
"""

from __future__ import annotations

import re
import time

from app.agents.state import RAGState
from app.retrieval.rerank import content_words

_CONTRAST_A = re.compile(r"\b(must not|may not|cannot|not allowed|prohibited|forbidden|never)\b", re.I)
_CONTRAST_B = re.compile(r"\b(may|allowed|permitted|can|optional)\b", re.I)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def detect_conflict_hint(chunks: list[dict]) -> bool:
    """Heuristic: two strong chunks from different documents that share topic
    words but carry opposing modality markers or disjoint numeric values.
    A true positive only *hints* — the generator is instructed to verify and
    report explicitly.  False positives/negatives are possible by design."""
    if len(chunks) < 2:
        return False
    top = chunks[:4]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            if a["document_id"] == b["document_id"]:
                continue
            if abs(a["score"] - b["score"]) > 0.15:
                continue
            words_a, words_b = content_words(a["text"]), content_words(b["text"])
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap < 0.15:
                continue
            modal_conflict = (
                (bool(_CONTRAST_A.search(a["text"])) and bool(_CONTRAST_B.search(b["text"])))
                or (bool(_CONTRAST_A.search(b["text"])) and bool(_CONTRAST_B.search(a["text"])))
            )
            nums_a, nums_b = set(_NUMBER_RE.findall(a["text"])), set(_NUMBER_RE.findall(b["text"]))
            numeric_conflict = bool(nums_a) and bool(nums_b) and not (nums_a & nums_b)
            if modal_conflict or numeric_conflict:
                return True
    return False


class ContextValidatorNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        chunks = state.get("reranked_results") or []
        confidence = state.get("retrieval_confidence", 0.0)
        threshold = self.settings.min_relevance_score

        if not chunks:
            status, reason = "insufficient", "No chunks were retrieved for this query."
        elif confidence < threshold:
            status, reason = (
                "insufficient",
                f"Best retrieval relevance {confidence:.2f} is below the configured "
                f"threshold {threshold:.2f}.",
            )
        else:
            status, reason = "sufficient", f"Retrieval confidence {confidence:.2f} >= {threshold:.2f}."

        conflict_hint = detect_conflict_hint(chunks) if status == "sufficient" else False
        validation = {
            "status": status,
            "reason": reason,
            "retrieval_confidence": confidence,
            "conflict_hint": conflict_hint,
            "chunk_count": len(chunks),
        }
        return {
            "context_validation": validation,
            "stages": [
                {
                    "stage": "context_validator",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": validation,
                }
            ],
        }
