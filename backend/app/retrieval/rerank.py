"""Cross-encoder reranking.

The default reranker scores (query, chunk) pairs with a local cross-encoder
(ms-marco-MiniLM-L-6-v2 by default) — bi-encoder retrieval scores are a cheap
approximation, while the cross-encoder attends over the pair jointly and is
substantially more accurate at the top of the ranking.  ``lexical`` is a
deterministic overlap-based reranker for dependency-free tests.
"""

from __future__ import annotations

import math
import re
import threading
import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval import ScoredChunk

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "what", "which", "who",
    "how", "do", "does", "did", "can", "could", "should", "would", "i", "we",
    "you", "my", "our", "their", "there", "s", "about", "with", "as", "by",
    "at", "from", "not", "no", "if", "then", "than", "so", "such", "may",
}


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.name = f"crossencoder:{model_name}"
        self._model = None
        self._lock = threading.Lock()
        self._log = get_logger("rerank.ce")

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._log.info("loading_reranker", extra={"context": {"model": self.name}})
                    self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> tuple[list[ScoredChunk], float]:
        if not chunks:
            return [], 0.0
        started = time.perf_counter()
        self._ensure_model()
        pairs = [[query, c.text] for c in chunks]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]  # sigmoid -> [0,1]
        latency = round((time.perf_counter() - started) * 1000, 2)

        ranked = sorted(
            zip(scores, chunks), key=lambda pair: (-pair[0], pair[1].chunk_id)
        )[:top_k]
        reranked = [
            ScoredChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_filename=c.document_filename,
                text=c.text,
                score=s,
                source="reranked",
                page_number=c.page_number,
                section=c.section,
                metadata=c.metadata,
            )
            for s, c in ranked
        ]
        return reranked, latency


class LexicalOverlapReranker:
    """Deterministic fallback: query-content-word coverage in the chunk."""

    name = "lexical_overlap"

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> tuple[list[ScoredChunk], float]:
        started = time.perf_counter()
        q_words = content_words(query)
        scored: list[tuple[float, ScoredChunk]] = []
        for chunk in chunks:
            c_words = content_words(chunk.text)
            overlap = len(q_words & c_words) / len(q_words) if q_words else 0.0
            scored.append((overlap, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        latency = round((time.perf_counter() - started) * 1000, 2)
        reranked = [
            ScoredChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_filename=c.document_filename,
                text=c.text,
                score=s,
                source="reranked",
                page_number=c.page_number,
                section=c.section,
                metadata=c.metadata,
            )
            for s, c in scored[:top_k]
        ]
        return reranked, latency


_cached_reranker = None


def get_reranker():
    """Cached singleton: loading the cross-encoder takes seconds and must
    happen once per process, not per query."""
    global _cached_reranker
    if _cached_reranker is None:
        settings = get_settings()
        if settings.reranker_provider == "crossencoder":
            _cached_reranker = CrossEncoderReranker(settings.reranker_model)
        else:
            _cached_reranker = LexicalOverlapReranker()
    return _cached_reranker
