"""Retrieval graph nodes: dense, lexical, fusion, rerank.

LangGraph executes the dense and lexical nodes in parallel worker threads, so
each opens its own short-lived database session — a SQLAlchemy Session must
never be shared across threads.  Sessions read committed data only, which is
exactly what retrieval needs.

Each retrieval node runs over the effective query plus any sub-queries from
the analyzer, merging results across sub-queries (best score per chunk) —
this implements multi-query retrieval for decomposed questions without
complicating the graph topology.
"""

from __future__ import annotations

import time

from app.agents.state import RAGState
from app.retrieval import ScoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.rerank import get_reranker


def _merge(merged: dict[str, ScoredChunk], chunks: list[ScoredChunk]) -> None:
    for chunk in chunks:
        current = merged.get(chunk.chunk_id)
        if current is None or chunk.score > current.score:
            merged[chunk.chunk_id] = chunk


def _ranked(merged: dict[str, ScoredChunk]) -> list[ScoredChunk]:
    return sorted(merged.values(), key=lambda c: (-c.score, c.chunk_id))


def _queries(state: RAGState) -> list[str]:
    effective = state.get("effective_query") or state["question"]
    sub_queries = (state.get("analysis") or {}).get("sub_queries") or []
    return [effective, *[q for q in sub_queries if q and q.strip()]]


def _new_session():
    from app.db.session import get_session_factory

    return get_session_factory()()


class RetrieveDenseNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        tenant_id = state["tenant_id"]
        merged: dict[str, ScoredChunk] = {}
        session = _new_session()
        try:
            retriever = DenseRetriever(session)
            for query in _queries(state):
                chunks, _latency = retriever.search(query, tenant_id, top_k=self.settings.dense_top_k)
                _merge(merged, chunks)
        finally:
            session.close()
        results = _ranked(merged)[: self.settings.dense_top_k]
        return {
            "dense_results": [c.to_dict() for c in results],
            "stages": [
                {
                    "stage": "retrieve_dense",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {"queries": _queries(state), "result_count": len(results)},
                }
            ],
        }


class RetrieveLexicalNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        tenant_id = state["tenant_id"]
        merged: dict[str, ScoredChunk] = {}
        session = _new_session()
        try:
            retriever = LexicalRetriever(session)
            for query in _queries(state):
                chunks, _latency = retriever.search(query, tenant_id, top_k=self.settings.lexical_top_k)
                _merge(merged, chunks)
        finally:
            session.close()
        results = _ranked(merged)[: self.settings.lexical_top_k]
        return {
            "lexical_results": [c.to_dict() for c in results],
            "stages": [
                {
                    "stage": "retrieve_lexical",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {"queries": _queries(state), "result_count": len(results)},
                }
            ],
        }


def _from_dicts(payload: list[dict], source: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            document_filename=c["document_filename"],
            text=c["text"],
            score=c["score"],
            source=source,
            page_number=c.get("page_number"),
            section=c.get("section"),
            metadata=c.get("metadata") or {},
        )
        for c in payload
    ]


class FuseResultsNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        dense = _from_dicts(state.get("dense_results") or [], "dense")
        lexical = _from_dicts(state.get("lexical_results") or [], "lexical")
        fused = reciprocal_rank_fusion([dense, lexical], k=self.settings.rrf_k, top_k=self.settings.fusion_top_k)
        return {
            "fused_results": [c.to_dict() for c in fused],
            "stages": [
                {
                    "stage": "fuse_results",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {
                        "dense_in": len(dense),
                        "lexical_in": len(lexical),
                        "fused_out": len(fused),
                        "method": "rrf",
                        "rrf_k": self.settings.rrf_k,
                    },
                }
            ],
        }


class RerankNode:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, state: RAGState) -> RAGState:
        started = time.perf_counter()
        fused = _from_dicts(state.get("fused_results") or [], "fused")
        query = state.get("effective_query") or state["question"]
        reranker = get_reranker()
        reranked, rerank_latency = reranker.rerank(query, fused, top_k=self.settings.rerank_top_k)
        confidence = max((c.score for c in reranked), default=0.0)
        return {
            "reranked_results": [c.to_dict() for c in reranked],
            "retrieval_confidence": round(float(confidence), 4),
            "stages": [
                {
                    "stage": "rerank",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "details": {
                        "model": getattr(reranker, "name", "unknown"),
                        "candidates": len(fused),
                        "selected": len(reranked),
                        "top_score": round(float(confidence), 4),
                    },
                }
            ],
        }
