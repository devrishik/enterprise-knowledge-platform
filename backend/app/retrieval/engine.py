"""Hybrid retrieval engine.

Orchestrates dense + lexical retrieval, RRF fusion and (optionally)
cross-encoder reranking.  Used both by the LangGraph workflow and by the
evaluation harness (which compares strategies: dense / lexical / hybrid /
hybrid+rerank).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.retrieval import RetrievalResult, RetrievalStrategy
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.rerank import get_reranker


class HybridRetrievalEngine:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self.dense = DenseRetriever(session)
        self.lexical = LexicalRetriever(session)
        self.reranker = get_reranker()

    def retrieve(
        self,
        query: str,
        tenant_id,
        strategy: RetrievalStrategy = "hybrid_rerank",
    ) -> RetrievalResult:
        result = RetrievalResult(strategy=strategy, query=query)
        s = self.settings

        if strategy in ("dense", "hybrid", "hybrid_rerank"):
            result.dense, latency = self.dense.search(query, tenant_id, top_k=s.dense_top_k)
            result.latency_ms["dense"] = latency

        if strategy in ("lexical", "hybrid", "hybrid_rerank"):
            result.lexical, latency = self.lexical.search(query, tenant_id, top_k=s.lexical_top_k)
            result.latency_ms["lexical"] = latency

        if strategy in ("hybrid", "hybrid_rerank"):
            result.fused = reciprocal_rank_fusion(
                [result.dense, result.lexical], k=s.rrf_k, top_k=s.fusion_top_k
            )
            result.latency_ms.setdefault("fusion", 0.0)

        if strategy == "hybrid_rerank":
            reranked, latency = self.reranker.rerank(query, result.fused, top_k=s.rerank_top_k)
            result.reranked = reranked
            result.latency_ms["rerank"] = latency

        return result
