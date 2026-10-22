"""Typed state for the LangGraph agentic RAG workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RAGState(TypedDict, total=False):
    # inputs
    question: str
    tenant_id: str
    conversation_history: list[dict]
    # query understanding
    analysis: dict  # {needs_rewrite, rewritten_query, sub_queries, answerable_from_documents, reason}
    effective_query: str
    # retrieval (separate keys: parallel nodes write without conflicts)
    dense_results: list[dict]
    lexical_results: list[dict]
    fused_results: list[dict]
    reranked_results: list[dict]
    retrieval_confidence: float
    retry_count: int
    # validation
    context_validation: dict  # {status, reason, conflict_hint}
    # generation
    generated: dict  # {answer, citations, confidence, insufficient_evidence, conflicting_evidence}
    generation_attempts: int
    citation_feedback: str
    citation_validation: dict  # {valid, invalid_refs, invalid_quotes, citations, regenerate}
    # outputs / bookkeeping
    final: dict
    model: str
    token_usage: dict
    error: str | None
    # per-node latency + details; additive across parallel branches
    stages: Annotated[list[dict], operator.add]
