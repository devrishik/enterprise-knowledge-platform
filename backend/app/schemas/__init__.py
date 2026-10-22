"""API request/response schemas (Pydantic)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ----------------------------------------------------------------- documents
class DocumentOut(ORMModel):
    id: UUID
    filename: str
    doc_type: str
    title: str
    status: str
    chunk_count: int
    size_bytes: int
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict, alias="metadata_json")
    created_at: datetime
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentDetailOut(DocumentOut):
    content_hash: str


class ChunkOut(ORMModel):
    id: UUID
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None
    section: str | None = None
    metadata: dict = Field(default_factory=dict, alias="metadata_json")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class IngestionResultOut(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    status: str
    chunk_count: int
    duration_ms: float
    timing_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------- query
class CitationOut(BaseModel):
    ref: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    section: str | None = None
    excerpt: str
    quote: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_filename: str
    score: float
    source: str
    page_number: int | None = None
    section: str | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    insufficient_evidence: bool
    conflicting_evidence: bool
    rewritten_question: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    context_chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    trace_id: str
    conversation_id: str
    message_id: str
    stages: list[dict] = Field(default_factory=list)
    model: str = ""
    token_usage: dict = Field(default_factory=dict)
    cost_usd: float | None = None
    total_latency_ms: float = 0.0


# ------------------------------------------------------------ conversations
class ConversationOut(ORMModel):
    id: UUID
    title: str
    created_at: datetime


class MessageOut(ORMModel):
    id: UUID
    role: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    trace_id: UUID | None = None
    created_at: datetime


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


# ------------------------------------------------------------------- traces
class TraceOut(BaseModel):
    id: str
    question: str
    rewritten_question: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    stages: list[dict] = Field(default_factory=list)
    retrieved: dict = Field(default_factory=dict)
    final_citations: list[dict] = Field(default_factory=list)
    answer: str | None = None
    insufficient_evidence: bool = False
    confidence: float | None = None
    model: str | None = None
    token_usage: dict = Field(default_factory=dict)
    cost_usd: float | None = None
    total_latency_ms: float | None = None
    error: str | None = None
    created_at: datetime | None = None


class TraceListItem(BaseModel):
    id: str
    question: str
    total_latency_ms: float | None = None
    insufficient_evidence: bool = False
    created_at: datetime | None = None


# --------------------------------------------------------------- evaluation
class EvaluationRunRequest(BaseModel):
    strategies: list[Literal["dense", "lexical", "hybrid", "hybrid_rerank"]] = Field(
        default=["dense", "lexical", "hybrid", "hybrid_rerank"]
    )
    include_generation: bool = False
    max_questions: int | None = Field(default=None, ge=1, le=100)


class MetricSummary(BaseModel):
    recall_at_5: float | None = None
    precision_at_5: float | None = None
    mrr: float | None = None
    ndcg_at_5: float | None = None


class EvaluationRunOut(BaseModel):
    id: str
    strategy: str
    status: str
    question_count: int
    summary: dict = Field(default_factory=dict)
    per_question: list[dict] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None


# --------------------------------------------------------------------- misc
class DemoUserOut(BaseModel):
    tenant_name: str
    tenant_slug: str
    email: str
    display_name: str
    api_key: str


class HealthOut(BaseModel):
    status: str
    database: str
    vector_backend: str
    llm_provider: str
    llm_configured: bool
    embedding_provider: str
    reranker_provider: str
    details: dict[str, Any] = Field(default_factory=dict)
