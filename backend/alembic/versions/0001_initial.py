"""Initial schema: tenants, users, documents, chunks (+embedding/tsvector),
ingestion jobs, conversations, messages, traces, evaluation tables.

The ``document_chunks.embedding`` column type follows the configured vector
backend (pgvector native column vs double precision[] compatibility mode) via
app.db.types.embedding_sql_type — the ORM and this migration share one source
of truth, so schema and runtime can never diverge.  In pgvector mode the
pgvector extension is created here and an HNSW cosine index is built.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from app.core.config import get_settings
from app.db.types import embedding_sql_type

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

USE_PGVECTOR = get_settings().vector_backend == "pgvector"


def upgrade() -> None:
    if USE_PGVECTOR:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("api_key", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "query_traces",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("rewritten_question", sa.Text(), nullable=True),
        sa.Column("sub_queries", JSONB(), nullable=False),
        sa.Column("stages", JSONB(), nullable=False),
        sa.Column("retrieved", JSONB(), nullable=False),
        sa.Column("final_citations", JSONB(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("insufficient_evidence", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("token_usage", JSONB(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("total_latency_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("uploader_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("doc_type", sa.String(20), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_tenant_created", "documents", ["tenant_id", "created_at"])
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(500), nullable=True),
        sa.Column("embedding", embedding_sql_type(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # Lexical retrieval: maintained by PostgreSQL as a stored generated column.
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', coalesce(section, '') || ' ' || text)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_tenant_document", "document_chunks", ["tenant_id", "document_id"])
    op.create_index("ix_chunks_tsv", "document_chunks", ["tsv"], postgresql_using="gin")
    if USE_PGVECTOR:
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("stage", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("timing_ms", JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("conversation_id", sa.Uuid(), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", JSONB(), nullable=False),
        sa.Column("trace_id", sa.Uuid(), sa.ForeignKey("query_traces.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "evaluation_questions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(30), nullable=False),
        sa.Column("expected_doc_filenames", JSONB(), nullable=False),
        sa.Column("expected_answer_keywords", JSONB(), nullable=False),
        sa.Column("expected_answerable", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("question_count", sa.Integer(), nullable=False),
        sa.Column("summary", JSONB(), nullable=False),
        sa.Column("per_question", JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_traces_tenant_created", "query_traces", ["tenant_id", "created_at"])


def downgrade() -> None:
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_questions")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_chunks_tsv", table_name="document_chunks")
    if USE_PGVECTOR:
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("query_traces")
    op.drop_table("users")
    op.drop_table("tenants")
