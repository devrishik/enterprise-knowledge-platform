"""Document service: tenant-scoped CRUD. Every query filters on tenant_id."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.utils import parse_uuid
from app.models import Document, DocumentChunk, IngestionJob, User


def list_documents(session: Session, tenant_id) -> list[Document]:
    return list(
        session.scalars(
            select(Document).where(Document.tenant_id == tenant_id).order_by(Document.created_at.desc())
        )
    )


def get_document(session: Session, tenant_id, document_id) -> Document:
    document = session.scalar(
        select(Document).where(
            Document.id == parse_uuid(document_id, "document"),
            Document.tenant_id == tenant_id,
        )
    )
    if document is None:
        raise NotFoundError("Document not found")
    return document


def get_chunks(session: Session, tenant_id, document_id) -> list[DocumentChunk]:
    get_document(session, tenant_id, document_id)  # enforce tenancy
    return list(
        session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id, DocumentChunk.tenant_id == tenant_id)
            .order_by(DocumentChunk.chunk_index)
        )
    )


def delete_document(session: Session, tenant_id, document_id) -> None:
    document = get_document(session, tenant_id, document_id)
    session.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id, DocumentChunk.tenant_id == tenant_id
    ).delete()
    session.query(IngestionJob).filter(
        IngestionJob.document_id == document_id, IngestionJob.tenant_id == tenant_id
    ).delete()
    session.delete(document)
    session.flush()


def get_user(session: Session, user_id) -> User | None:
    return session.get(User, user_id)
