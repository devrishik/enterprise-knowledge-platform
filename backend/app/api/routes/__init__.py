"""API routes: documents, query, conversations, evaluation, traces, health."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import schemas
from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import FileTooLarge
from app.db.session import get_db
from app.ingestion.pipeline import run_ingestion
from app.ingestion.parsers import safe_filename
from app.models import Document, EvaluationRun, Message, QueryTrace, User
from app.services import documents as doc_service
from app.services import query_service

router = APIRouter(prefix="/api")


@router.get("/auth/demo-users", response_model=list[schemas.DemoUserOut], tags=["auth"])
def list_demo_users(session: Session = Depends(get_db)) -> list[schemas.DemoUserOut]:
    from app.services.seed import DEMO_TENANTS

    out: list[schemas.DemoUserOut] = []
    for tenant in DEMO_TENANTS:
        for user in tenant["users"]:
            out.append(
                schemas.DemoUserOut(
                    tenant_name=tenant["name"],
                    tenant_slug=tenant["slug"],
                    email=user["email"],
                    display_name=user["display_name"],
                    api_key=user["api_key"],
                )
            )
    return out


# ------------------------------------------------------------------ documents
@router.post(
    "/documents/upload",
    response_model=schemas.IngestionResultOut,
    status_code=201,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.IngestionResultOut:
    settings = get_settings()
    payload = await file.read()
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLarge(f"File exceeds the {settings.max_upload_mb} MB limit.")
    if not payload:
        raise FileTooLarge("Uploaded file is empty.")
    filename = safe_filename(file.filename or "upload")
    result = run_ingestion(
        session, filename=filename, payload=payload, tenant_id=user.tenant_id, user_id=user.id
    )
    return schemas.IngestionResultOut(**result.__dict__)


@router.get("/documents", response_model=list[schemas.DocumentOut], tags=["documents"])
def list_documents(
    user: User = Depends(get_current_user), session: Session = Depends(get_db)
) -> list[schemas.DocumentOut]:
    docs = doc_service.list_documents(session, user.tenant_id)
    return [schemas.DocumentOut.model_validate(d) for d in docs]


@router.get("/documents/{document_id}", response_model=schemas.DocumentDetailOut, tags=["documents"])
def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.DocumentDetailOut:
    doc = doc_service.get_document(session, user.tenant_id, document_id)
    return schemas.DocumentDetailOut.model_validate(doc)


@router.get("/documents/{document_id}/chunks", response_model=list[schemas.ChunkOut], tags=["documents"])
def get_document_chunks(
    document_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> list[schemas.ChunkOut]:
    chunks = doc_service.get_chunks(session, user.tenant_id, document_id)
    return [schemas.ChunkOut.model_validate(c) for c in chunks]


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> None:
    doc_service.delete_document(session, user.tenant_id, document_id)


# ---------------------------------------------------------------------- query
@router.post("/query", response_model=schemas.QueryResponse, tags=["query"])
def run_query(
    request: schemas.QueryRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.QueryResponse:
    payload = query_service.run_query(
        session, question=request.question, user=user, conversation_id=request.conversation_id
    )
    return schemas.QueryResponse(**payload)


@router.get("/conversations", response_model=list[schemas.ConversationOut], tags=["conversations"])
def list_conversations(
    user: User = Depends(get_current_user), session: Session = Depends(get_db)
) -> list[schemas.ConversationOut]:
    rows = query_service.list_conversations(session, user.tenant_id, user.id)
    return [schemas.ConversationOut.model_validate(c) for c in rows]


@router.get("/conversations/{conversation_id}", response_model=schemas.ConversationDetailOut, tags=["conversations"])
def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.ConversationDetailOut:
    conversation = query_service.get_conversation(session, user.tenant_id, user.id, conversation_id)
    if conversation is None:
        from app.core.errors import NotFoundError

        raise NotFoundError("Conversation not found")
    messages = list(
        session.scalars(
            select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)
        )
    )
    out = schemas.ConversationDetailOut.model_validate(conversation)
    out.messages = [schemas.MessageOut.model_validate(m) for m in messages]
    return out


# --------------------------------------------------------------------- traces
@router.get("/traces", response_model=list[schemas.TraceListItem], tags=["traces"])
def list_traces(
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> list[schemas.TraceListItem]:
    rows = session.scalars(
        select(QueryTrace)
        .where(QueryTrace.tenant_id == user.tenant_id)
        .order_by(QueryTrace.created_at.desc())
        .limit(limit)
    )
    return [
        schemas.TraceListItem(
            id=str(t.id),
            question=t.question,
            total_latency_ms=t.total_latency_ms,
            insufficient_evidence=t.insufficient_evidence,
            created_at=t.created_at,
        )
        for t in rows
    ]


@router.get("/traces/{trace_id}", response_model=schemas.TraceOut, tags=["traces"])
def get_trace(
    trace_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.TraceOut:
    from app.core.errors import NotFoundError
    from app.core.utils import parse_uuid

    trace = session.scalar(
        select(QueryTrace).where(
            QueryTrace.id == parse_uuid(trace_id, "trace"), QueryTrace.tenant_id == user.tenant_id
        )
    )
    if trace is None:
        raise NotFoundError("Trace not found")
    return schemas.TraceOut(
        id=str(trace.id),
        question=trace.question,
        rewritten_question=trace.rewritten_question,
        sub_queries=trace.sub_queries or [],
        stages=trace.stages or [],
        retrieved=trace.retrieved or {},
        final_citations=trace.final_citations or [],
        answer=trace.answer,
        insufficient_evidence=trace.insufficient_evidence,
        confidence=trace.confidence,
        model=trace.model,
        token_usage=trace.token_usage or {},
        cost_usd=trace.cost_usd,
        total_latency_ms=trace.total_latency_ms,
        error=trace.error,
        created_at=trace.created_at,
    )


# ----------------------------------------------------------------- evaluation
@router.post("/evaluation/run", status_code=202, tags=["evaluation"])
def start_evaluation(
    request: schemas.EvaluationRunRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> dict:
    from app.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(session)
    runs = runner.run(
        tenant_id=user.tenant_id,
        strategies=request.strategies,
        include_generation=request.include_generation,
        max_questions=request.max_questions,
    )
    return {
        "run_ids": [str(r) for r in runs],
        "detail": "Evaluation runs completed and stored.",
    }


@router.get("/evaluation/runs", response_model=list[schemas.EvaluationRunOut], tags=["evaluation"])
def list_evaluation_runs(
    user: User = Depends(get_current_user), session: Session = Depends(get_db)
) -> list[schemas.EvaluationRunOut]:
    rows = session.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.tenant_id == user.tenant_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(50)
    )
    return [
        schemas.EvaluationRunOut(
            id=str(r.id),
            strategy=r.strategy,
            status=r.status,
            question_count=r.question_count,
            summary=r.summary or {},
            per_question=[],
            started_at=r.started_at,
            finished_at=r.finished_at,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/evaluation/runs/{run_id}", response_model=schemas.EvaluationRunOut, tags=["evaluation"])
def get_evaluation_run(
    run_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> schemas.EvaluationRunOut:
    from app.core.errors import NotFoundError
    from app.core.utils import parse_uuid

    run = session.scalar(
        select(EvaluationRun).where(
            EvaluationRun.id == parse_uuid(run_id, "evaluation run"), EvaluationRun.tenant_id == user.tenant_id
        )
    )
    if run is None:
        raise NotFoundError("Evaluation run not found")
    return schemas.EvaluationRunOut(
        id=str(run.id),
        strategy=run.strategy,
        status=run.status,
        question_count=run.question_count,
        summary=run.summary or {},
        per_question=run.per_question or [],
        started_at=run.started_at,
        finished_at=run.finished_at,
        created_at=run.created_at,
    )


# ----------------------------------------------------------------- dashboard
@router.get("/metrics/summary", tags=["dashboard"])
def metrics_summary(
    user: User = Depends(get_current_user), session: Session = Depends(get_db)
) -> dict:
    tenant = user.tenant_id
    doc_count = session.scalar(select(func.count(Document.id)).where(Document.tenant_id == tenant))
    from app.models import DocumentChunk, IngestionJob

    total_chunks = session.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.tenant_id == tenant)
    )
    total_traces = session.scalar(
        select(func.count(QueryTrace.id)).where(QueryTrace.tenant_id == tenant)
    )
    completed_jobs = session.scalar(
        select(func.count(IngestionJob.id)).where(
            IngestionJob.tenant_id == tenant, IngestionJob.status == "completed"
        )
    )
    failed_jobs = session.scalar(
        select(func.count(IngestionJob.id)).where(
            IngestionJob.tenant_id == tenant, IngestionJob.status == "failed"
        )
    )
    avg_latency = session.scalar(
        select(func.avg(QueryTrace.total_latency_ms)).where(QueryTrace.tenant_id == tenant)
    )
    return {
        "documents": doc_count or 0,
        "chunks": total_chunks or 0,
        "queries": total_traces or 0,
        "ingestion_jobs_completed": completed_jobs or 0,
        "ingestion_jobs_failed": failed_jobs or 0,
        "avg_query_latency_ms": round(float(avg_latency), 1) if avg_latency else None,
    }
