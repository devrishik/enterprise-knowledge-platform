"""Query service: runs the agentic workflow and persists traces + messages."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents import RAGWorkflow
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.utils import parse_uuid
from app.models import Conversation, Message, QueryTrace


def get_history(session: Session, conversation_id) -> list[dict]:
    messages = list(
        session.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
        )
    )
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in messages
    ]


def run_query(
    session: Session,
    *,
    question: str,
    user,
    conversation_id: str | None = None,
) -> dict:
    """Execute the RAG workflow, persist trace + conversation, return payload."""
    settings = get_settings()
    log = get_logger("query_service")
    started = time.perf_counter()

    conversation = None
    if conversation_id:
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.id == parse_uuid(conversation_id, "conversation"),
                Conversation.tenant_id == user.tenant_id,
            )
        )
    if conversation is None:
        conversation = Conversation(
            tenant_id=user.tenant_id, user_id=user.id, title=question[:200]
        )
        session.add(conversation)
        session.flush()

    history = get_history(session, conversation.id)

    workflow = RAGWorkflow(session)
    try:
        state = workflow.run(question=question, tenant_id=user.tenant_id, conversation_history=history)
    except Exception as exc:
        log.exception("workflow_failed")
        trace = QueryTrace(
            tenant_id=user.tenant_id,
            user_id=user.id,
            conversation_id=conversation.id,
            question=question,
            error=f"{type(exc).__name__}: {exc}"[:1000],
            total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        session.add(trace)
        session.flush()
        raise

    final = state.get("final", {})
    stages = state.get("stages", [])
    total_latency = round((time.perf_counter() - started) * 1000, 2)

    citations = final.get("citations", [])
    trace = QueryTrace(
        tenant_id=user.tenant_id,
        user_id=user.id,
        conversation_id=conversation.id,
        question=question,
        rewritten_question=final.get("rewritten_question"),
        sub_queries=final.get("sub_queries", []),
        stages=stages,
        retrieved={
            "dense": state.get("dense_results", []),
            "lexical": state.get("lexical_results", []),
            "fused": state.get("fused_results", []),
            "reranked": state.get("reranked_results", []),
        },
        final_citations=citations,
        answer=final.get("answer"),
        insufficient_evidence=final.get("insufficient_evidence", False),
        confidence=final.get("confidence"),
        model=final.get("model"),
        token_usage=final.get("token_usage", {}),
        cost_usd=final.get("cost_usd"),
        total_latency_ms=total_latency,
    )
    session.add(trace)

    user_message = Message(conversation_id=conversation.id, role="user", content=question)
    session.add(user_message)
    session.flush()
    # strictly-later timestamp so ORDER BY created_at never ties the pair
    # (datetime.now() can return identical microseconds on Windows)
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=final.get("answer", ""),
        citations=citations,
        trace_id=trace.id,
        created_at=(user_message.created_at or datetime.now(timezone.utc)) + timedelta(milliseconds=1),
    )
    session.add(assistant_message)
    session.flush()

    context_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "document_filename": c["document_filename"],
            "score": c["score"],
            "source": c["source"],
            "page_number": c.get("page_number"),
            "section": c.get("section"),
            "text": c["text"],
        }
        for c in state.get("reranked_results", [])
    ]

    return {
        "answer": final.get("answer", ""),
        "citations": citations,
        "confidence": final.get("confidence", 0.0),
        "insufficient_evidence": final.get("insufficient_evidence", False),
        "conflicting_evidence": final.get("conflicting_evidence", False),
        "rewritten_question": final.get("rewritten_question"),
        "sub_queries": final.get("sub_queries", []),
        "retrieval_confidence": final.get("retrieval_confidence", 0.0),
        "context_chunks": context_chunks,
        "trace_id": str(trace.id),
        "conversation_id": str(conversation.id),
        "message_id": str(assistant_message.id),
        "stages": stages,
        "model": final.get("model", ""),
        "token_usage": final.get("token_usage", {}),
        "cost_usd": final.get("cost_usd"),
        "total_latency_ms": total_latency,
    }


def list_conversations(session: Session, tenant_id, user_id) -> list[Conversation]:
    return list(
        session.scalars(
            select(Conversation)
            .where(Conversation.tenant_id == tenant_id, Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
    )


def get_conversation(session: Session, tenant_id, user_id, conversation_id) -> Conversation | None:
    return session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == tenant_id,
            Conversation.user_id == user_id,
        )
    )
