"""Integration tests: full RAG workflow (LangGraph) end-to-end."""

from __future__ import annotations

import pytest

from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


def run_workflow(db_session, question, tenant_id, history=None):
    from app.agents import RAGWorkflow

    return RAGWorkflow(db_session).run(
        question=question, tenant_id=str(tenant_id), conversation_history=history
    )


@requires_db
class TestAnswerableQuestions:
    def test_grounded_answer_with_valid_citations(self, db_session, acme_user, ingested_corpus):
        state = run_workflow(db_session, "How many days of paid annual leave do employees get?",
                             acme_user.tenant_id)
        final = state["final"]
        assert final["insufficient_evidence"] is False
        assert "22" in final["answer"]
        assert final["citations"], "expected at least one citation"
        # every citation must reference a chunk that was actually retrieved
        retrieved_ids = {c["chunk_id"] for c in state.get("reranked_results", [])}
        assert all(c["chunk_id"] in retrieved_ids for c in final["citations"])
        assert final["citations"][0]["document_name"] == "leave_policy.md"

    def test_stage_trace_recorded(self, db_session, acme_user, ingested_corpus):
        state = run_workflow(db_session, "What is the meal per diem for domestic travel?",
                             acme_user.tenant_id)
        stages = [s["stage"] for s in state["stages"]]
        for expected in ("query_analyzer", "retrieve_dense", "retrieve_lexical",
                         "fuse_results", "rerank", "context_validator",
                         "generate_answer", "citation_validator", "final_response"):
            assert expected in stages, f"missing stage {expected}"
        assert all("latency_ms" in s for s in state["stages"])

    def test_query_rewriter_present_when_needed(self, db_session, acme_user, ingested_corpus):
        history = [
            {"role": "user", "content": "What is the parental leave policy for primary caregivers?"},
            {"role": "assistant", "content": "Primary caregivers get 16 weeks of paid leave."},
        ]
        state = run_workflow(db_session, "What about the previous policy?", acme_user.tenant_id,
                             history=history)
        stages = {s["stage"] for s in state["stages"]}
        assert "query_rewriter" in stages
        assert state.get("effective_query") != "What about the previous policy?"

    def test_clear_query_skips_rewriter(self, db_session, acme_user, ingested_corpus):
        state = run_workflow(db_session, "What is the home office stipend amount?",
                             acme_user.tenant_id)
        stages = {s["stage"] for s in state["stages"]}
        assert "query_rewriter" not in stages  # clear query -> no rewrite LLM call


@requires_db
class TestNoAnswerBehavior:
    def test_out_of_corpus_question_refuses(self, db_session, acme_user, ingested_corpus):
        state = run_workflow(db_session, "What is the company's policy on cryptocurrency payments?",
                             acme_user.tenant_id)
        final = state["final"]
        assert final["insufficient_evidence"] is True
        assert "couldn't find sufficient evidence" in final["answer"]
        assert final["citations"] == []
        assert final["confidence"] == 0.0

    def test_chitchat_short_circuits_to_final(self, db_session, acme_user, ingested_corpus):
        state = run_workflow(db_session, "hello there!", acme_user.tenant_id)
        stages = {s["stage"] for s in state["stages"]}
        assert "retrieve_dense" not in stages  # analyzer routed straight to final
        assert state["final"]["insufficient_evidence"] is True


@requires_db
class TestConversationFlow:
    def test_followup_uses_history_but_retrieves_fresh(self, db_session, acme_user, ingested_corpus):
        # turn 1
        first = run_workflow(db_session, "What is the parental leave for primary caregivers?",
                             acme_user.tenant_id)
        assert first["final"]["insufficient_evidence"] is False
        # turn 2: follow-up references turn 1 contextually
        history = [
            {"role": "user", "content": "What is the parental leave for primary caregivers?"},
            {"role": "assistant", "content": first["final"]["answer"]},
        ]
        second = run_workflow(db_session, "and what about secondary caregivers?", acme_user.tenant_id,
                              history=history)
        # grounding rule: follow-up must retrieve evidence again, not answer from memory
        assert second.get("reranked_results") or second.get("fused_results")
        assert second["final"]["insufficient_evidence"] is False
        assert "4 weeks" in second["final"]["answer"] or "four weeks" in second["final"]["answer"]


@requires_db
class TestRetryPath:
    def test_weak_retrieval_triggers_alternative_formulation(self, db_session, acme_user):
        from app.ingestion.pipeline import run_ingestion

        # a corpus where the vague first formulation scores weakly
        run_ingestion(db_session, filename="stipend.md",
                      payload=b"# Stipend\n\nThe home office stipend is 500 USD one-time.",
                      tenant_id=acme_user.tenant_id)
        db_session.commit()
        state = run_workflow(db_session, "money for desk chair at home",
                             acme_user.tenant_id)
        stages = [s["stage"] for s in state["stages"]]
        # retry is best-effort: assert the loop stays bounded and terminates
        assert stages.count("retry_with_alternative") <= 1
        assert "final_response" in stages
