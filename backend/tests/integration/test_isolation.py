"""SECURITY tests: cross-tenant isolation.

These tests verify the platform's core security invariant: a tenant's query
can never retrieve another tenant's documents, neither via retrieval nor via
direct API/document access.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


@pytest.fixture()
def two_tenant_corpus(db_session, acme_user, globex_user):
    """Acme gets the leave policy; Globex gets a different document."""
    from app.ingestion.pipeline import run_ingestion

    acme_doc = run_ingestion(
        db_session, filename="leave_policy.md",
        payload=b"# Leave Policy\n\nEmployees receive 22 days of paid annual leave.\n",
        tenant_id=acme_user.tenant_id, user_id=acme_user.id,
    )
    globex_doc = run_ingestion(
        db_session, filename="globex_secrets.md",
        payload=b"# Globex Internal\n\nThe confidential launch code is ORION-7 for the Acme account.\n",
        tenant_id=globex_user.tenant_id, user_id=globex_user.id,
    )
    db_session.commit()
    assert acme_doc.status == "completed" and globex_doc.status == "completed"
    return acme_doc, globex_doc


@requires_db
class TestRetrievalIsolation:
    def test_acme_cannot_retrieve_globex_documents(self, db_session, acme_user, two_tenant_corpus):
        from app.retrieval.engine import HybridRetrievalEngine

        engine = HybridRetrievalEngine(db_session)
        # ask exactly for Globex content while authenticated as Acme
        for strategy in ("dense", "lexical", "hybrid", "hybrid_rerank"):
            result = engine.retrieve("launch code ORION-7 confidential", acme_user.tenant_id,
                                     strategy=strategy)
            filenames = {c.document_filename for c in result.final_chunks()}
            assert "globex_secrets.md" not in filenames, f"{strategy} leaked: {filenames}"

    def test_globex_cannot_retrieve_acme_documents(self, db_session, globex_user, two_tenant_corpus):
        from app.retrieval.engine import HybridRetrievalEngine

        engine = HybridRetrievalEngine(db_session)
        result = engine.retrieve("annual leave entitlement days", globex_user.tenant_id,
                                 strategy="hybrid_rerank")
        filenames = {c.document_filename for c in result.final_chunks()}
        assert "leave_policy.md" not in filenames

    def test_each_tenant_sees_own_documents(self, db_session, acme_user, globex_user, two_tenant_corpus):
        from app.retrieval.engine import HybridRetrievalEngine

        engine = HybridRetrievalEngine(db_session)
        acme = engine.retrieve("annual leave", acme_user.tenant_id, strategy="hybrid")
        globex = engine.retrieve("launch code", globex_user.tenant_id, strategy="hybrid")
        assert {c.document_filename for c in acme.fused} == {"leave_policy.md"}
        assert {c.document_filename for c in globex.fused} == {"globex_secrets.md"}


@requires_db
class TestServiceIsolation:
    def test_cross_tenant_document_access_is_404(self, db_session, acme_user, globex_user, two_tenant_corpus):
        from app.core.errors import NotFoundError
        from app.services.documents import get_document

        acme_doc, globex_doc = two_tenant_corpus
        with pytest.raises(NotFoundError):
            get_document(db_session, acme_user.tenant_id, globex_doc.document_id)
        with pytest.raises(NotFoundError):
            get_document(db_session, globex_user.tenant_id, acme_doc.document_id)

    def test_cross_tenant_delete_is_404(self, db_session, globex_user, two_tenant_corpus):
        from app.core.errors import NotFoundError
        from app.services.documents import delete_document

        acme_doc, _ = two_tenant_corpus
        with pytest.raises(NotFoundError):
            delete_document(db_session, globex_user.tenant_id, acme_doc.document_id)


@requires_db
class TestWorkflowIsolation:
    def test_query_never_answers_from_other_tenant(self, db_session, acme_user, two_tenant_corpus):
        from app.agents import RAGWorkflow

        state = RAGWorkflow(db_session).run(
            question="What is the confidential launch code for the Acme account?",
            tenant_id=acme_user.tenant_id,
        )
        final = state["final"]
        # the answer must NOT contain the Globex-only secret
        assert "ORION-7" not in final["answer"]
        retrieved = state.get("reranked_results", []) or state.get("fused_results", [])
        assert all(c["document_filename"] != "globex_secrets.md" for c in retrieved)
