"""Integration tests: REST API surface (FastAPI TestClient)."""

from __future__ import annotations

import pytest

from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

ACME_KEY = "demo-acme-alice-key"
GLOBEX_KEY = "demo-globex-gina-key"
ACME_HEADERS = {"X-API-Key": ACME_KEY}
GLOBEX_HEADERS = {"X-API-Key": GLOBEX_KEY}


@requires_db
class TestAuth:
    def test_missing_api_key_is_401(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_invalid_api_key_is_401(self, client):
        response = client.get("/api/documents", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_demo_users_endpoint_lists_seed_users(self, client):
        response = client.get("/api/auth/demo-users")
        assert response.status_code == 200
        users = response.json()
        assert any(u["api_key"] == ACME_KEY for u in users)
        assert any(u["api_key"] == GLOBEX_KEY for u in users)


@requires_db
class TestDocumentEndpoints:
    def test_upload_list_get_delete_flow(self, client):
        payload = (
            b"# Expense Policy\n\nThe daily meal per diem is 60 USD for domestic travel.\n"
        )
        upload = client.post(
            "/api/documents/upload",
            headers=ACME_HEADERS,
            files={"file": ("expense.md", payload, "text/markdown")},
        )
        assert upload.status_code == 201, upload.text
        body = upload.json()
        assert body["status"] == "completed"
        assert body["chunk_count"] >= 1
        document_id = body["document_id"]

        listing = client.get("/api/documents", headers=ACME_HEADERS)
        assert listing.status_code == 200
        assert any(d["id"] == document_id for d in listing.json())

        detail = client.get(f"/api/documents/{document_id}", headers=ACME_HEADERS)
        assert detail.status_code == 200
        assert detail.json()["doc_type"] == "md"

        chunks = client.get(f"/api/documents/{document_id}/chunks", headers=ACME_HEADERS)
        assert chunks.status_code == 200
        assert len(chunks.json()) == body["chunk_count"]

        deleted = client.delete(f"/api/documents/{document_id}", headers=ACME_HEADERS)
        assert deleted.status_code == 204
        assert client.get(f"/api/documents/{document_id}", headers=ACME_HEADERS).status_code == 404

    def test_invalid_document_id_is_404(self, client):
        response = client.get("/api/documents/00000000-0000-0000-0000-000000000000",
                              headers=ACME_HEADERS)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_malformed_id_returns_safe_error(self, client):
        response = client.get("/api/documents/not-a-uuid", headers=ACME_HEADERS)
        assert response.status_code in (404, 422)
        assert "Traceback" not in response.text

    def test_unsupported_file_type_is_415(self, client):
        response = client.post(
            "/api/documents/upload",
            headers=ACME_HEADERS,
            files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_file_type"

    def test_oversized_upload_is_413(self, client):
        from app.core.config import get_settings

        big = b"x" * (get_settings().max_upload_mb * 1024 * 1024 + 1024)
        response = client.post(
            "/api/documents/upload",
            headers=ACME_HEADERS,
            files={"file": ("big.txt", big, "text/plain")},
        )
        assert response.status_code == 413

    def test_malformed_pdf_returns_failed_result(self, client):
        response = client.post(
            "/api/documents/upload",
            headers=ACME_HEADERS,
            files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
        )
        assert response.status_code == 201  # upload accepted; ingestion failed and is recorded
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"]

    def test_cross_tenant_document_access_is_404(self, client):
        upload = client.post(
            "/api/documents/upload",
            headers=ACME_HEADERS,
            files={"file": ("acme_private.md", b"# Acme Private\n\nInternal acme data only.", "text/markdown")},
        )
        document_id = upload.json()["document_id"]
        assert client.get(f"/api/documents/{document_id}", headers=GLOBEX_HEADERS).status_code == 404
        assert client.delete(f"/api/documents/{document_id}", headers=GLOBEX_HEADERS).status_code == 404

    def test_cross_tenant_listing_is_isolated(self, client):
        client.post(
            "/api/documents/upload", headers=ACME_HEADERS,
            files={"file": ("acme_only.md", b"# Acme\n\nacme content", "text/markdown")},
        )
        listing = client.get("/api/documents", headers=GLOBEX_HEADERS).json()
        assert all(d["filename"] != "acme_only.md" for d in listing)


@requires_db
class TestQueryEndpoint:
    def test_query_returns_grounded_answer_and_citations(self, client, ingested_corpus):
        response = client.post(
            "/api/query",
            headers=ACME_HEADERS,
            json={"question": "How many days of paid annual leave do employees get?"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "22" in body["answer"]
        assert body["citations"]
        assert body["citations"][0]["document_name"] == "leave_policy.md"
        assert body["trace_id"]
        assert body["conversation_id"]
        assert body["total_latency_ms"] > 0
        assert body["context_chunks"]

    def test_query_refuses_out_of_corpus_questions(self, client, ingested_corpus):
        response = client.post(
            "/api/query",
            headers=ACME_HEADERS,
            json={"question": "What is the CEO's favorite color?"},
        )
        body = response.json()
        assert body["insufficient_evidence"] is True
        assert "couldn't find sufficient evidence" in body["answer"]

    def test_conversation_followup_flow(self, client, ingested_corpus):
        first = client.post(
            "/api/query", headers=ACME_HEADERS,
            json={"question": "What is the parental leave for primary caregivers?"},
        ).json()
        conversation_id = first["conversation_id"]
        second = client.post(
            "/api/query", headers=ACME_HEADERS,
            json={
                "question": "What about the previous policy for secondary caregivers?",
                "conversation_id": conversation_id,
            },
        ).json()
        assert second["conversation_id"] == conversation_id

        detail = client.get(f"/api/conversations/{conversation_id}", headers=ACME_HEADERS)
        assert detail.status_code == 200
        messages = detail.json()["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_conversations_listing(self, client, ingested_corpus):
        client.post("/api/query", headers=ACME_HEADERS,
                    json={"question": "What is the home office stipend?"})
        listing = client.get("/api/conversations", headers=ACME_HEADERS)
        assert listing.status_code == 200
        assert len(listing.json()) >= 1

    def test_cross_tenant_conversation_access_is_404(self, client, ingested_corpus):
        created = client.post(
            "/api/query", headers=ACME_HEADERS,
            json={"question": "What is the sick leave entitlement?"},
        ).json()
        response = client.get(f"/api/conversations/{created['conversation_id']}",
                              headers=GLOBEX_HEADERS)
        assert response.status_code == 404


@requires_db
class TestTraces:
    def test_trace_list_and_detail(self, client, ingested_corpus):
        query = client.post(
            "/api/query", headers=ACME_HEADERS,
            json={"question": "What is the meal per diem for domestic travel?"},
        ).json()
        trace_id = query["trace_id"]

        listing = client.get("/api/traces", headers=ACME_HEADERS)
        assert listing.status_code == 200
        assert any(t["id"] == trace_id for t in listing.json())

        detail = client.get(f"/api/traces/{trace_id}", headers=ACME_HEADERS)
        assert detail.status_code == 200
        trace = detail.json()
        stage_names = {s["stage"] for s in trace["stages"]}
        assert {"query_analyzer", "retrieve_dense", "retrieve_lexical", "fuse_results",
                "rerank", "generate_answer"} <= stage_names
        assert trace["retrieved"]["dense"] and trace["retrieved"]["lexical"]
        assert trace["model"]

    def test_cross_tenant_trace_is_404(self, client, ingested_corpus):
        query = client.post(
            "/api/query", headers=ACME_HEADERS,
            json={"question": "How often must production passwords rotate?"},
        ).json()
        response = client.get(f"/api/traces/{query['trace_id']}", headers=GLOBEX_HEADERS)
        assert response.status_code == 404


@requires_db
class TestEvaluationEndpoints:
    def test_retrieval_evaluation_run(self, client, ingested_corpus):
        response = client.post(
            "/api/evaluation/run",
            headers=ACME_HEADERS,
            json={"strategies": ["dense", "hybrid"], "include_generation": False},
        )
        assert response.status_code == 202, response.text
        run_ids = response.json()["run_ids"]
        assert len(run_ids) == 2

        runs = client.get("/api/evaluation/runs", headers=ACME_HEADERS).json()
        assert len(runs) >= 2
        by_strategy = {r["strategy"]: r for r in runs}
        assert "dense" in by_strategy and "hybrid" in by_strategy

        run_id = by_strategy["hybrid"]["id"]
        detail = client.get(f"/api/evaluation/runs/{run_id}", headers=ACME_HEADERS).json()
        summary = detail["summary"]
        for key in ("recall@5", "precision@5", "mrr", "ndcg@5"):
            assert key in summary, f"missing metric {key}"
            assert 0.0 <= summary[key] <= 1.0
        assert "benchmarks" in summary
        assert len(detail["per_question"]) == detail["question_count"]

    def test_generation_evaluation_run(self, client, ingested_corpus):
        response = client.post(
            "/api/evaluation/run",
            headers=ACME_HEADERS,
            json={"strategies": ["hybrid_rerank"], "include_generation": True},
        )
        assert response.status_code == 202
        runs = client.get("/api/evaluation/runs", headers=ACME_HEADERS).json()
        generation = next(r for r in runs if r["strategy"] == "generation")
        detail = client.get(f"/api/evaluation/runs/{generation['id']}", headers=ACME_HEADERS).json()
        summary = detail["summary"]
        assert "faithfulness_rate" in summary
        assert "faithfulness_rate_answerable" in summary
        assert "avg_relevancy_answerable" in summary
        assert "citation_correctness_rate" in summary
        generation_row = next(q for q in detail["per_question"] if q["question_type"] == "negative")
        assert "context" in generation_row
        assert generation_row["expected_answerable"] is False
        assert summary["evaluator"] == "heuristic+rules"  # extractive provider in tests


@requires_db
class TestHealthAndDashboard:
    def test_health_reports_configuration(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["database"] == "up"
        assert body["vector_backend"] in ("pgvector", "array")
        assert body["llm_provider"]

    def test_metrics_summary(self, client, ingested_corpus):
        response = client.get("/api/metrics/summary", headers=ACME_HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["documents"] >= 10
        assert body["chunks"] > 30
