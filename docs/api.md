# API Reference

Interactive OpenAPI documentation is served at **`/docs`** (Swagger UI) when the backend runs.
All endpoints are prefixed with `/api`. Errors are always safe JSON — no stack traces:

```json
{ "error": { "code": "not_found", "message": "Document not found" } }
```

Common error codes: `unauthorized` (401, missing/invalid key) · `not_found` (404) ·
`file_too_large` (413) · `unsupported_file_type` (415) · `validation_error` (422) ·
`provider_unavailable` (503, e.g. missing API key) · `internal_error` (500).

## Authentication

Lightweight demo auth: send the `X-API-Key` header. Seeded identities are listed by
`GET /api/auth/demo-users` (e.g. `demo-acme-alice-key`, `demo-globex-gina-key`). Every
subsequent call is tenant-scoped server-side.

## Health

`GET /api/health` → database status, vector backend, LLM/embedding/reranker providers and
whether the LLM is configured (with an actionable message when the API key is missing).

## Documents

| Endpoint | Notes |
|---|---|
| `POST /api/documents/upload` | multipart file. Returns 201 with `{document_id, status, chunk_count, duration_ms, timing_ms, error?}`. Ingestion failures (corrupt/empty file) return **201 with `status: "failed"`** and the recorded error — the document row exists for auditing. Client errors (unsupported type / too large) are 415/413 and create nothing. |
| `GET /api/documents` | tenant's library, newest first. |
| `GET /api/documents/{id}` | details incl. ingestion metadata. |
| `GET /api/documents/{id}/chunks` | chunks with page/section/token metadata. |
| `DELETE /api/documents/{id}` | 204; cascades chunks/jobs. Cross-tenant access → 404. |

## Query

`POST /api/query`

```json
{ "question": "How many days of paid annual leave?", "conversation_id": null }
```

Response (abridged):

```json
{
  "answer": "…22 days…",
  "citations": [
    { "ref": 1, "chunk_id": "…", "document_id": "…", "document_name": "leave_policy.md",
      "page": null, "section": "Annual Leave Entitlement", "excerpt": "…" }
  ],
  "confidence": 0.54,
  "insufficient_evidence": false,
  "conflicting_evidence": false,
  "rewritten_question": null,
  "sub_queries": [],
  "retrieval_confidence": 0.99,
  "context_chunks": [ { "document_filename": "leave_policy.md", "score": 0.99, "text": "…" } ],
  "stages": [ { "stage": "retrieve_dense", "latency_ms": 38.6, "details": {…} } ],
  "model": "extractive",
  "token_usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
  "cost_usd": null,
  "trace_id": "…", "conversation_id": "…", "message_id": "…",
  "total_latency_ms": 210.4
}
```

Sending `conversation_id` from a previous response enables follow-up questions — the workflow
rewrites pronoun-style follow-ups into self-contained queries and **always retrieves fresh
evidence**.

## Conversations

`GET /api/conversations` · `GET /api/conversations/{id}` (with messages + citations +
trace ids). Tenant- and user-scoped.

## Traces

`GET /api/traces?limit=50` — recent traces (question, latency, outcome).
`GET /api/traces/{id}` — the full observability record: rewritten query, sub-queries,
per-stage latencies+details, retrieved chunks **per stage** (dense/lexical/fused/reranked with
scores), final citations, answer, confidence, model, token usage, cost, total latency, error.

## Evaluation

| Endpoint | Notes |
|---|---|
| `POST /api/evaluation/run` | `{ "strategies": ["dense","lexical","hybrid","hybrid_rerank"], "include_generation": false, "max_questions": null }` → 202 with run ids. Runs synchronously (portfolio scale). |
| `GET /api/evaluation/runs` | run list with summaries. |
| `GET /api/evaluation/runs/{id}` | full run: summary metrics + benchmarks + per-question detail. |

## Dashboard

`GET /api/metrics/summary` — document/chunk/query counts, ingestion job outcomes, average
query latency (tenant-scoped).
