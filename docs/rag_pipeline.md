# The RAG Pipeline

> **Interactive version:** [diagrams/rag-workflow.html](diagrams/rag-workflow.html) — a
> self-contained Archify diagram of this exact workflow with guided views (happy path,
> refusal, retry, citation regeneration, observability). Regeneration notes:
> [diagrams/README.md](diagrams/README.md).

This document walks the exact path a question takes, with the guarantees each stage provides.

## 1. Query analysis (`query_analyzer`)

With a generative LLM: a JSON-mode prompt decides `needs_rewrite`, `rewritten_query`,
`sub_queries` (max 3), `answerable_from_documents`. With the offline extractive provider:
deterministic, unit-tested heuristics (pronoun/ellipsis follow-up detection, "what about …"
patterns, compound-question splitting, chit-chat detection).

**Guarantee:** clear queries are never rewritten — the rewrite path is conditional, not
mandatory.

## 2. Query rewriting (`query_rewriter`, conditional)

Applies the analyzer's decision: produces a self-contained `effective_query` (conversation
context folded in for follow-ups) and exposes any `sub_queries`. Rewrites are used *for
retrieval*; the original question is what the user sees and what generation answers.

## 3. Retrieval (parallel: `retrieve_dense` ∥ `retrieve_lexical`)

Each node runs over `[effective_query, *sub_queries]` and merges results across queries
(best score per chunk) — multi-query retrieval without extra graph complexity.

- **Dense:** query embedded by the same provider as ingestion → pgvector cosine ANN
  (`embedding <=> query`, HNSW index) — or exact in-Python cosine in array mode. Score =
  cosine similarity.
- **Lexical:** query OR-expanded (`ts_lexize` stems joined by `|`) → PostgreSQL FTS
  (`tsv @@ query`) ranked by `ts_rank_cd` cover-density, length-normalised (flag 32), scores
  normalised per-query to [0,1].

Both are strictly tenant-scoped by `tenant_id` in the WHERE clause. LangGraph runs the two
nodes on worker threads — each uses its own short-lived DB session (a SQLAlchemy session is
not thread-safe).

## 4. Fusion (`fuse_results`)

Reciprocal Rank Fusion over the two ranked lists:

```
rrf(d) = Σ_lists 1 / (rrf_k + rank_i(d))      # rrf_k = 60 (configurable)
```

Top `fusion_top_k` (default 10) candidates survive. Ranks-only fusion is immune to the
incomparable score scales of cosine and ts_rank_cd.

## 5. Rerank (`rerank`)

A local cross-encoder scores each `(query, chunk)` pair jointly (sigmoid → [0,1]); the top
`rerank_top_k` (default 5) become the final context. `retrieval_confidence` = best rerank
score.

## 6. Context validation (`context_validator`)

Refusal gate — the documented heuristic:

- **zero chunks** → insufficient
- **best score < `min_relevance_score`** (0.35) → insufficient
- **conflict hint**: two strong chunks from different documents sharing topic words with
  opposing modality markers ("may not"/"prohibited" vs "may"/"allowed") or disjoint numbers →
  flag for the generator to verify and report explicitly

Routing: sufficient → generate; insufficient **with weak-but-nonzero retrieval and retries
left** → one reformulation retry (LLM reformulation, or deterministic keyword variant
offline); otherwise → grounded refusal. Retrying on *zero* results is skipped — an
out-of-corpus question will not become answerable by rephrasing.

**Limitation (explicit):** thresholds are heuristics, not calibrated probabilities; the
conflict detector is deliberately conservative and can miss/report false conflicts. The
generator is instructed to verify flagged conflicts rather than trust the hint.

## 7. Grounded generation (`generate_answer`)

The prompt contains labelled context blocks:

```
[1] (chunk_id=…, document=leave_policy.md, section=Annual Leave Entitlement)
Full-time employees are entitled to 22 days of paid annual leave…

[QUESTION]
How many days of leave do employees get?
```

The system prompt enforces: only context-based claims, no outside knowledge, explicit
insufficiency, cite used blocks with a verbatim quote, never fabricate page/document names, report
conflicts. The response is a validated Pydantic model:

```json
{
  "answer": "…",
  "citations": [{"ref": 1, "quote": "exact words copied from block 1"}],
  "confidence": 0.0,
  "insufficient_evidence": false,
  "conflicting_evidence": false
}
```

Conversation history is included as *context for understanding follow-ups only* — never as a
source of facts; retrieval always runs again.

## 8. Citation validation (`citation_validator`)

The validator:

1. maps every `ref` to the actual retrieved chunk and **rebuilds** document name, page,
   section, chunk id, and excerpt from the chunk. A quote is kept only when it is an exact
   span of that chunk;
2. drops invalid refs and quotes that are missing or not verbatim; if any existed and the
   regeneration budget remains, one retry runs with feedback that names the bad refs and quotes;
3. still-invalid citations are simply absent from the final response.

## 9. Final response (`final_response`)

Assembles answer + citations + confidence (+ retrieval confidence, token usage, estimated
cost via configurable pricing, per-stage latencies). Insufficient evidence ⇒ canonical refusal
sentence, zero citations, zero confidence. Everything is persisted to `query_traces` and the
conversation.

## No-answer and conflict behaviour

- Refusal: *"I couldn't find sufficient evidence in the available documents to answer this
  question."* — asserted by tests for out-of-corpus questions.
- Conflicts: the generator must state that sources conflict and cite **every** conflicting
  block (the sample corpus contains a designed PTO conflict: handbook says 18 days, leave
  policy says 22).

## Observability

Every query writes one trace row: question, rewrite, sub-queries, per-stage latencies and
details, retrieved chunks (with scores) for dense/lexical/fused/reranked, final citations,
model, token usage, estimated cost, total latency, errors. The UI's trace view renders the
entire pipeline for any query.
