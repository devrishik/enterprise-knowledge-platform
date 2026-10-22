# Design Decisions

This document explains *why* each significant choice was made and what was traded away —
the goal is that every line of the system is interview-defensible.

## Why PostgreSQL + pgvector (single database for everything)?

One database serves documents, chunks, **and both retrieval signals**:

- Dense vectors live in a native `vector` column with an **HNSW cosine index** — approximate
  nearest-neighbour search in the same transactional store.
- Lexical search uses PostgreSQL's built-in **full-text engine** (`tsvector` generated column +
  GIN index) — no separate Elasticsearch/OpenSearch cluster to deploy, sync, and monitor.
- Traces, conversations and evaluation results live alongside, so observability joins are
  trivial and there is exactly one backup/restore story.

Trade-offs: PostgreSQL FTS is not a modern BM25 engine (no IDF-exposed ranking, weaker
multilingual handling), and pgvector ANN is slower than a dedicated vector DB at very large
scale. For an enterprise knowledge base (10³–10⁶ chunks) both are well within their comfort
zone, and operating one datastore instead of three is the right trade.

### Vector backend: `pgvector` vs `array` (compatibility mode)

The production path stores embeddings in a pgvector column and searches in-database
(`ORDER BY embedding <=> query`). The `array` backend (env `EKIP_VECTOR_BACKEND=array`) stores
the identical vectors as `double precision[]` and computes exact cosine similarity in Python.
It exists so the full pipeline can run and be tested on PostgreSQL installations where the
pgvector extension cannot be installed (e.g. plain PostgreSQL on Windows without admin access
or a compiler). Both implement the same retriever interface; the ORM model *and* the Alembic
migration derive the column type from one function (`app/db/types.embedding_sql_type`), so
schema and runtime can never diverge. The compatibility mode is exact-but-O(n) — fine at
portfolio scale, not a production substitute for ANN.

## Why hybrid retrieval (dense + lexical)?

They fail differently:

- **Dense** (MiniLM bi-encoder embeddings) understands paraphrase ("time off" ≈ "annual leave")
  but misses rare exact tokens — policy numbers, product codes, names — that never appear in
  training contexts.
- **Lexical** (FTS) matches exact terms with high precision but is blind to synonyms and
  acronyms.

Measured on this corpus (see `docs/evaluation.md`), hybrid recall ≥ either single signal on the
eval set, and the combination never does worse — consistent with the published hybrid-retrieval
literature.

## Why BM25-style lexical via `ts_rank_cd` with OR-expansion?

PostgreSQL core has no exact BM25. The implementation:

1. **OR-expands** the query: each content word is stemmed (`ts_lexize('english_stem', …)`) and
   joined with `|`. Plain `websearch_to_tsquery` uses strict AND semantics — one unmatchable
   word ("How *many* days are employees entitled to…") returns **zero rows**, which we observed
   in practice during development. OR-expansion keeps recall sane.
2. Ranks with **`ts_rank_cd` cover-density**, normalised by document length (flag 32) —
   TF-style, length-normalised ranking in the spirit of BM25 but **without IDF** weighting.

Why this is acceptable here: RRF fusion (below) consumes **ranks, not raw scores**, so the
lexical ranker's score calibration is largely irrelevant to the final ordering, and the
cross-encoder re-scores the fused candidates jointly with the query. Exact BM25 (IDF) would be
the first upgrade — via ParadeDB/pg_search or an external engine.

## Why Reciprocal Rank Fusion?

RRF (`score(d) = Σ 1/(k + rank_i(d))`, k=60) combines ranked lists without any assumption about
score scales. Cosine similarities (0.3–0.9) and `ts_rank_cd` values (~0.01–0.2) are utterly
incomparable; normalising them into a weighted sum is fragile and requires per-signal tuning.
RRF needs one robust constant and provably works well (Cormack et al., SIGIR 2009). Trade-off:
it discards magnitude information (a near-perfect dense hit is treated the same as a marginal
one) — which is exactly why a cross-encoder follows.

## Why cross-encoder reranking?

Bi-encoder retrieval scores are a cheap approximation: query and passage are embedded
independently, so fine-grained interaction is impossible. A cross-encoder encodes the
(query, passage) **pair jointly** with full attention — substantially better precision at the
top of the ranking, at the cost of a model forward pass per candidate (hence only
`fusion_top_k=10` candidates are reranked, bounded by configuration).

Measured effect on the eval corpus: **Precision@5 roughly doubles vs hybrid without reranking**
(0.21 → 0.38) — see `docs/evaluation.md` for the run that produced those numbers.

## Why LangGraph for the agent workflow?

The workflow is genuinely a **graph**: parallel branches (dense ∥ lexical), joins (fusion),
conditional edges (skip rewrite / retry weak retrieval / refuse on insufficient evidence /
bounded citation regeneration), and cycles (retry loop). LangGraph gives exactly this with a
typed state contract (`TypedDict` with an additive `stages` reducer for the parallel branches)
and per-node observability for free. A linear "chain" would have hidden these decisions in
nested ifs.

Design guardrails: the graph stays **practical** — bounded retries (`max_query_rewrites=1`),
bounded regeneration (1), no autonomous tool-calling loops. An unbounded agent is a liability
in an enterprise Q&A system, not a feature.

## Why structure-aware chunking?

Arbitrary fixed-size splits cut semantic units mid-thought and destroy heading context, which
hurts both embedding quality and citation precision. The chunker groups parsed blocks under
their heading section, prepends the heading as context, merges small blocks up to
`chunk_size`, splits oversized blocks at **sentence boundaries** with overlap, and **never
merges across PDF pages** so citations can name an exact page.

Defaults (1200 chars ≈ 300 tokens, 200 overlap) fit the MiniLM context window (256 word
pieces) with headroom for the heading prefix. A naive `fixed` strategy remains selectable for
A/B comparison (chunker is config, `EKIP_CHUNKING_STRATEGY`).

## Why is the relevance gate a heuristic threshold?

`min_relevance_score` (default 0.35) compares normalised retrieval scores against a threshold
to decide "sufficient evidence vs refuse". It is a **documented heuristic**, not a calibrated
probability: cross-encoder sigmoids and normalised RRF scores are only loosely comparable
across queries. It errs toward refusal (conservative), which is the right failure mode for
grounded enterprise Q&A. Calibration against labelled data would be the principled upgrade.

## Why no-answer handling?

A RAG system that always answers is a hallucination machine. When retrieval is empty or below
the gate, the workflow short-circuits to an explicit refusal ("couldn't find sufficient
evidence…") with zero citations and zero confidence. Negative questions in the evaluation
dataset assert this behaviour mechanically (refusal must occur, citation correctness requires
it).

## Why local (Sentence Transformers) embeddings?

- **No per-query cost or network dependency** for the retrieval half of the system; the
  platform works fully offline except the LLM.
- **Stable vectors**: swapping to a hosted embedder silently re-indexes nothing here — vectors
  are tied to one model, versioned in document metadata, and the dimension is validated at
  load (`EKIP_EMBEDDING_DIM` mismatch is a startup error, not a runtime surprise).

Trade-off: MiniLM-L6-v2 (384-dim) is weaker than large hosted embedders on hard semantic
matching. The provider abstraction (`app/embeddings/`) makes swapping a configuration change
plus re-ingestion.

## Why the "extractive" offline LLM provider?

Interviewers (and teammates) should be able to run the demo with **zero API keys**. The
extractive provider parses the *same grounded prompt* a real LLM receives, extracts the
best-covered context sentences, and returns the same JSON contract with citations. It is
explicitly labelled `extractive` everywhere (traces, health, evaluation reports) — it is **not
an LLM** and makes no quality claims; generation-quality metrics produced under it measure
pipeline mechanics. With `EKIP_LLM_PROVIDER=openai` the identical workflow runs GPT-class
generation, analysis, rewriting and LLM-as-judge evaluation.

## Why server-constructed citations?

Every displayed citation field — document name, page, section, chunk id, excerpt — is rebuilt
from the actual retrieved chunk by the citation validator. A quote is kept only when it is an
exact span of that chunk. Fabricated page numbers, document names, or quotes are dropped;
invalid citations trigger one bounded regeneration.

## Why RLS-free tenant isolation?

Isolation is enforced by a denormalised `tenant_id` on every row plus mandatory tenant filters
in the single retrieval/service layer (and in the raw SQL of the lexical retriever). Security
tests assert cross-tenant retrieval is impossible for every strategy and via direct ID access.
Row-Level Security would be the belt-and-braces production addition (documented as a future
improvement) — for this scope, tested query-level isolation with a small trusted surface is
sufficient and much easier to reason about.

## Why lightweight API-key auth?

Portfolio scope: the data model (users/tenants) is production-shaped; only the credential
mechanism is lightweight (opaque demo keys in `X-API-Key`). The swap path to OIDC/JWT touches
one dependency (`app/api/deps.py`) and no business logic. Documented as a security limitation,
not hidden.

## Performance notes

- Embeddings and reranker load **once per process** (cached singletons; a reranker-reload-per-
  query bug was caught by latency tracing during development and fixed).
- Batched embedding generation at ingestion.
- Deterministic tie-breaking everywhere (score, then chunk id) — flaky rankings are worse than
  marginally slower ones; found via test-suite flakes.
- Parallel retrieval nodes use separate DB sessions (a SQLAlchemy session is not thread-safe;
  LangGraph runs parallel nodes on worker threads).
- Retrieval Ks, RRF constant and rerank candidate counts are configuration, not constants.
