# Evaluation

Everything in this document was produced by **actual runs** of the evaluation harness
(`scripts/evaluate.py`, `scripts/benchmark.py`, `scripts/run_llm_eval.py`, or
`POST /api/evaluation/run`) against the included sample corpus. Three execution contexts
are recorded, each labelled where its numbers appear:

1. **Local, offline** — Windows, CPU-only inference, PostgreSQL 18 (array vector
   backend), `all-MiniLM-L6-v2` embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder
   reranker, deterministic **extractive** provider (generation metrics measure
   pipeline mechanics, not LLM quality).
2. **Local, real LLM** — same stack with `qwen2.5:3b` served by Ollama through the
   OpenAI-compatible endpoint: real LLM query analysis, grounded generation and
   LLM-as-judge evaluation (*Real-LLM generation run* below).
3. **Docker Compose, pgvector backend** — the full containerized stack
   (*Containerized run* below).

No number on this page is invented, and none is a general claim about the system on other
corpora — rerun the harness on your own data and the reports will show your numbers.

## Dataset

19 questions over 10 synthetic enterprise documents (~64 chunks), spanning eight question
types: direct lookup (5), numerical (3), multi-hop (3), comparison (2), policy reasoning (2),
negative/no-answer (3), ambiguous follow-up (1), cross-document (2). Each question lists the
expected source documents (retrieval labels) and expected answer keywords; negative questions
require a grounded refusal. Source: `data/evaluation/questions.json`.

## Retrieval metrics — definitions

Relevance unit is the **document** (retrieved identifiers are deduplicated in rank order, so
multiple chunks of one document count once — the metric suite has unit tests for this).

- **Recall@5** — of all relevant documents, the fraction appearing in the top-5 retrieved.
  Measures: does the right document make it into the context window at all?
- **Precision@5** — of the top-5 retrieved documents, the fraction that is relevant.
  Measures: how concentrated the context window is on the right sources (what the LLM sees).
- **MRR** (mean reciprocal rank) — average of 1/rank of the first relevant document.
  Measures: how quickly the single best source surfaces.
- **nDCG@5** — discounted cumulative gain with binary relevance, normalised by the ideal
  ranking. Measures: ranking quality across the whole top-5, rewarding relevant documents
  appearing earlier.

## Measured results (final run on the sample corpus)

| Strategy | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| Dense only | 0.790 | 0.179 | 0.763 | 0.770 |
| Lexical only | 0.790 | 0.200 | 0.763 | 0.770 |
| Hybrid (RRF) | 0.790 | 0.209 | 0.763 | 0.770 |
| **Hybrid + rerank** | **0.790** | **0.383** | **0.790** | **0.790** |

Reading the numbers honestly:

- On this small corpus the three pre-rerank strategies share recall (~0.79) — the four
  questions not covered include negative questions with no relevant documents by design, so
  ~0.79 is close to the practical ceiling for the answerable set.
- The cross-encoder reranker's contribution is visible exactly where theory predicts:
  **Precision@5 nearly doubles** (0.21 → 0.38) and MRR/nDCG tick up — the context window the
  LLM sees becomes concentrated on the right documents rather than merely containing them.
- Lexical alone matches dense recall here because the synthetic questions reuse corpus
  vocabulary; real-world paraphrased questions are where dense pulls ahead (and why hybrid is
  the default).

## Latency benchmarks (same run, p50 / p95 ms per query)

| Strategy | median | p95 |
|---|---|---|
| Dense only | 23.7 | 32.3 |
| Lexical only | 4.0 | 9.0 |
| Hybrid | 33.5 | 38.0 |
| Hybrid + rerank | 61.1 | 83.2 |

Per-stage (p50, from the same evaluation runs): lexical 3.1–4.0 ms · dense 29–39 ms (array
backend — exact in-Python cosine; the pgvector backend pushes ranking into the database) ·
fusion <0.1 ms · rerank ~24 ms steady-state (cross-encoder over 10 candidates on CPU).

Notes: the very first rerank in a process pays model loading (observed up to ~2.7 s); the
benchmark harness reports steady-state medians, and the evaluation run's p95 includes the
first call. First-ever ingestion also downloads models from HuggingFace.

## Generation metrics

Two clearly-labelled evaluation modes:

- **`llm_judge`** — when a generative provider is configured, an LLM judges faithfulness
  (every claim supported by context?) and relevancy (1–5, does the answer address the
  question?). LLM-as-judge is itself imperfect: judges are sensitive to phrasing, can be
  sycophantic, and their scores are not ground truth. Treat as estimates.
- **`heuristic`** (the recorded run) — deterministic measures used with the offline provider:
  faithfulness = fraction of answer sentences with ≥60% keyword coverage in the retrieved
  context; relevancy = query-keyword coverage mapped to 1–5. These measure **pipeline
  mechanics** (is the answer text actually derived from retrieved evidence?), not semantic
  quality.

The faithfulness judge receives the question, the retrieved context, and the
answer. A **canonical refusal** (the pipeline's exact no-evidence sentence, with
no extra claims) on a question marked `expected_answerable: false` is scored
faithful by rule (`mode: "rule"`, reason `expected_refusal`) and is not sent to
the judge. The same sentence on an answerable question is still judged.

Summary fields:

- `faithfulness_rate` / `avg_relevancy` — every question, including refusals.
- `faithfulness_rate_answerable` / `avg_relevancy_answerable` — questions with
  `expected_answerable: true` only. `answerable_count` is the denominator.

`scripts/run_llm_eval.py` writes the generation run, including each question's
context chunks, to `data/evaluation/last_generation_run.json`.

Citation correctness is always deterministic: every citation must reference an
actually-retrieved chunk, and negative questions must produce an explicit refusal.

The tables below were recorded **before** the refusal rule and before the judge
received the question. Do not compare a new run's `faithfulness_rate` with 0.368
or 0.684 directly. Use `faithfulness_rate_answerable` for the answered set.

Recorded run (`extractive` provider, `heuristic+rules`):

| Metric | Value |
|---|---|
| Faithfulness rate | 0.684 |
| Avg relevancy (1–5) | 2.58 |
| Citation correctness rate | 0.842 |
| End-to-end query latency | p50 120 ms / p95 298 ms |

## Real-LLM generation run (local qwen2.5:3b via Ollama, LLM-as-judge)

To verify the LLM path end-to-end without external API credits, the identical
pipeline was run with a real language model: `qwen2.5:3b` served locally by
Ollama through its OpenAI-compatible endpoint (`EKIP_OPENAI_BASE_URL`,
config snippet in `.env.example`). LLM query analysis, grounded JSON
generation, and LLM-as-judge faithfulness/relevancy all executed as real model
calls; token usage is reported from the model server.

| Metric | Value |
|---|---|
| Evaluator | `llm_judge+rules` |
| Faithfulness rate | 0.368 |
| Avg relevancy (1–5) | 2.95 |
| Citation correctness rate | 0.895 |
| Refusals on negative questions | 3/3 correct (all 4 out-of-corpus questions refused; 5/19 total refusals) |
| End-to-end query latency | p50 1.9 s / p95 5.9 s (CPU inference) |
| Token usage | real (e.g. 1,049 prompt + 131 completion on the smoke query) |

Reading these numbers honestly:

- **Faithfulness on this run is dragged down by refusals.** This run's judge
  call did not receive the question and did not special-case refusals, so a
  refusal with no claims was scored unfaithful. Five refusals cap that run's
  rate at 14/19 = 0.737; answered-only faithfulness is 7/14 = 0.50. Current
  code scores a canonical refusal on an unanswerable question as faithful and
  reports `faithfulness_rate_answerable` separately. Re-run to replace this table.
- **Citation correctness improved vs extractive mode** (0.895 vs 0.842) with a
  real model; the single failure is the dataset's deliberately-ambiguous
  pronoun question ("What about the previous one?"), which is labeled
  answerable but is correctly refused when asked without conversation history —
  an expectation mismatch in the dataset, not a pipeline error.
- Retrieval metrics reproduced **identically** across runs (Recall@5 0.790,
  Precision@5 0.383 for hybrid+rerank) — the retrieval pipeline is
  deterministic given the same corpus and models.
- Latency reflects CPU inference of a 3B model; a hosted API model changes
  latency (not methodology).

Reproduce with any OpenAI-compatible endpoint (see `.env.example`):
`python scripts/run_llm_eval.py`. The script prints every run summary and writes
the generation run to `data/evaluation/last_generation_run.json`.

## Generation run (qwen2.5:32b, LLM-as-judge)

This is the current generation result. The file
`data/evaluation/last_generation_run.json` is that run (`run_id`
`b78061fa-1009-47ac-9540-b0f679cf17e4`, finished 2026-10-22). Do not compare its
`faithfulness_rate` with the 0.368 row above. That 3B judge did not receive the
question and scored correct refusals as unfaithful.

Host: PostgreSQL 16 with pgvector, embeddings and the cross-encoder on CPU
(`sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`),
generation on `qwen2.5:32b` Q4_K_M through Ollama. The GPU was an NVIDIA B200 MIG
slice of about 20.5 GiB. The model was served with an 8k context and a q8 KV cache
so the weights fit.

| Metric | Value |
|---|---|
| Evaluator | `llm_judge+rules` |
| Faithfulness rate (19 questions) | 0.8421 |
| Faithfulness rate, answerable (16) | 0.8125 |
| Avg relevancy (1–5) | 4.1053 |
| Avg relevancy, answerable | 4.5 |
| Citation correctness rate | 0.8947 |
| Refusals on negative questions | 3/3 canonical refusals, scored faithful by rule |
| End-to-end query latency | p50 13.4 s / p95 23.2 s |

The same script printed a hybrid+rerank retrieval summary on this host of
Recall@5 0.7895 and Precision@5 0.3702. The table at the top of this page
(0.790 / 0.383) is the published comparison and was not replaced.

Three answerable questions were still weak, and the retrieved text contained
the expected keywords for each of them:

- Retirement-match question: the model refused. The context states the 401(k) match.
- 250 USD client dinner: the limit is right, then the answer adds one unsupported claim.
- Remote days plus home-office stipend: the main facts are right, then one extra claim
  is unsupported. The stored judge explanation is truncated before it names that claim.

"What about the previous one?" has no expected document and was asked with no
conversation history. The model refused. The judge marked that refusal faithful
and scored relevancy 1.

## Containerized run (Docker Compose, pgvector backend)

The full stack was built and executed via `docker compose up --build`
(postgres `pgvector/pgvector:pg16` + backend + frontend). Verified in the
containerized environment: pgvector extension **0.8.6** active, HNSW index
`ix_chunks_embedding_hnsw` created by the Alembic migration, dense retrieval
executing in-database (`EXPLAIN` shows `Index Scan using ix_chunks_embedding_hnsw`
with the `<=>` cosine operator), all 10 sample documents ingested through the
HTTP API, and warm query latency **122 ms total / dense 15 ms** — the
in-database ANN path is faster than the array compatibility backend used on
plain PostgreSQL. The UI served through nginx on port 8080 confirmed the same.

On the extractive-mode table above: its citation-correctness failures are
dominated by refusal-adjacent cases where the offline provider answered weakly
instead of refusing; they are real failures of that provider, reported as such.

## Reproducing

```bash
python scripts/ingest.py --tenant acme          # corpus into a tenant
python scripts/evaluate.py --tenant acme        # retrieval metrics, 4 strategies
python scripts/evaluate.py --tenant acme --generation
python scripts/benchmark.py --tenant acme --repeat 2
```

Or via the API (`POST /api/evaluation/run`) and the Evaluation dashboard. Every run is stored
(`evaluation_runs` table) with per-question detail: retrieved docs, per-metric values, and
per-stage latencies — inspectable in the UI.
