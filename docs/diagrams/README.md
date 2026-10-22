# Diagrams

Interactive, self-contained HTML diagrams generated with
[Archify](https://github.com/tt-a1i/archify) from versioned JSON intermediate
representations (IR). Open the HTML files directly in any browser — no server
or network needed; they embed their own viewer (themes, pan/zoom, focus views,
PNG/SVG export).

| File | Contents |
|---|---|
| [rag-workflow.html](rag-workflow.html) | The LangGraph **agentic RAG query workflow**: lanes for Client / Agent / Hybrid Retrieval / Grounding & Safety / Observability / Fallbacks, the parallel dense ∥ lexical retrieval fan-out, RRF fusion, cross-encoder rerank, the evidence gate, bounded retry + citation-regeneration loops, refusal path, and trace persistence. Guided views: Happy path · No-evidence refusal · Weak-retrieval retry · Citation regeneration · Observability. |
| [rag-workflow.workflow.json](rag-workflow.workflow.json) | The Archify workflow-mode IR the HTML is generated from (source of truth — edit this, not the HTML). |
| [rag-workflow.delivery.json](rag-workflow.delivery.json) | Delivery receipt from the Archify pipeline (artifact SHA-256, checks). |

## Regenerating

The JSON IR is the source of truth. After editing it, validate and re-deliver
with the Archify CLI (Node.js ≥ 18; clone [tt-a1i/archify](https://github.com/tt-a1i/archify)
or run it via `npx skills use tt-a1i/archify@archify`):

```bash
cd <repo root>
node <path-to-archify>/archify/bin/archify.mjs validate workflow \
    docs/diagrams/rag-workflow.workflow.json --quality standard --json
node <path-to-archify>/archify/bin/archify.mjs deliver workflow \
    docs/diagrams/rag-workflow.workflow.json docs/diagrams/rag-workflow.html \
    --quality standard --json
```

Notes:

- The IR is validated against Archify's schema (workflow mode, schema v2) and
  rendered only after all artifact checks pass; `deliver` replaces the HTML
  atomically and emits the receipt.
- Quality is intentionally `standard`: the stricter `showcase` composition
  rules (zero edge crossings / merged corridors) cannot be met by this graph's
  truthful topology — three sources fan into the dense/lexical retrieval pair
  and two bounded feedback loops exist — without deleting semantic edges, which
  we won't do.
- Node `sources` (source-file links) were omitted: Archify pins those to a
  committed GitHub repository (`meta.repository`), which this project does not
  have yet. The code location of every node is documented in
  [../rag_pipeline.md](../rag_pipeline.md) instead.
