import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api/client'
import type { RetrievedChunk, TraceDetail } from '../types'

export default function TraceDetail() {
  const { id } = useParams<{ id: string }>()
  const [trace, setTrace] = useState<TraceDetail | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!id) return
    api.getTrace<TraceDetail>(id).then(setTrace).catch((e) => setError(e.message))
  }, [id])

  if (error) return <div className="error-banner">{error}</div>
  if (!trace) return <div className="empty">Loading…</div>

  return (
    <div>
      <div className="card">
        <h2>
          Trace <span className="mono small">{trace.id.slice(0, 8)}</span>
          <Link to="/traces" className="small" style={{ float: 'right' }}>← all traces</Link>
        </h2>
        <table className="data">
          <tbody>
            <tr><td className="muted" style={{ width: 200 }}>Question</td><td>{trace.question}</td></tr>
            {trace.rewritten_question && <tr><td className="muted">Rewritten query</td><td className="mono">{trace.rewritten_question}</td></tr>}
            {trace.sub_queries.length > 0 && <tr><td className="muted">Sub-queries</td><td>{trace.sub_queries.join(' · ')}</td></tr>}
            <tr><td className="muted">Answer</td><td>{trace.answer ?? '–'}</td></tr>
            <tr><td className="muted">Outcome</td><td>{trace.insufficient_evidence ? <span className="tag amber">insufficient evidence</span> : <span className="tag green">answered</span>}</td></tr>
            <tr><td className="muted">Confidence</td><td>{trace.confidence ?? '–'}</td></tr>
            <tr><td className="muted">Model</td><td className="mono">{trace.model ?? '–'}</td></tr>
            <tr><td className="muted">Total latency</td><td>{trace.total_latency_ms != null ? `${Math.round(trace.total_latency_ms)} ms` : '–'}</td></tr>
            <tr><td className="muted">Tokens</td><td>{trace.token_usage?.total_tokens ?? 0} total ({trace.token_usage?.prompt_tokens ?? 0} prompt + {trace.token_usage?.completion_tokens ?? 0} completion)</td></tr>
            <tr><td className="muted">Estimated cost</td><td>{trace.cost_usd != null ? `$${trace.cost_usd.toFixed(6)}` : 'unavailable (no pricing for model)'}</td></tr>
            {trace.error && <tr><td className="muted">Error</td><td style={{ color: 'var(--danger)' }}>{trace.error}</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Pipeline stages</h2>
        <table className="data">
          <thead><tr><th>Stage</th><th>Latency</th><th>Details</th></tr></thead>
          <tbody>
            {trace.stages.map((s, i) => (
              <tr key={i}>
                <td><strong>{s.stage.replace(/_/g, ' ')}</strong></td>
                <td>{s.latency_ms.toFixed(2)} ms</td>
                <td className="mono small" style={{ maxWidth: 640, wordBreak: 'break-all' }}>
                  {JSON.stringify(s.details)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(['dense', 'lexical', 'fused', 'reranked'] as const).map((stage) => {
        const chunks = (trace.retrieved?.[stage] ?? []) as RetrievedChunk[]
        if (chunks.length === 0) return null
        return (
          <div className="card" key={stage}>
            <h2>{stage} retrieval — {chunks.length} chunks</h2>
            <div className="chunk-list">
              {chunks.slice(0, 10).map((c) => (
                <div key={c.chunk_id} className="chunk-card">
                  <div className="chunk-head">
                    <span>{c.document_filename}{c.page_number != null ? ` · p.${c.page_number}` : ''}{c.section ? ` · ${c.section}` : ''}</span>
                    <span className="tag blue">{c.score.toFixed(3)}</span>
                  </div>
                  <div>{c.text.slice(0, 260)}…</div>
                </div>
              ))}
            </div>
          </div>
        )
      })}

      {trace.final_citations.length > 0 && (
        <div className="card">
          <h2>Final citations</h2>
          {trace.final_citations.map((c: any, i: number) => (
            <div key={i} className="citation">
              <div className="cite-head">[{c.ref}] {c.document_name}{c.page != null ? `, page ${c.page}` : ''}{c.section ? `, “${c.section}”` : ''}</div>
              <div className="excerpt mono small">chunk {c.chunk_id}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
