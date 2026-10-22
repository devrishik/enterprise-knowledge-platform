import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { HealthStatus, MetricsSummary, TraceListItem } from '../types'

export default function Dashboard() {
  const [summary, setSummary] = useState<MetricsSummary | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [traces, setTraces] = useState<TraceListItem[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    api.metricsSummary<MetricsSummary>().then(setSummary).catch((e) => setError(e.message))
    api.health<HealthStatus>().then(setHealth).catch(() => {})
    api.listTraces<TraceListItem[]>().then(setTraces).catch(() => {})
  }, [])

  return (
    <div>
      {error && <div className="error-banner">{error}</div>}
      <div className="grid cols-4">
        <div className="card"><div className="stat-value">{summary?.documents ?? '–'}</div><div className="stat-label">Documents</div></div>
        <div className="card"><div className="stat-value">{summary?.chunks ?? '–'}</div><div className="stat-label">Indexed chunks</div></div>
        <div className="card"><div className="stat-value">{summary?.queries ?? '–'}</div><div className="stat-label">Queries answered</div></div>
        <div className="card">
          <div className="stat-value">{summary?.avg_query_latency_ms ? `${summary.avg_query_latency_ms} ms` : '–'}</div>
          <div className="stat-label">Avg query latency</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h2>System configuration</h2>
          {health ? (
            <table className="data">
              <tbody>
                <tr><td className="muted">Database</td><td><span className={health.database === 'up' ? 'tag green' : 'tag red'}>{health.database}</span></td></tr>
                <tr><td className="muted">Vector backend</td><td className="mono">{health.vector_backend}</td></tr>
                <tr><td className="muted">LLM provider</td><td className="mono">{health.llm_provider} {health.llm_configured ? '' : '(not configured)'}</td></tr>
                <tr><td className="muted">Embeddings</td><td className="mono small">{health.embedding_provider}</td></tr>
                <tr><td className="muted">Reranker</td><td className="mono small">{health.reranker_provider}</td></tr>
                <tr><td className="muted">Ingestion jobs</td><td>{summary?.ingestion_jobs_completed ?? 0} completed · <span style={{ color: summary?.ingestion_jobs_failed ? 'var(--danger)' : undefined }}>{summary?.ingestion_jobs_failed ?? 0} failed</span></td></tr>
              </tbody>
            </table>
          ) : (
            <div className="empty">Backend unreachable</div>
          )}
        </div>

        <div className="card">
          <h2>Recent queries <Link to="/traces" className="small" style={{ float: 'right' }}>all traces →</Link></h2>
          {traces.length === 0 ? (
            <div className="empty">No queries yet — ask one in <Link to="/chat">Knowledge Query</Link>.</div>
          ) : (
            <table className="data">
              <thead><tr><th>Question</th><th>Latency</th><th>Outcome</th></tr></thead>
              <tbody>
                {traces.slice(0, 6).map((t) => (
                  <tr key={t.id} className="clickable" onClick={() => (window.location.href = `/traces/${t.id}`)}>
                    <td>{t.question.slice(0, 60)}{t.question.length > 60 ? '…' : ''}</td>
                    <td>{t.total_latency_ms ? `${Math.round(t.total_latency_ms)} ms` : '–'}</td>
                    <td>{t.insufficient_evidence ? <span className="tag amber">no evidence</span> : <span className="tag green">answered</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
