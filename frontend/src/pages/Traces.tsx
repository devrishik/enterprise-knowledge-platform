import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { TraceListItem } from '../types'

export default function Traces() {
  const navigate = useNavigate()
  const [traces, setTraces] = useState<TraceListItem[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    api.listTraces<TraceListItem[]>().then(setTraces).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="card">
      <h2>Query traces ({traces.length})</h2>
      {error && <div className="error-banner">{error}</div>}
      {traces.length === 0 ? (
        <div className="empty">No traces yet — every query records one automatically.</div>
      ) : (
        <table className="data">
          <thead><tr><th>Question</th><th>Latency</th><th>Outcome</th><th>When</th></tr></thead>
          <tbody>
            {traces.map((t) => (
              <tr key={t.id} className="clickable" onClick={() => navigate(`/traces/${t.id}`)}>
                <td>{t.question.slice(0, 90)}{t.question.length > 90 ? '…' : ''}</td>
                <td>{t.total_latency_ms != null ? `${Math.round(t.total_latency_ms)} ms` : '–'}</td>
                <td>{t.insufficient_evidence ? <span className="tag amber">no evidence</span> : <span className="tag green">answered</span>}</td>
                <td className="muted small">{t.created_at ? new Date(t.created_at).toLocaleString() : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
