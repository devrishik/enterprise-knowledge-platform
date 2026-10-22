import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ChunkItem, DocumentItem } from '../types'

export default function DocumentDetail() {
  const { id } = useParams<{ id: string }>()
  const [document, setDocument] = useState<DocumentItem | null>(null)
  const [chunks, setChunks] = useState<ChunkItem[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!id) return
    api.getDocument<DocumentItem>(id).then(setDocument).catch((e) => setError(e.message))
    api.getChunks<ChunkItem[]>(id).then(setChunks).catch((e) => setError(e.message))
  }, [id])

  if (error) return <div className="error-banner">{error}</div>
  if (!document) return <div className="empty">Loading…</div>

  return (
    <div>
      <div className="card">
        <h2>{document.filename} <Link to="/documents" className="small" style={{ float: 'right' }}>← library</Link></h2>
        <table className="data">
          <tbody>
            <tr><td className="muted" style={{ width: 180 }}>Title</td><td>{document.title}</td></tr>
            <tr><td className="muted">Type</td><td><span className="tag gray">{document.doc_type}</span></td></tr>
            <tr><td className="muted">Status</td><td>{document.status === 'completed' ? <span className="tag green">completed</span> : <span className="tag red">{document.status}</span>}</td></tr>
            <tr><td className="muted">Chunks</td><td>{document.chunk_count}</td></tr>
            <tr><td className="muted">Size</td><td>{(document.size_bytes / 1024).toFixed(1)} KB</td></tr>
            <tr><td className="muted">SHA-256</td><td className="mono small">{document.content_hash}</td></tr>
            {document.error_message && <tr><td className="muted">Error</td><td style={{ color: 'var(--danger)' }}>{document.error_message}</td></tr>}
          </tbody>
        </table>
        {Object.keys(document.metadata ?? {}).length > 0 && (
          <>
            <h3 style={{ marginTop: 16 }}>Ingestion metadata</h3>
            <pre className="mono" style={{ background: '#f8fafc', padding: 12, borderRadius: 6, overflow: 'auto' }}>
              {JSON.stringify(document.metadata, null, 2)}
            </pre>
          </>
        )}
      </div>

      <div className="card">
        <h2>Chunks ({chunks.length})</h2>
        <div className="chunk-list">
          {chunks.map((c) => (
            <div key={c.id} className="chunk-card">
              <div className="chunk-head">
                <span>
                  #{c.chunk_index}
                  {c.section ? <span className="tag blue" style={{ marginLeft: 8 }}>{c.section}</span> : null}
                  {c.page_number ? <span className="tag gray" style={{ marginLeft: 8 }}>page {c.page_number}</span> : null}
                </span>
                <span className="muted small">{c.token_count} tokens · <span className="mono">{c.id.slice(0, 8)}</span></span>
              </div>
              <div>{c.text.slice(0, 400)}{c.text.length > 400 ? '…' : ''}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
