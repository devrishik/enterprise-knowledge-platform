import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { DocumentItem } from '../types'

interface UploadState {
  filename: string
  status: 'uploading' | 'ok' | 'failed'
  chunks?: number
  error?: string
  ms?: number
}

export default function Documents() {
  const navigate = useNavigate()
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [uploads, setUploads] = useState<UploadState[]>([])
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const refresh = () => api.listDocuments<DocumentItem[]>().then(setDocuments).catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
  }, [])

  async function handleFiles(files: FileList | null) {
    if (!files) return
    for (const file of Array.from(files)) {
      setUploads((u) => [{ filename: file.name, status: 'uploading' }, ...u])
      try {
        const result = await api.uploadDocument(file)
        setUploads((u) =>
          u.map((x) =>
            x.filename === file.name && x.status === 'uploading'
              ? {
                  filename: file.name,
                  status: result.status === 'completed' ? 'ok' : 'failed',
                  chunks: result.chunk_count,
                  ms: Math.round(result.duration_ms),
                  error: result.error ?? undefined,
                }
              : x,
          ),
        )
        refresh()
      } catch (e) {
        setUploads((u) =>
          u.map((x) => (x.filename === file.name && x.status === 'uploading' ? { filename: file.name, status: 'failed', error: (e as Error).message } : x)),
        )
      }
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  async function handleDelete(id: string) {
    try {
      await api.deleteDocument(id)
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div>
      {error && <div className="error-banner">{error}</div>}
      <div className="card">
        <h2>Upload documents</h2>
        <p className="muted small">
          Supported: PDF, DOCX, TXT, Markdown, HTML (max size enforced server-side). Documents are
          parsed, structure-aware chunked, embedded and indexed for hybrid retrieval.
        </p>
        <input ref={inputRef} type="file" multiple accept=".pdf,.docx,.txt,.md,.markdown,.html,.htm"
          onChange={(e) => handleFiles(e.target.files)} />
        {uploads.length > 0 && (
          <div style={{ marginTop: 14 }}>
            {uploads.map((u, i) => (
              <div key={i} className="chunk-card" style={{ marginBottom: 6 }}>
                <div className="chunk-head">
                  <span>{u.filename}</span>
                  {u.status === 'uploading' && <span className="progress-line">ingesting…</span>}
                  {u.status === 'ok' && <span className="tag green">{u.chunks} chunks · {u.ms} ms</span>}
                  {u.status === 'failed' && <span className="tag red">failed</span>}
                </div>
                {u.error && <div className="muted small">{u.error}</div>}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Document library ({documents.length})</h2>
        {documents.length === 0 ? (
          <div className="empty">No documents yet.</div>
        ) : (
          <table className="data">
            <thead>
              <tr><th>Filename</th><th>Type</th><th>Status</th><th>Chunks</th><th>Size</th><th>Uploaded</th><th></th></tr>
            </thead>
            <tbody>
              {documents.map((d) => (
                <tr key={d.id} className="clickable" onClick={() => navigate(`/documents/${d.id}`)}>
                  <td><strong>{d.filename}</strong><br /><span className="muted small">{d.title?.slice(0, 60)}</span></td>
                  <td><span className="tag gray">{d.doc_type}</span></td>
                  <td>
                    {d.status === 'completed' && <span className="tag green">completed</span>}
                    {d.status === 'processing' && <span className="tag amber">processing</span>}
                    {d.status === 'failed' && <span className="tag red" title={d.error_message ?? ''}>failed</span>}
                  </td>
                  <td>{d.chunk_count}</td>
                  <td>{(d.size_bytes / 1024).toFixed(0)} KB</td>
                  <td className="muted small">{new Date(d.created_at).toLocaleString()}</td>
                  <td>
                    <button className="secondary" onClick={(e) => { e.stopPropagation(); handleDelete(d.id) }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
