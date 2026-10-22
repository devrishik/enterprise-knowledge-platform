import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { ConversationItem, QueryResponse } from '../types'

interface Turn {
  question: string
  response?: QueryResponse
  error?: string
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.listConversations<ConversationItem[]>().then(setConversations).catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  async function ask(q: string) {
    if (!q.trim() || busy) return
    setBusy(true)
    setQuestion('')
    setTurns((t) => [...t, { question: q }])
    try {
      const response = await api.query(q, conversationId)
      setConversationId(response.conversation_id)
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response } : turn)))
      api.listConversations<ConversationItem[]>().then(setConversations).catch(() => {})
    } catch (e) {
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: (e as Error).message } : turn)))
    } finally {
      setBusy(false)
    }
  }

  async function loadConversation(id: string) {
    const detail = await api.getConversation<ConversationItem>(id)
    setConversationId(id)
    const pairs: Turn[] = []
    const messages = detail.messages ?? []
    for (let i = 0; i < messages.length; i += 2) {
      pairs.push({ question: messages[i]?.content ?? '', response: undefined })
      const assistant = messages[i + 1]
      if (assistant) {
        pairs[pairs.length - 1].response = {
          answer: assistant.content,
          citations: assistant.citations,
          confidence: 0, insufficient_evidence: false, conflicting_evidence: false,
          rewritten_question: null, sub_queries: [], retrieval_confidence: 0,
          context_chunks: [], trace_id: assistant.trace_id ?? '', conversation_id: id,
          message_id: assistant.id, stages: [], model: '',
          token_usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          cost_usd: null, total_latency_ms: 0,
        }
      }
    }
    setTurns(pairs)
  }

  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: conversations.length ? '260px 1fr' : '1fr' }}>
      {conversations.length > 0 && (
        <div className="card">
          <h2>Conversations</h2>
          {conversations.map((c) => (
            <div key={c.id} className="chunk-card" style={{ marginBottom: 6, cursor: 'pointer' }}
              onClick={() => loadConversation(c.id)}>
              <div className="small">{c.title.slice(0, 70)}</div>
              <div className="muted small">{new Date(c.created_at).toLocaleString()}</div>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h2>Knowledge Query</h2>
        <div className="chat-thread">
          {turns.length === 0 && (
            <div className="empty">
              Ask a question about the ingested documents. Try:<br />
              <em>“How many days of paid annual leave do employees get?”</em><br />
              <em>“What is the meal per diem for international travel?”</em><br />
              <em>“What is the CEO's favorite color?” (expect a grounded refusal)</em>
            </div>
          )}
          {turns.map((turn, i) => (
            <div key={i}>
              <div className="message user">
                <div className="role">You</div>
                {turn.question}
              </div>
              <div className="message assistant">
                <div className="role">Assistant {turn.response?.model ? `· ${turn.response.model}` : ''}</div>
                {turn.error ? (
                  <div className="error-banner">{turn.error}</div>
                ) : !turn.response ? (
                  <span className="progress-line">running agent workflow…</span>
                ) : (
                  <>
                    {turn.response.insufficient_evidence && (
                      <div className="tag amber" style={{ marginBottom: 8 }}>insufficient evidence — grounded refusal</div>
                    )}
                    {turn.response.conflicting_evidence && (
                      <div className="tag amber" style={{ marginBottom: 8 }}>conflicting sources reported</div>
                    )}
                    <div className="answer">{turn.response.answer}</div>

                    {turn.response.rewritten_question && (
                      <div className="muted small" style={{ marginTop: 8 }}>
                        Rewritten query: <span className="mono">{turn.response.rewritten_question}</span>
                      </div>
                    )}

                    {turn.response.citations.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <h3>Citations</h3>
                        {turn.response.citations.map((c) => (
                          <div key={c.ref} className="citation">
                            <div className="cite-head">
                              [{c.ref}] {c.document_name}
                              {c.page != null ? `, page ${c.page}` : ''}
                              {c.section ? `, “${c.section}”` : ''}
                            </div>
                            {c.quote ? <div className="excerpt">“{c.quote}”</div> : null}
                            <div className="excerpt">{c.excerpt.slice(0, 220)}…</div>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="stage-bar">
                      {turn.response.stages.map((s, j) => (
                        <span key={j} className="stage-pill">
                          {s.stage.replace(/_/g, ' ')} · {s.latency_ms.toFixed(1)} ms
                        </span>
                      ))}
                    </div>
                    <div className="muted small" style={{ marginTop: 8 }}>
                      confidence {turn.response.confidence} · retrieval {turn.response.retrieval_confidence} ·
                      total {Math.round(turn.response.total_latency_ms)} ms
                      {turn.response.token_usage?.total_tokens
                        ? ` · ${turn.response.token_usage.total_tokens} tokens` : ''}
                      {turn.response.cost_usd != null ? ` · $${turn.response.cost_usd.toFixed(6)}` : ''}
                      {' · '}
                      <Link to={`/traces/${turn.response.trace_id}`}>view full trace →</Link>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <form
          style={{ display: 'flex', gap: 10, marginTop: 16 }}
          onSubmit={(e) => { e.preventDefault(); ask(question) }}
        >
          <input
            type="text"
            value={question}
            placeholder="Ask about your documents…"
            onChange={(e) => setQuestion(e.target.value)}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !question.trim()}>Ask</button>
        </form>
      </div>
    </div>
  )
}
