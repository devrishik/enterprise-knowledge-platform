import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { EvaluationRun } from '../types'

const STRATEGY_LABELS: Record<string, string> = {
  dense: 'Dense only',
  lexical: 'Lexical only',
  hybrid: 'Hybrid (RRF)',
  hybrid_rerank: 'Hybrid + rerank',
  generation: 'Generation',
}

export default function Evaluation() {
  const [runs, setRuns] = useState<EvaluationRun[]>([])
  const [selected, setSelected] = useState<EvaluationRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [includeGeneration, setIncludeGeneration] = useState(false)
  const [error, setError] = useState('')

  async function refresh() {
    const list = await api.listEvaluationRuns<EvaluationRun[]>()
    setRuns(list)
    if (list.length > 0 && !selected) setSelected(list[0])
  }

  useEffect(() => {
    refresh().catch((e) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function runEvaluation() {
    setBusy(true)
    setError('')
    try {
      await api.runEvaluation(['dense', 'lexical', 'hybrid', 'hybrid_rerank'], includeGeneration)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // only the newest run per strategy (runs arrive newest-first)
  const latestByStrategy = new Map<string, EvaluationRun>()
  for (const run of runs) {
    if (!latestByStrategy.has(run.strategy)) latestByStrategy.set(run.strategy, run)
  }
  const retrievalRuns = [...latestByStrategy.values()].filter((r) => r.strategy !== 'generation')
  const generationRuns = [...latestByStrategy.values()].filter((r) => r.strategy === 'generation')

  const metricRows = retrievalRuns.map((r) => ({
    strategy: r.strategy,
    recall: r.summary['recall@5'],
    precision: r.summary['precision@5'],
    mrr: r.summary['mrr'],
    ndcg: r.summary['ndcg@5'],
    count: r.question_count,
  }))

  return (
    <div>
      {error && <div className="error-banner">{error}</div>}
      <div className="card">
        <h2>Run evaluation</h2>
        <p className="muted small">
          Executes every evaluation question through the live retrieval pipeline for all four
          strategies (dense / lexical / hybrid / hybrid+rerank) and stores the measured metrics.
          Numbers shown are always produced by actual runs — never pre-baked.
        </p>
        <label className="small" style={{ display: 'block', marginBottom: 10 }}>
          <input type="checkbox" checked={includeGeneration} style={{ width: 'auto', marginRight: 8 }}
            onChange={(e) => setIncludeGeneration(e.target.checked)} />
          Include generation evaluation (faithfulness / relevancy / citation correctness — one full
          agent run per question)
        </label>
        <button onClick={runEvaluation} disabled={busy}>
          {busy ? 'Running evaluation…' : 'Run evaluation'}
        </button>
      </div>

      <div className="card">
        <h2>Retrieval metrics by strategy</h2>
        {metricRows.length === 0 ? (
          <div className="empty">No evaluation runs yet.</div>
        ) : (
          <table className="data">
            <thead>
              <tr><th>Strategy</th><th>Recall@5</th><th>Precision@5</th><th>MRR</th><th>nDCG@5</th><th>Questions</th><th>Run</th></tr>
            </thead>
            <tbody>
              {metricRows.map((row) => (
                <tr key={row.strategy + row.count}>
                  <td><strong>{STRATEGY_LABELS[row.strategy] ?? row.strategy}</strong></td>
                  <td>{fmt(row.recall)}</td>
                  <td>{fmt(row.precision)}</td>
                  <td>{fmt(row.mrr)}</td>
                  <td>{fmt(row.ndcg)}</td>
                  <td>{row.count}</td>
                  <td className="muted small">{new Date(runs.find((r) => r.strategy === row.strategy)!.created_at ?? '').toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {generationRuns.length > 0 && (
        <div className="card">
          <h2>Generation quality</h2>
          {generationRuns.map((run) => (
            <div key={run.id}>
              <table className="data">
                <tbody>
                  <tr><td className="muted">Faithfulness rate</td><td><strong>{pct(run.summary['faithfulness_rate'])}</strong></td></tr>
                  <tr><td className="muted">Faithfulness, answerable only</td><td><strong>{pct(run.summary['faithfulness_rate_answerable'])}</strong></td></tr>
                  <tr><td className="muted">Avg relevancy (1–5)</td><td><strong>{run.summary['avg_relevancy']?.toFixed(2)}</strong></td></tr>
                  <tr><td className="muted">Avg relevancy, answerable only (1–5)</td><td><strong>{typeof run.summary['avg_relevancy_answerable'] === 'number' ? run.summary['avg_relevancy_answerable'].toFixed(2) : '–'}</strong></td></tr>
                  <tr><td className="muted">Citation correctness rate</td><td><strong>{pct(run.summary['citation_correctness_rate'])}</strong></td></tr>
                  <tr><td className="muted">Evaluator</td><td className="mono small">{run.summary['evaluator']}</td></tr>
                </tbody>
              </table>
              <p className="muted small">
                LLM-judged metrics are themselves imperfect; heuristic mode measures pipeline
                mechanics (keyword coverage), not answer quality. See docs/evaluation.md.
              </p>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h2>Per-question detail</h2>
        <select onChange={(e) => setSelected(runs.find((r) => r.id === e.target.value) ?? null)} value={selected?.id ?? ''}>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {STRATEGY_LABELS[r.strategy] ?? r.strategy} — {new Date(r.created_at ?? '').toLocaleString()}
            </option>
          ))}
        </select>
        {selected && selected.per_question.length > 0 && (
          <table className="data" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Question</th>
                <th>Type</th>
                {selected.strategy !== 'generation' ? (
                  <><th>Recall@5</th><th>MRR</th><th>Retrieved docs</th></>
                ) : (
                  <><th>Answered</th><th>Faithful</th><th>Citations OK</th></>
                )}
              </tr>
            </thead>
            <tbody>
              {selected.per_question.map((q: any) => (
                <tr key={q.question_id}>
                  <td style={{ maxWidth: 340 }}>{q.question}</td>
                  <td><span className="tag gray">{q.question_type}</span></td>
                  {selected.strategy !== 'generation' ? (
                    <>
                      <td>{fmt(q['recall@5'])}</td>
                      <td>{fmt(q['mrr'])}</td>
                      <td className="small">{(q.retrieved_docs ?? []).slice(0, 3).join(', ')}</td>
                    </>
                  ) : (
                    <>
                      <td>{q.insufficient_evidence ? <span className="tag amber">refused</span> : <span className="tag green">answered</span>}</td>
                      <td>{q.faithfulness?.faithful ? <span className="tag green">yes</span> : <span className="tag red">no</span>}</td>
                      <td>{q.citation_correctness?.pass ? <span className="tag green">yes</span> : <span className="tag red">no</span>}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function fmt(v: number | undefined) {
  return v === undefined ? '–' : v.toFixed(3)
}

function pct(v: number | undefined) {
  return v === undefined ? '–' : `${(v * 100).toFixed(1)}%`
}
