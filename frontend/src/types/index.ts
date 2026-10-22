// Shared API types mirroring the backend Pydantic schemas.

export interface DemoUser {
  tenant_name: string
  tenant_slug: string
  email: string
  display_name: string
  api_key: string
}

export interface DocumentItem {
  id: string
  filename: string
  doc_type: string
  title: string
  status: string
  chunk_count: number
  size_bytes: number
  error_message: string | null
  metadata: Record<string, unknown>
  created_at: string
  content_hash?: string
}

export interface ChunkItem {
  id: string
  chunk_index: number
  text: string
  token_count: number
  page_number: number | null
  section: string | null
  metadata: Record<string, unknown>
}

export interface Citation {
  ref: number
  chunk_id: string
  document_id: string
  document_name: string
  page: number | null
  section: string | null
  excerpt: string
  quote?: string
}

export interface RetrievedChunk {
  chunk_id: string
  document_filename: string
  score: number
  source: string
  page_number: number | null
  section: string | null
  text: string
}

export interface StageInfo {
  stage: string
  latency_ms: number
  details: Record<string, unknown>
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  confidence: number
  insufficient_evidence: boolean
  conflicting_evidence: boolean
  rewritten_question: string | null
  sub_queries: string[]
  retrieval_confidence: number
  context_chunks: RetrievedChunk[]
  trace_id: string
  conversation_id: string
  message_id: string
  stages: StageInfo[]
  model: string
  token_usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
  cost_usd: number | null
  total_latency_ms: number
}

export interface ConversationItem {
  id: string
  title: string
  created_at: string
  messages?: MessageItem[]
}

export interface MessageItem {
  id: string
  role: string
  content: string
  citations: Citation[]
  trace_id: string | null
  created_at: string
}

export interface TraceListItem {
  id: string
  question: string
  total_latency_ms: number | null
  insufficient_evidence: boolean
  created_at: string | null
}

export interface TraceDetail {
  id: string
  question: string
  rewritten_question: string | null
  sub_queries: string[]
  stages: StageInfo[]
  retrieved: {
    dense?: RetrievedChunk[]
    lexical?: RetrievedChunk[]
    fused?: RetrievedChunk[]
    reranked?: RetrievedChunk[]
  }
  final_citations: Citation[]
  answer: string | null
  insufficient_evidence: boolean
  confidence: number | null
  model: string | null
  token_usage: Record<string, number>
  cost_usd: number | null
  total_latency_ms: number | null
  error: string | null
  created_at: string | null
}

export interface EvaluationRun {
  id: string
  strategy: string
  status: string
  question_count: number
  summary: Record<string, any>
  per_question: any[]
  started_at: string | null
  finished_at: string | null
  created_at: string | null
}

export interface MetricsSummary {
  documents: number
  chunks: number
  queries: number
  ingestion_jobs_completed: number
  ingestion_jobs_failed: number
  avg_query_latency_ms: number | null
}

export interface HealthStatus {
  status: string
  database: string
  vector_backend: string
  llm_provider: string
  llm_configured: boolean
  embedding_provider: string
  reranker_provider: string
}
