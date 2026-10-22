// Typed API client. The demo API key is stored in localStorage after the
// user picks a demo identity on the login screen.

import type { DemoUser } from '../types'

const KEY_STORAGE = 'ekip_api_key'
const USER_STORAGE = 'ekip_user'

export function getApiKey(): string | null {
  return localStorage.getItem(KEY_STORAGE)
}

export function getCurrentUser(): string | null {
  return localStorage.getItem(USER_STORAGE)
}

export function setIdentity(apiKey: string, displayName: string) {
  localStorage.setItem(KEY_STORAGE, apiKey)
  localStorage.setItem(USER_STORAGE, displayName)
}

export function clearIdentity() {
  localStorage.removeItem(KEY_STORAGE)
  localStorage.removeItem(USER_STORAGE)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  }
  const key = getApiKey()
  if (key) headers['X-API-Key'] = key
  if (init.body && !(init.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }
  const response = await fetch(path, { ...init, headers })
  if (response.status === 204) return undefined as T
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.error?.message ?? `Request failed (${response.status})`
    throw new Error(message)
  }
  return payload as T
}

export const api = {
  demoUsers: () => request<DemoUser[]>('/api/auth/demo-users'),
  health: <T>() => request<T>('/api/health'),
  metricsSummary: <T>() => request<T>('/api/metrics/summary'),
  listDocuments: <T>() => request<T>('/api/documents'),
  getDocument: <T>(id: string) => request<T>(`/api/documents/${id}`),
  getChunks: <T>(id: string) => request<T>(`/api/documents/${id}/chunks`),
  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: 'DELETE' }),
  uploadDocument: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<{
      document_id: string; filename: string; status: string
      chunk_count: number; duration_ms: number; error: string | null
    }>('/api/documents/upload', { method: 'POST', body: form })
  },
  query: (question: string, conversationId: string | null) =>
    request<import('../types').QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify({ question, conversation_id: conversationId }),
    }),
  listConversations: <T>() => request<T>('/api/conversations'),
  getConversation: <T>(id: string) => request<T>(`/api/conversations/${id}`),
  listTraces: <T>() => request<T>('/api/traces'),
  getTrace: <T>(id: string) => request<T>(`/api/traces/${id}`),
  runEvaluation: (strategies: string[], includeGeneration: boolean) =>
    request<{ run_ids: string[] }>('/api/evaluation/run', {
      method: 'POST',
      body: JSON.stringify({ strategies, include_generation: includeGeneration }),
    }),
  listEvaluationRuns: <T>() => request<T>('/api/evaluation/runs'),
  getEvaluationRun: <T>(id: string) => request<T>(`/api/evaluation/runs/${id}`),
}
