import { useEffect, useState } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { api, getApiKey, getCurrentUser, clearIdentity } from './api/client'
import type { DemoUser, HealthStatus } from './types'

const NAV = [
  { to: '/', label: 'Dashboard', icon: '◧' },
  { to: '/documents', label: 'Documents', icon: '▤' },
  { to: '/chat', label: 'Knowledge Query', icon: '◈' },
  { to: '/evaluation', label: 'Evaluation', icon: '◑' },
  { to: '/traces', label: 'Query Traces', icon: '∮' },
]

export default function App() {
  const navigate = useNavigate()
  const [health, setHealth] = useState<HealthStatus | null>(null)

  useEffect(() => {
    if (!getApiKey()) {
      navigate('/login', { replace: true })
      return
    }
    api.health<HealthStatus>().then(setHealth).catch(() => setHealth(null))
  }, [navigate])

  if (!getApiKey()) return <Outlet />

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          Enterprise Knowledge<br />
          <span>Intelligence Platform</span>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : '')}>
              <span aria-hidden>{item.icon}</span> {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          {health ? (
            <>
              <div>DB: {health.database === 'up' ? '● healthy' : '● down'}</div>
              <div>LLM: {health.llm_provider}{health.llm_configured ? '' : ' (unconfigured)'}</div>
              <div>Vectors: {health.vector_backend}</div>
            </>
          ) : (
            <div>API unreachable</div>
          )}
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <h1>Enterprise Knowledge Intelligence Platform</h1>
          <div className="user-chip">
            <span>{getCurrentUser() ?? 'unknown'}</span>
            <button className="secondary" onClick={() => { clearIdentity(); window.location.href = '/login' }}>
              Switch user
            </button>
          </div>
        </header>
        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}

export function LoginScreen() {
  const navigate = useNavigate()
  const [users, setUsers] = useState<DemoUser[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    api.demoUsers().then(setUsers).catch((e) => setError(String(e.message)))
  }, [])

  return (
    <div className="login-screen">
      <div className="login-card">
        <h2 style={{ marginTop: 0 }}>Choose a demo identity</h2>
        <p className="muted small">
          Each identity belongs to a different tenant. Tenant isolation is enforced for every
          query: a tenant can never retrieve another tenant's documents.
        </p>
        {error && <div className="error-banner">{error} — is the backend running?</div>}
        {users.map((u) => (
          <button
            key={u.api_key}
            className="user-option"
            onClick={() => {
              localStorage.setItem('ekip_api_key', u.api_key)
              localStorage.setItem('ekip_user', `${u.display_name} · ${u.tenant_name}`)
              navigate('/')
            }}
          >
            <span>
              <strong>{u.display_name}</strong>
              <br />
              <span className="muted small">{u.tenant_name}</span>
            </span>
            <span className="tag blue">{u.tenant_slug}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
