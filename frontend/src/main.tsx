import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App, { LoginScreen } from './App'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import DocumentDetail from './pages/DocumentDetail'
import Chat from './pages/Chat'
import Evaluation from './pages/Evaluation'
import Traces from './pages/Traces'
import TraceDetail from './pages/TraceDetail'
import './styles/global.css'

const router = createBrowserRouter([
  { path: '/login', element: <LoginScreen /> },
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'documents', element: <Documents /> },
      { path: 'documents/:id', element: <DocumentDetail /> },
      { path: 'chat', element: <Chat /> },
      { path: 'evaluation', element: <Evaluation /> },
      { path: 'traces', element: <Traces /> },
      { path: 'traces/:id', element: <TraceDetail /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
