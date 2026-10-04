import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ToastProvider } from './contexts/ToastContext';
import TopHeader from './components/TopHeader';
import Sidebar from './components/Sidebar';
import AuthPage from './pages/AuthPage';
import DashboardPage from './pages/DashboardPage';
import ProjectPage from './pages/ProjectPage';
import ModelDetailPage from './pages/ModelDetailPage';
import ChatPage from './pages/ChatPage';
import PredictPage from './pages/PredictPage';
import MonitoringPage from './pages/MonitoringPage';
import DatasetsPage from './pages/DatasetsPage';
import ExperimentsPage from './pages/ExperimentsPage';
import ModelsPage from './pages/ModelsPage';
import KnowledgePage from './pages/KnowledgePage';
import SettingsPage from './pages/SettingsPage';

// Protected shell — only rendered when user is authenticated
function AuthenticatedShell({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-base)' }}>
      <TopHeader />
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}

// Require authentication — redirect to /login if not logged in
function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: '#f0f6ff' }}>
        <div className="spinner" style={{ width: 48, height: 48 }} />
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  return (
    <AuthenticatedShell>
      {children}
    </AuthenticatedShell>
  );
}

// All routes — requires BrowserRouter, AuthProvider, ToastProvider in parent
function AppRoutes() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: '#f0f6ff' }}>
        <div className="spinner" style={{ width: 48, height: 48 }} />
      </div>
    );
  }

  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <AuthPage />} />

      {/* Protected */}
      <Route path="/" element={<RequireAuth><DashboardPage /></RequireAuth>} />
      <Route path="/projects" element={<RequireAuth><DashboardPage /></RequireAuth>} />
      <Route path="/projects/:id" element={<RequireAuth><ProjectPage /></RequireAuth>} />
      <Route path="/projects/:projectId/model/:modelId" element={<RequireAuth><ModelDetailPage /></RequireAuth>} />
      <Route path="/projects/:projectId/chat" element={<RequireAuth><ChatPage /></RequireAuth>} />
      <Route path="/projects/:projectId/predict" element={<RequireAuth><PredictPage /></RequireAuth>} />
      <Route path="/datasets" element={<RequireAuth><DatasetsPage /></RequireAuth>} />
      <Route path="/experiments" element={<RequireAuth><ExperimentsPage /></RequireAuth>} />
      <Route path="/models" element={<RequireAuth><ModelsPage /></RequireAuth>} />
      <Route path="/monitoring" element={<RequireAuth><MonitoringPage /></RequireAuth>} />
      <Route path="/monitoring/:deploymentId" element={<RequireAuth><MonitoringPage /></RequireAuth>} />
      <Route path="/knowledge" element={<RequireAuth><KnowledgePage /></RequireAuth>} />
      <Route path="/settings" element={<RequireAuth><SettingsPage /></RequireAuth>} />

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ToastProvider>
          <AppRoutes />
        </ToastProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
