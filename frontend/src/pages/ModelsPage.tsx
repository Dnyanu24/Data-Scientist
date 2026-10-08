import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Zap, BrainCircuit, AlertCircle, RefreshCw } from 'lucide-react';
import { deploymentsApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';
import PredictPlaygroundDrawer, { type SelectedEndpoint } from '../components/PredictPlaygroundDrawer';

interface DeploymentRecord {
  id: number;
  endpoint_name: string;
  model_name?: string;
  algorithm_name?: string;
  project_name?: string;
  project_id?: number;
  status: string;
  avg_latency_ms?: number;
  request_count?: number;
  metrics?: Record<string, number>;
}

export default function ModelsPage() {
  const [deployments, setDeployments] = useState<DeploymentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEndpoint, setSelectedEndpoint] = useState<SelectedEndpoint | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();
  const { addToast } = useToast();

  const fetchDeployments = () => {
    setLoading(true);
    setError(null);
    deploymentsApi
      .list()
      .then((r) => setDeployments(r.data || []))
      .catch(() => {
        setError('Failed to load deployments. Check that the backend is running.');
        addToast('Failed to load deployed models', 'error');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchDeployments(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const formatMetric = (key: string, val: number): string => {
    const pctKeys = ['accuracy', 'r2', 'roc_auc', 'f1', 'precision', 'recall', 'auc'];
    if (pctKeys.some((k) => key.toLowerCase().includes(k))) return (val * 100).toFixed(1) + '%';
    return val.toFixed(4);
  };

  const getPrimaryMetric = (metrics?: Record<string, number>): string => {
    if (!metrics || Object.keys(metrics).length === 0) return '—';
    const priority = ['accuracy', 'roc_auc', 'r2', 'f1', 'auc'];
    for (const key of priority) {
      const found = Object.entries(metrics).find(([k]) => k.toLowerCase().includes(key));
      if (found) return formatMetric(found[0], found[1]);
    }
    const [k, v] = Object.entries(metrics)[0];
    return formatMetric(k, v);
  };

  return (
    <div className="page-body">
      <PredictPlaygroundDrawer
        endpoint={selectedEndpoint}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Model Registry &amp; Active Endpoints</h1>
          <p className="page-desc">Production deployed machine learning models with live monitoring metrics</p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={fetchDeployments}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
          disabled={loading}
        >
          <RefreshCw size={14} />
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      <div className="card">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px 24px' }}>
            <span className="spinner" style={{ width: 36, height: 36 }} />
            <p style={{ marginTop: 12, color: '#64748b' }}>Loading deployed models…</p>
          </div>
        ) : error ? (
          <div style={{ textAlign: 'center', padding: '48px 24px', color: '#ef4444', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
            <AlertCircle size={36} />
            <p style={{ fontWeight: 600 }}>{error}</p>
            <button className="btn btn-primary btn-sm" onClick={fetchDeployments}>Retry</button>
          </div>
        ) : deployments.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '64px 24px', color: '#94a3b8', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
            <BrainCircuit size={48} strokeWidth={1.2} />
            <p style={{ fontSize: 16, fontWeight: 600, color: '#475569' }}>No deployed models yet</p>
            <p style={{ fontSize: 13, maxWidth: 380 }}>
              Complete the AutoML pipeline in a project and deploy your best-performing model to see it here.
            </p>
            <button className="btn btn-primary" onClick={() => navigate('/projects')} style={{ marginTop: 4 }}>
              Go to Projects
            </button>
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Endpoint Name</th>
                  <th>Model Architecture</th>
                  <th>Project Scope</th>
                  <th>Status</th>
                  <th>Accuracy / Score</th>
                  <th>Avg Latency</th>
                  <th>Inference Calls</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {deployments.map((m) => (
                  <tr key={m.id}>
                    <td style={{ fontWeight: 700, color: '#0f172a' }}>
                      <code>{m.endpoint_name}</code>
                    </td>
                    <td>
                      <span className="badge badge-primary">
                        {m.algorithm_name || m.model_name || 'Unknown'}
                      </span>
                    </td>
                    <td style={{ color: '#475569' }}>
                      {m.project_name || ('Project #' + (m.project_id ?? '—'))}
                    </td>
                    <td>
                      <span className={m.status === 'active' ? 'badge badge-success' : 'badge badge-secondary'}>
                        {m.status === 'active' ? '● Active' : m.status}
                      </span>
                    </td>
                    <td>
                      <strong style={{ color: '#0284c7' }}>{getPrimaryMetric(m.metrics)}</strong>
                    </td>
                    <td>
                      <strong>
                        {m.avg_latency_ms != null && m.avg_latency_ms > 0
                          ? Math.round(m.avg_latency_ms) + 'ms'
                          : '—'}
                      </strong>
                    </td>
                    <td>{m.request_count != null ? m.request_count.toLocaleString() : '0'}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => {
                          setSelectedEndpoint(m as SelectedEndpoint);
                          setDrawerOpen(true);
                        }}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
                      >
                        <Zap size={14} /> Predict Playground
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
