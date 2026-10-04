import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { deploymentsApi, projectsApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';

export default function PredictPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();

  const [deployments, setDeployments] = useState<Array<Record<string, unknown>>>([]);
  const [selectedDeploy, setSelectedDeploy] = useState<number | null>(null);
  const [jsonInput, setJsonInput] = useState('{}');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<unknown[]>([]);
  const [loading, setLoading] = useState(false);
  const [project, setProject] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    projectsApi.get(Number(projectId)).then((r) => setProject(r.data)).catch(() => {});
    deploymentsApi.list(Number(projectId))
      .then((r) => {
        setDeployments(r.data);
        const active = r.data.find((d: Record<string, unknown>) => d.status === 'active');
        if (active) setSelectedDeploy(active.id as number);
      })
      .catch(() => {});
  }, [projectId]);

  useEffect(() => {
    if (!selectedDeploy) return;
    deploymentsApi.history(selectedDeploy).then((r) => setHistory(r.data)).catch(() => {});
  }, [selectedDeploy]);

  const predict = async () => {
    if (!selectedDeploy) return addToast('Select a deployment', 'error');
    let features: Record<string, unknown>;
    try {
      features = JSON.parse(jsonInput);
    } catch {
      return addToast('Invalid JSON input', 'error');
    }
    setLoading(true);
    try {
      const r = await deploymentsApi.predict(selectedDeploy, features);
      setResult(r.data);
      addToast('Prediction successful!', 'success');
      deploymentsApi.history(selectedDeploy).then((hr) => setHistory(hr.data)).catch(() => {});
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Prediction failed';
      addToast(msg, 'error');
    } finally { setLoading(false); }
  };

  const loadSampleInput = () => {
    if (!project || !project.columns_info) {
      // Generate a generic sample
      const sample: Record<string, unknown> = { feature_1: 0, feature_2: 0 };
      setJsonInput(JSON.stringify(sample, null, 2));
      return;
    }
  };

  return (
    <div className="page-body">
      <div className="page-header flex items-center gap-4">
        <button className="btn btn-secondary btn-sm" onClick={() => navigate(`/projects/${projectId}`)}>← Back</button>
        <div>
          <h1 className="page-title">🔮 Real-time Prediction</h1>
          <p className="page-desc">Run inference against deployed model</p>
        </div>
      </div>

      {deployments.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🚀</div>
          <div className="empty-title">No active deployments</div>
          <div className="empty-desc">Deploy a trained model first from the project page</div>
        </div>
      ) : (
        <div className="grid-2" style={{ alignItems: 'flex-start' }}>
          {/* Input Panel */}
          <div>
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title" style={{ marginBottom: 16 }}>Select Deployment</div>
              <select className="form-select" value={selectedDeploy || ''} onChange={(e) => setSelectedDeploy(Number(e.target.value))}>
                {deployments.map((d) => (
                  <option key={d.id as number} value={d.id as number}>
                    {d.endpoint_name as string} ({d.status as string})
                  </option>
                ))}
              </select>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">Input Features (JSON)</div>
                <button className="btn btn-secondary btn-sm" onClick={loadSampleInput}>Sample</button>
              </div>
              <textarea
                className="form-textarea"
                style={{ fontFamily: 'var(--font-mono)', minHeight: 240, fontSize: 13 }}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder='{"feature_1": 1.5, "feature_2": "category_A", ...}'
              />
              <button className="btn btn-primary w-full" style={{ marginTop: 12 }} onClick={predict} disabled={loading}>
                {loading ? <span className="spinner" /> : '🔮'}
                {loading ? 'Predicting...' : 'Run Prediction'}
              </button>
            </div>
          </div>

          {/* Output Panel */}
          <div>
            {result && (
              <div className="card" style={{ marginBottom: 16, borderColor: 'var(--success)' }}>
                <div className="card-title" style={{ marginBottom: 16 }}>✅ Prediction Result</div>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>PREDICTION</div>
                  <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--primary)', fontFamily: 'var(--font-mono)' }}>
                    {String(result.prediction)}
                  </div>
                </div>
                {result.probability && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>PROBABILITY</div>
                    <div style={{ fontSize: 16, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                      {JSON.stringify(result.probability, null, 2)}
                    </div>
                  </div>
                )}
                <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                  <span>⚡ {result.latency_ms as number}ms</span>
                  <span>🕐 {new Date(result.timestamp as string).toLocaleTimeString()}</span>
                </div>
              </div>
            )}

            {/* Prediction History */}
            {history.length > 0 && (
              <div className="card">
                <div className="card-title" style={{ marginBottom: 16 }}>Recent Predictions</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 320, overflowY: 'auto' }}>
                  {(history as Array<Record<string, unknown>>).slice(0, 10).map((h) => (
                    <div key={h.id as number} style={{
                      padding: '10px 12px', background: 'var(--bg-input)', borderRadius: 8,
                      fontSize: 13, display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                    }}>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>
                        {JSON.stringify((h.prediction as Record<string, unknown>)?.prediction ?? h.prediction)}
                      </span>
                      <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-muted)' }}>
                        <span>{h.latency_ms as number}ms</span>
                        <span>{new Date(h.created_at as string).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
