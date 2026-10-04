import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { experimentsApi, deploymentsApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ArrowLeft, Rocket, Trophy, BarChart3 } from 'lucide-react';

export default function ModelDetailPage() {
  const { projectId, modelId } = useParams<{ projectId: string; modelId: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();

  const [model, setModel] = useState<Record<string, unknown> | null>(null);
  const [evaluation, setEval] = useState<Record<string, unknown> | null>(null);
  const [explain, setExplain] = useState<Record<string, unknown> | null>(null);
  const [activeTab, setActiveTab] = useState('metrics');
  const [deploying, setDeploying] = useState(false);

  useEffect(() => {
    const id = Number(modelId);
    experimentsApi.getModel(id).then((r) => setModel(r.data)).catch(() => {});
    experimentsApi.getEvaluation(id).then((r) => setEval(r.data)).catch(() => {});
    experimentsApi.getExplainability(id).then((r) => setExplain(r.data)).catch(() => {});
  }, [modelId]);

  const deployModel = async () => {
    setDeploying(true);
    try {
      await deploymentsApi.deploy(Number(modelId));
      addToast('Model deployed!', 'success');
      navigate(`/projects/${projectId}/predict`);
    } catch { addToast('Deploy failed', 'error'); }
    finally { setDeploying(false); }
  };

  if (!model) return <div className="loading-center"><div className="spinner" style={{ width: 40, height: 40 }} /></div>;

  const metrics = (model.metrics as Record<string, number>) || {};
  const featureImportance = (explain?.feature_importance as Record<string, number>) || {};
  const fiEntries = Object.entries(featureImportance).sort((a, b) => b[1] - a[1]).slice(0, 15);
  const maxFI = fiEntries[0]?.[1] || 1;

  const metricsData = Object.entries(metrics).filter(([, v]) => typeof v === 'number' && isFinite(v as number))
    .map(([k, v]) => ({ name: k, value: parseFloat((v as number).toFixed(4)) }));

  return (
    <div className="page-body">
      <div className="page-header">
        <div className="flex items-center gap-4" style={{ marginBottom: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={() => navigate(`/projects/${projectId}`)} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <ArrowLeft size={16} /> Back
          </button>
          <h1 className="page-title">{model.name as string}</h1>
          <span className="badge badge-accent">{model.algorithm_name as string}</span>
          {model.is_best && (
            <span className="badge badge-warning" style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <Trophy size={14} /> Champion
            </span>
          )}
        </div>
        <p className="page-desc">Stage {model.training_stage as number} · {model.status as string}</p>
      </div>

      <div className="flex gap-2" style={{ marginBottom: 20 }}>
        <button className="btn btn-success" onClick={deployModel} disabled={deploying} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          {deploying ? <span className="spinner" /> : <Rocket size={16} />} Deploy Model
        </button>
        <button className="btn btn-secondary" onClick={() => experimentsApi.setBest(Number(modelId)).then(() => addToast('Set as champion', 'success'))} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <Trophy size={16} /> Set Champion
        </button>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {['metrics', 'feature importance', 'evaluation', 'hyperparams'].map((t) => (
          <button key={t} className={`tab-btn ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}
            style={{ textTransform: 'capitalize' }}>{t}</button>
        ))}
      </div>

      {activeTab === 'metrics' && (
        <div>
          <div className="metrics-grid" style={{ marginBottom: 24 }}>
            {Object.entries(metrics).filter(([, v]) => typeof v === 'number').map(([k, v]) => (
              <div key={k} className="metric-item">
                <div className="metric-label">{k}</div>
                <div className="metric-value">{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
              </div>
            ))}
          </div>
          {metricsData.length > 0 && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 16 }}>Metrics Chart</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metricsData}>
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 13 }} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {metricsData.map((_, i) => (
                      <Cell key={i} fill={['#4f8ef7','#7c3aed','#10b981','#f59e0b','#ef4444','#06b6d4'][i % 6]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {activeTab === 'feature importance' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 20 }}>Feature Importance</div>
          {fiEntries.length === 0 ? (
            <div className="empty-state" style={{ padding: 32 }}>
              <div className="empty-icon"><BarChart3 size={32} /></div>
              <div className="empty-title">Not available</div>
            </div>
          ) : (
            fiEntries.map(([feat, val]) => (
              <div key={feat} className="fi-row">
                <div className="fi-label">{feat}</div>
                <div className="fi-bar-container">
                  <div className="fi-bar" style={{ width: `${(val / maxFI) * 100}%` }} />
                </div>
                <div className="fi-value">{val.toFixed(3)}</div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'evaluation' && (
        <div>
          {evaluation?.classification_report && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title" style={{ marginBottom: 12 }}>Classification Report</div>
              <pre style={{ fontSize: 12, color: 'var(--text-secondary)', overflowX: 'auto', fontFamily: 'var(--font-mono)' }}>
                {JSON.stringify(evaluation.classification_report, null, 2)}
              </pre>
            </div>
          )}
          {evaluation?.confusion_matrix && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 12 }}>Confusion Matrix</div>
              <pre style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                {JSON.stringify(evaluation.confusion_matrix, null, 2)}
              </pre>
            </div>
          )}
          {!evaluation?.classification_report && !evaluation?.confusion_matrix && (
            <div className="empty-state">
              <div className="empty-icon"><BarChart3 size={32} /></div>
              <div className="empty-title">No evaluation data</div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'hyperparams' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Hyperparameters</div>
          <pre style={{ fontSize: 13, color: 'var(--text-secondary)', overflowX: 'auto', fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
            {JSON.stringify(model.hyperparameters || {}, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
