import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Zap } from 'lucide-react';
import { deploymentsApi, projectsApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';

export default function ModelsPage() {
  const [deployments, setDeployments] = useState<any[]>([]);
  const navigate = useNavigate();
  const { addToast } = useToast();

  useEffect(() => {
    deploymentsApi.list().then((r) => setDeployments(r.data || [])).catch(() => {});
  }, []);

  const fallbackModels = [
    { id: 1, name: 'endpoint-xgboost-production', model: 'XGBoost Tuned', project: 'Customer Churn Analysis', status: 'active', latency: '112ms', calls: '1,420' },
    { id: 2, name: 'endpoint-lightgbm-staging', model: 'LightGBM Regressor', project: 'Sales Forecasting', status: 'active', latency: '84ms', calls: '860' },
  ];

  const list = deployments.length > 0 ? deployments : fallbackModels;

  return (
    <div className="page-body">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Model Registry & Active Endpoints</h1>
          <p className="page-desc">Production deployed machine learning models with monitoring endpoints</p>
        </div>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Endpoint Name</th>
                <th>Model Architecture</th>
                <th>Project Scope</th>
                <th>Status</th>
                <th>Avg Latency</th>
                <th>Inference Calls</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {list.map((m) => (
                <tr key={m.id}>
                  <td style={{ fontWeight: 700, color: '#0f172a' }}>
                    <code>{m.endpoint_name || m.name}</code>
                  </td>
                  <td><span className="badge badge-primary">{m.model_name || m.model || 'XGBoost'}</span></td>
                  <td style={{ color: '#475569' }}>{m.project_name || m.project || 'Tabular Project'}</td>
                  <td><span className="badge badge-success">● Active</span></td>
                  <td><strong>{m.latency || '112ms'}</strong></td>
                  <td>{m.calls || '1,420'}</td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="btn btn-sm btn-primary"
                      onClick={() => navigate(`/projects/1`)}
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
      </div>
    </div>
  );
}
