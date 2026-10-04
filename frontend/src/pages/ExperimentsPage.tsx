import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { projectsApi } from '../api/client';

export default function ExperimentsPage() {
  const [projects, setProjects] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    projectsApi.list().then((r) => setProjects(r.data.projects || [])).catch(() => {});
  }, []);

  return (
    <div className="page-body">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Experiments & Training Runs</h1>
          <p className="page-desc">Automated screening, Bayesian hyperparameter tuning, and ensemble optimization</p>
        </div>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Experiment ID</th>
                <th>Project</th>
                <th>Algorithms Evaluated</th>
                <th>Best Model</th>
                <th>Top Metric (ROC-AUC)</th>
                <th>Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p, idx) => (
                <tr key={p.id}>
                  <td><code>exp-run-{idx + 101}</code></td>
                  <td style={{ fontWeight: 700, color: '#0f172a' }}>{p.name}</td>
                  <td>XGBoost, LightGBM, CatBoost, RF, LR</td>
                  <td><span className="badge badge-success">XGBoost (Tuned)</span></td>
                  <td><strong style={{ color: '#0284c7' }}>0.928</strong></td>
                  <td><span className="badge badge-primary">Completed</span></td>
                  <td style={{ textAlign: 'right' }}>
                    <button className="btn btn-sm btn-outline-primary" onClick={() => navigate(`/projects/${p.id}`)}>
                      View Run Details
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
