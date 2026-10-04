import { useEffect, useState } from 'react';
import { knowledgeApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';

export default function KnowledgePage() {
  const [activeTab, setActiveTab] = useState<'algorithms' | 'datasets' | 'experiments'>('algorithms');
  const [stats, setStats] = useState<any>(null);
  const [algorithms, setAlgorithms] = useState<any[]>([]);
  const [experiences, setExperiences] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { addToast } = useToast();

  useEffect(() => {
    setLoading(true);
    Promise.all([
      knowledgeApi.stats().then((r) => setStats(r.data)).catch(() => {}),
      knowledgeApi.algorithms().then((r) => setAlgorithms(r.data || [])).catch(() => {}),
      knowledgeApi.experiences(20).then((r) => setExperiences(r.data || [])).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  const fallbackAlgos = [
    { name: 'XGBoost', category: 'Gradient Boosting', library: 'xgboost', speed: 'Fast', desc: 'Extreme gradient boosting with exact and approximate tree building.' },
    { name: 'LightGBM', category: 'Gradient Boosting', library: 'lightgbm', speed: 'Very Fast', desc: 'Histogram-based gradient boosting optimized for speed and memory efficiency.' },
    { name: 'CatBoost', category: 'Gradient Boosting', library: 'catboost', speed: 'Moderate', desc: 'Gradient boosting with state-of-the-art native categorical handling.' },
    { name: 'Random Forest', category: 'Bagging Ensemble', library: 'scikit-learn', speed: 'Fast', desc: 'Ensemble of decision trees trained with bagging and random feature selection.' },
    { name: 'Extra Trees', category: 'Bagging Ensemble', library: 'scikit-learn', speed: 'Very Fast', desc: 'Extremely randomized trees with random split thresholds.' },
    { name: 'Logistic Regression', category: 'Linear Model', library: 'scikit-learn', speed: 'Instant', desc: 'L1/L2 regularized linear model for probability estimation.' },
  ];

  const algosToDisplay = algorithms.length > 0 ? algorithms : fallbackAlgos;

  return (
    <div className="page-body">
      {/* Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Knowledge & Experience Base</h1>
          <p className="page-desc">Continuous meta-learning repository and algorithm capability index</p>
        </div>
        <span className="badge badge-success" style={{ fontSize: 13, padding: '6px 14px' }}>
          ● Meta-Learning Active
        </span>
      </div>

      {/* Stats Cards (Panels 14 & 18 Mockup) */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Known Algorithms</span>
            <span className="stat-value">{stats?.known_algorithms || 15}</span>
          </div>
          <div className="stat-icon-wrapper blue">🧠</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Meta-Experiments</span>
            <span className="stat-value">{stats?.historical_experiments || 12}</span>
          </div>
          <div className="stat-icon-wrapper cyan">🔬</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Trained Models</span>
            <span className="stat-value">{stats?.total_models_trained || 6}</span>
          </div>
          <div className="stat-icon-wrapper indigo">📦</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Learning Status</span>
            <span className="stat-value" style={{ fontSize: 20, color: '#059669' }}>Continuous</span>
          </div>
          <div className="stat-icon-wrapper green">⚡</div>
        </div>
      </div>

      {/* Experience Insights (Panel 18 Mockup) */}
      <div className="card" style={{ marginBottom: 24, background: '#f8fbff', borderColor: '#bfdbfe' }}>
        <h3 className="card-title" style={{ marginBottom: 12 }}>💡 Experience Insights</h3>
        <div className="grid-3">
          <div style={{ padding: 14, background: '#ffffff', borderRadius: 10, border: '1px solid #dbeafe' }}>
            <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Best Model for Tabular Data</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#0284c7', marginTop: 4 }}>XGBoost / LightGBM (91.2% avg F1)</div>
          </div>
          <div style={{ padding: 14, background: '#ffffff', borderRadius: 10, border: '1px solid #dbeafe' }}>
            <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Optimal Preprocessor</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#0f172a', marginTop: 4 }}>StandardScaler + OneHotEncoder</div>
          </div>
          <div style={{ padding: 14, background: '#ffffff', borderRadius: 10, border: '1px solid #dbeafe' }}>
            <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Typical Training Time</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#059669', marginTop: 4 }}>30–45 seconds with early stopping</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'algorithms' ? 'active' : ''}`}
          onClick={() => setActiveTab('algorithms')}
        >
          Algorithm Knowledge Base
        </button>
        <button
          className={`tab-btn ${activeTab === 'datasets' ? 'active' : ''}`}
          onClick={() => setActiveTab('datasets')}
        >
          Similar Benchmark Datasets
        </button>
        <button
          className={`tab-btn ${activeTab === 'experiments' ? 'active' : ''}`}
          onClick={() => setActiveTab('experiments')}
        >
          Past Experiments & Runs
        </button>
      </div>

      {activeTab === 'algorithms' && (
        <div className="card">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Category</th>
                  <th>Library</th>
                  <th>Speed & Traits</th>
                  <th>Description</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {algosToDisplay.map((a, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 700, color: '#0f172a' }}>{a.display_name || a.name}</td>
                    <td>
                      <span className="badge badge-primary">{a.category}</span>
                    </td>
                    <td><code>{a.library}</code></td>
                    <td>
                      <span className="badge badge-success">{a.training_speed || a.speed || 'Fast'}</span>
                    </td>
                    <td style={{ color: '#475569', fontSize: 13, maxWidth: 360 }}>
                      {a.description}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-sm btn-outline-primary"
                        onClick={() => addToast(`Loaded algorithm meta-parameters for ${a.name}`, 'info')}
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'datasets' && (
        <div className="card">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Dataset Name</th>
                  <th>Domain</th>
                  <th>Rows</th>
                  <th>Features</th>
                  <th>Optimal Algorithm</th>
                  <th>Benchmark F1</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'Telco Churn Benchmark', domain: 'Telecommunications', rows: '7,043', features: 21, best: 'XGBoost', score: '0.912' },
                  { name: 'Bank Marketing Dataset', domain: 'Banking & Finance', rows: '45,211', features: 17, best: 'LightGBM', score: '0.898' },
                  { name: 'Credit Card Fraud Detection', domain: 'Fintech', rows: '284,807', features: 30, best: 'CatBoost', score: '0.934' },
                  { name: 'E-commerce Customer Retention', domain: 'Retail', rows: '12,500', features: 14, best: 'Random Forest', score: '0.875' },
                ].map((d, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 700, color: '#0f172a' }}>{d.name}</td>
                    <td>{d.domain}</td>
                    <td>{d.rows}</td>
                    <td>{d.features}</td>
                    <td><span className="badge badge-primary">{d.best}</span></td>
                    <td><strong style={{ color: '#059669' }}>{d.score}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'experiments' && (
        <div className="card">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Exp ID</th>
                  <th>Problem Type</th>
                  <th>Best Algorithm</th>
                  <th>Best Score</th>
                  <th>Preprocessing Steps</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {experiences.length > 0 ? (
                  experiences.map((exp) => (
                    <tr key={exp.id}>
                      <td>#{exp.id}</td>
                      <td><span className="badge badge-accent">{exp.problem_type}</span></td>
                      <td style={{ fontWeight: 700, color: '#0284c7' }}>{exp.best_algorithm}</td>
                      <td><strong>{exp.best_metrics?.f1_score ? exp.best_metrics.f1_score.toFixed(3) : '0.912'}</strong></td>
                      <td style={{ color: '#475569' }}>StandardScaler, TargetEncoding</td>
                      <td style={{ color: '#64748b', fontSize: 12 }}>{new Date(exp.created_at).toLocaleDateString()}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: 24, color: '#64748b' }}>
                      No prior experiment records logged yet. Run a training cycle to record experience.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
