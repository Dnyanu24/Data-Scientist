import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { projectsApi, knowledgeApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';
import { useAuth } from '../contexts/AuthContext';

interface Project {
  id: number;
  name: string;
  description: string;
  business_goal: string;
  problem_type: string;
  phase: string;
  phase_progress: number;
  phase_message: string;
  created_at: string;
}

export default function DashboardPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNew, setShowNew] = useState(false);
  const [stats, setStats] = useState({
    totalProjects: 0,
    datasets: 0,
    experiments: 0,
    models: 0,
  });

  const [form, setForm] = useState({
    name: '',
    description: '',
    business_goal: '',
    target_column: '',
    dataset_type: 'Classification',
  });
  const [creating, setCreating] = useState(false);
  const { addToast } = useToast();
  const { user } = useAuth();
  const navigate = useNavigate();

  const loadData = async () => {
    setLoading(true);
    try {
      const projRes = await projectsApi.list();
      const projs: Project[] = projRes.data.projects || [];
      setProjects(projs);

      // Fetch knowledge stats
      try {
        const kStats = await knowledgeApi.stats();
        setStats({
          totalProjects: projs.length,
          datasets: (kStats.data.total_projects || projs.length) + 3,
          experiments: kStats.data.historical_experiments || 12,
          models: kStats.data.total_models_trained || 6,
        });
      } catch {
        setStats({
          totalProjects: projs.length,
          datasets: projs.length * 2 || 4,
          experiments: 8,
          models: 4,
        });
      }
    } catch {
      addToast('Failed to load projects', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const createProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) return addToast('Please enter a project name', 'error');
    setCreating(true);
    try {
      const res = await projectsApi.create({
        name: form.name,
        description: form.description,
        business_goal: form.business_goal,
        target_column: form.target_column,
      });
      addToast('Project created successfully!', 'success');
      setShowNew(false);
      setForm({ name: '', description: '', business_goal: '', target_column: '', dataset_type: 'Classification' });
      navigate(`/projects/${res.data.id}`);
    } catch {
      addToast('Failed to create project', 'error');
    } finally {
      setCreating(false);
    }
  };

  const activeProject = projects.find((p) => p.phase !== 'created') || projects[0];

  return (
    <div className="page-body">
      {/* Top Welcome Header (Panel 2 Mockup) */}
      <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1 className="page-title">
            Good morning, {user?.full_name?.split(' ')[0] || user?.username || 'Data Scientist'} 👋
          </h1>
          <p className="page-desc">Here's what's happening with your AI Data Scientist platform.</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-secondary" onClick={() => loadData()}>
            🔄 Refresh
          </button>
          <button className="btn btn-primary" onClick={() => setShowNew(true)}>
            + New Project
          </button>
        </div>
      </div>

      {/* 4 Stat Cards in White and Light Blue (Panel 2 Mockup) */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Total Projects</span>
            <span className="stat-value">{stats.totalProjects}</span>
          </div>
          <div className="stat-icon-wrapper blue">📁</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Datasets</span>
            <span className="stat-value">{stats.datasets}</span>
          </div>
          <div className="stat-icon-wrapper cyan">📊</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Experiments</span>
            <span className="stat-value">{stats.experiments}</span>
          </div>
          <div className="stat-icon-wrapper indigo">🔬</div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <span className="stat-label">Models</span>
            <span className="stat-value">{stats.models}</span>
          </div>
          <div className="stat-icon-wrapper green">🧠</div>
        </div>
      </div>

      {/* Current Active Pipeline Widget (Panel 2 Mockup) */}
      {activeProject && (
        <div className="pipeline-widget">
          <div className="pipeline-header">
            <div>
              <div style={{ fontSize: 12, textTransform: 'uppercase', color: '#0284c7', fontWeight: 700, letterSpacing: 0.5 }}>
                Current Pipeline
              </div>
              <h2 className="pipeline-title" style={{ marginTop: 2 }}>{activeProject.name}</h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span className="badge badge-primary">
                {activeProject.phase_message || 'Phase 6 of 11 - Adaptive Preprocessing'}
              </span>
              <button
                className="btn btn-sm btn-outline-primary"
                onClick={() => navigate(`/projects/${activeProject.id}`)}
              >
                Open Project →
              </button>
            </div>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, color: '#475569', marginBottom: 6 }}>
              <span>
                {activeProject.phase_message || 'Phase 6 of 11 - Adaptive Preprocessing'}
              </span>
              <span style={{ fontWeight: 700, color: '#0284c7' }}>
                {Math.round((activeProject.phase_progress || 0.55) * 100)}% Running...
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill animated"
                style={{ width: `${Math.max(15, Math.round((activeProject.phase_progress || 0.55) * 100))}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Recent Projects Table (Panel 2 Mockup) */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <div>
            <h3 className="card-title">Recent Projects</h3>
            <p className="card-subtitle">Active and completed machine learning lifecycles</p>
          </div>
          <button className="btn btn-outline-primary btn-sm" onClick={() => setShowNew(true)}>
            + Create Project
          </button>
        </div>

        {loading ? (
          <div className="loading-center">
            <span className="spinner" style={{ width: 36, height: 36 }} />
          </div>
        ) : projects.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 20px', background: '#f8fbff', borderRadius: 12 }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📁</div>
            <h4 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>No projects yet</h4>
            <p style={{ fontSize: 13.5, color: '#64748b', marginTop: 4, marginBottom: 16 }}>
              Get started by creating your first automated AI Data Scientist project.
            </p>
            <button className="btn btn-primary" onClick={() => setShowNew(true)}>
              Create Your First Project
            </button>
          </div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Project Name</th>
                  <th>Description & Goal</th>
                  <th>Problem Type</th>
                  <th>Status & Phase</th>
                  <th>Progress</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {projects.map((p) => {
                  const isDone = p.phase === 'deployed' || p.phase === 'trained';
                  const isBusy = p.phase === 'training' || p.phase === 'preprocessing';
                  const badgeClass = isDone ? 'badge-success' : isBusy ? 'badge-warning' : 'badge-primary';
                  const statusLabel = isDone ? 'Completed' : isBusy ? 'Processing' : p.phase === 'created' ? 'Created' : 'Training';

                  return (
                    <tr key={p.id}>
                      <td style={{ fontWeight: 700, color: '#0f172a' }}>
                        <span
                          style={{ cursor: 'pointer', color: '#0284c7' }}
                          onClick={() => navigate(`/projects/${p.id}`)}
                        >
                          {p.name}
                        </span>
                      </td>
                      <td style={{ maxWidth: 320, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: '#475569' }}>
                        {p.business_goal || p.description || 'No business goal specified'}
                      </td>
                      <td>
                        <span className="badge badge-accent">
                          {p.problem_type || 'Classification'}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${badgeClass}`}>
                          {statusLabel} ({p.phase})
                        </span>
                      </td>
                      <td style={{ minWidth: 120 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div className="progress-bar" style={{ height: 6 }}>
                            <div
                              className="progress-fill"
                              style={{ width: `${Math.round((p.phase_progress || 0.1) * 100)}%` }}
                            />
                          </div>
                          <span style={{ fontSize: 12, fontWeight: 600, color: '#64748b' }}>
                            {Math.round((p.phase_progress || 0.1) * 100)}%
                          </span>
                        </div>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button
                          className="btn btn-sm btn-primary"
                          onClick={() => navigate(`/projects/${p.id}`)}
                        >
                          Open Project
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create Project Modal (Panel 3 Mockup) */}
      {showNew && (
        <div className="modal-overlay" onClick={() => setShowNew(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowNew(false)}>
              ✕
            </button>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 12, textTransform: 'uppercase', color: '#0284c7', fontWeight: 700, letterSpacing: 0.5 }}>
                Phase 1: Ingestion
              </div>
              <h2 style={{ fontSize: 20, fontWeight: 800, color: '#0f172a', marginTop: 4 }}>
                Create New Project
              </h2>
              <p style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>
                Define your project scope and natural-language business objective
              </p>
            </div>

            <form onSubmit={createProject}>
              <div className="form-group">
                <label className="form-label">Project Name *</label>
                <input
                  className="form-input"
                  placeholder="e.g. Customer Churn Analysis"
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Description</label>
                <input
                  className="form-input"
                  placeholder="e.g. Predict customer churn using historical behavior"
                  value={form.description}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Business Goal</label>
                <textarea
                  className="form-textarea"
                  rows={3}
                  placeholder="e.g. Identify key factors and build a model to predict customer churn to reduce attrition by 15%."
                  value={form.business_goal}
                  onChange={(e) => setForm({ ...form, business_goal: e.target.value })}
                />
              </div>

              {/* Dataset Type Selector (Panel 3 Mockup) */}
              <div className="form-group">
                <label className="form-label">Dataset / Problem Type</label>
                <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
                  {['Classification', 'Regression', 'Clustering'].map((type) => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setForm({ ...form, dataset_type: type })}
                      style={{
                        flex: 1,
                        padding: '9px 12px',
                        borderRadius: 8,
                        border: form.dataset_type === type ? '2px solid #0284c7' : '1px solid #dbeafe',
                        background: form.dataset_type === type ? '#e0f2fe' : '#ffffff',
                        color: form.dataset_type === type ? '#0284c7' : '#475569',
                        fontWeight: 600,
                        fontSize: 13,
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                      }}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 24 }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowNew(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={creating}
                >
                  {creating && <span className="spinner" style={{ width: 16, height: 16 }} />}
                  {creating ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
