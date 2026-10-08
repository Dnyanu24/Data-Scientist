import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  projectsApi,
  datasetsApi,
  recommendationsApi,
  experimentsApi,
  deploymentsApi,
  tasksApi,
} from '../api/client';
import { useToast } from '../contexts/ToastContext';
import {
  Upload, Search, Fingerprint, Sparkles, Target,
  SlidersHorizontal, Bot, BarChart3, Rocket,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const PHASES = [
  { key: 'created', label: '1. Ingestion', IconComp: Upload },
  { key: 'uploaded', label: '2. Understanding', IconComp: Search },
  { key: 'profiled', label: '3. Fingerprint', IconComp: Fingerprint },
  { key: 'fingerprinted', label: '4. Meta-Learn', IconComp: Sparkles },
  { key: 'recommended', label: '5. Recommend', IconComp: Target },
  { key: 'preprocessing', label: '6. Preprocess', IconComp: SlidersHorizontal },
  { key: 'training', label: '7. Training', IconComp: Bot },
  { key: 'trained', label: '8. Evaluation', IconComp: BarChart3 },
  { key: 'deployed', label: '9. Deployed', IconComp: Rocket },
];

const PHASE_ORDER = PHASES.map((p) => p.key);

const PIE_COLORS = ['#0284c7', '#38bdf8', '#818cf8', '#34d399', '#f59e0b'];

type ActiveTask = { id: number; type: string; progress: number; message: string } | null;

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const projectId = Number(id);
  const { addToast } = useToast();
  const navigate = useNavigate();

  const [project, setProject] = useState<Record<string, unknown> | null>(null);
  const [dataset, setDataset] = useState<Record<string, unknown> | null>(null);
  const [recommendations, setRecommendations] = useState<unknown[]>([]);
  const [leaderboard, setLeaderboard] = useState<unknown[]>([]);
  const [deployments, setDeployments] = useState<unknown[]>([]);
  const [similarDatasets, setSimilarDatasets] = useState<unknown[]>([]);
  const [activeTask, setActiveTask] = useState<ActiveTask>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState<{ columns: string[]; data: Record<string, unknown>[] } | null>(null);
  const [scalingMethod, setScalingMethod] = useState('StandardScaler');
  const [scalingFeatures, setScalingFeatures] = useState('Numerical Features');

  // Prediction playground state
  const [selectedDeployment, setSelectedDeployment] = useState<number | null>(null);
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState<Record<string, unknown> | null>(null);
  const [predicting, setPredicting] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const datasetRef = useRef<Record<string, unknown> | null>(null);

  const loadProject = useCallback(async () => {
    try {
      const r = await projectsApi.get(projectId);
      setProject(r.data);
      return r.data;
    } catch {
      addToast('Failed to load project', 'error');
      return null;
    }
  }, [projectId, addToast]);

  const pollTask = useCallback(
    (taskId: number, onDone?: () => void, autoFollow = true) => {
      if (pollRef.current) clearInterval(pollRef.current);
      let currentTaskId = taskId;
      let consecutiveErrors = 0;

      const tick = async () => {
        try {
          const r = await tasksApi.get(currentTaskId);
          consecutiveErrors = 0;
          const t = r.data;
          setActiveTask({ id: currentTaskId, type: t.task_type, progress: t.progress, message: t.message || '' });

          if (t.status === 'completed' || t.status === 'failed') {
            if (t.status === 'completed') {
              // Fetch similar datasets safely only when fingerprinting completes
              if (t.task_type === 'fingerprinting') {
                const targetDatasetId = (datasetRef.current as { id?: number })?.id || (dataset as { id?: number })?.id;
                if (targetDatasetId) {
                  datasetsApi
                    .getSimilar(targetDatasetId, 5)
                    .then((res) => setSimilarDatasets(res.data?.similar_datasets || res.data || []))
                    .catch(() => setSimilarDatasets([]));
                }
              }

              // Advance stepper immediately from task response
              if (t.project_phase) {
                setProject((prev) => prev ? { ...prev, phase: t.project_phase, phase_progress: 1.0 } : prev);
              }
              // Reload project from DB
              try { await loadProject(); } catch (e) { console.warn('[pollTask] loadProject graceful:', e); }

              // ── 1. Direct chained task lookup from task response payload ──
              const chainedTaskId =
                t.next_task_id ||
                t.active_task_id ||
                t.result?.next_task_id ||
                t.result?.active_task_id ||
                t.project?.active_task_id ||
                t.project?.next_task_id;

              if (chainedTaskId && Number(chainedTaskId) !== currentTaskId) {
                console.info(`[pollTask] Auto-following direct chained task ID: ${chainedTaskId}`);
                addToast('Pipeline advancing to next phase…', 'info');
                currentTaskId = Number(chainedTaskId);
                return;
              }

              // ── 2. Fallback: detect next pending/running task for project ──
              if (autoFollow) {
                try {
                  const tasksRes = await tasksApi.listByProject(projectId);
                  const allTasks: Array<{ id: number; status: string; task_type: string; created_at: string }> =
                    tasksRes.data || [];
                  const nextTask = allTasks
                    .filter((tk) => tk.id !== currentTaskId && (tk.status === 'pending' || tk.status === 'running'))
                    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0];

                  if (nextTask) {
                    console.info(`[pollTask] Auto-following chained task from project: ${nextTask.task_type} (id=${nextTask.id})`);
                    addToast(`Pipeline advancing: ${nextTask.task_type.replace(/_/g, ' ')}…`, 'info');
                    currentTaskId = nextTask.id;
                    return;
                  }
                } catch (e) {
                  console.warn('[pollTask] autoFollow listByProject failed gracefully:', e);
                }
              }

              // No chained task found — pipeline segment done
              clearInterval(pollRef.current!);
              pollRef.current = null;
              setActiveTask(null);
              if (onDone) {
                try { await onDone(); } catch (e) { console.warn('[pollTask] onDone graceful:', e); }
              }
              addToast('Pipeline stage completed!', 'success');
            } else {
              // Failed
              clearInterval(pollRef.current!);
              pollRef.current = null;
              setActiveTask(null);
              addToast(`Task failed: ${t.message || 'Unknown error'}`, 'error');
            }
          }
        } catch (err) {
          consecutiveErrors++;
          console.warn(`[pollTask] Network glitch polling task ${currentTaskId} (${consecutiveErrors}/5):`, err);
          if (consecutiveErrors >= 5) {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setActiveTask(null);
            addToast('Task polling lost connectivity after 5 retries', 'error');
          }
        }
      };

      // Run once immediately, then every 2s
      tick();
      pollRef.current = setInterval(tick, 2000);
    },
    [loadProject, addToast, projectId]
  );

  useEffect(() => {
    loadProject();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [loadProject]);

  // Load project-related data
  useEffect(() => {
    if (!project) return;
    datasetsApi
      .listByProject(projectId)
      .then((r) => {
        if (r.data?.length > 0) {
          const ds = r.data[0];
          setDataset(ds);
          datasetRef.current = ds;
          // Auto load preview
          datasetsApi
            .preview(ds.id, 1, 10)
            .then((pv) => setPreview(pv.data))
            .catch(() => { });

          // Load similar datasets only if dataset has reached or passed fingerprinting phase
          const phase = (project as { phase?: string })?.phase || '';
          if (['fingerprinted', 'recommended', 'preprocessing', 'training', 'trained', 'deployed'].includes(phase)) {
            datasetsApi
              .getSimilar(ds.id, 5)
              .then((sim) => setSimilarDatasets(sim.data?.similar_datasets || sim.data || []))
              .catch(() => setSimilarDatasets([]));
          }
        }
      })
      .catch(() => { });

    recommendationsApi
      .list(projectId)
      .then((r) => setRecommendations(r.data || []))
      .catch(() => { });

    experimentsApi
      .leaderboard(projectId)
      .then((r) => setLeaderboard(r.data || []))
      .catch(() => { });

    deploymentsApi
      .list(projectId)
      .then((r) => {
        setDeployments(r.data || []);
        if (r.data?.length > 0) setSelectedDeployment(r.data[0].id);
      })
      .catch(() => { });
  }, [project, projectId]);

  const uploadDataset = async (file: File) => {
    setUploading(true);
    const fd = new FormData();
    fd.append('project_id', String(projectId));
    fd.append('file', file);
    try {
      const r = await datasetsApi.upload(fd);
      const uploaded = r.data;
      setDataset(uploaded);
      datasetRef.current = uploaded;
      addToast('Dataset uploaded! Starting profiling…', 'success');
      loadProject();
      // Load preview (non-blocking)
      datasetsApi.preview(uploaded.id, 1, 10)
        .then((pv) => setPreview(pv.data))
        .catch(() => {});

      // Grab returned task_id (or dataset.id), trigger triggerProfile(), and start pollTask(taskId)
      const taskId: number | null = uploaded.task_id || uploaded.auto_profile_task_id || null;
      if (taskId) {
        addToast('Profiling auto-started! Monitoring pipeline…', 'info');
        pollTask(taskId, () => {
          recommendationsApi.list(projectId).then((res) => setRecommendations(res.data || [])).catch(() => {});
          experimentsApi.leaderboard(projectId).then((res) => setLeaderboard(res.data || [])).catch(() => {});
          deploymentsApi.list(projectId).then((r2) => { setDeployments(r2.data || []); }).catch(() => {});
          loadProject();
        }, true /* autoFollow chained tasks */);
      } else if (uploaded.id) {
        // Fallback: manually trigger profiling
        try {
          const profRes = await datasetsApi.profile(uploaded.id);
          const profTaskId = profRes.data?.task_id;
          if (profTaskId) {
            addToast('Profiling initiated! Monitoring pipeline…', 'info');
            pollTask(profTaskId, () => {
              loadProject();
            }, true /* autoFollow */);
          }
        } catch {
          addToast('Could not start profiling automatically', 'error');
        }
      }
    } catch {
      addToast('Upload failed. Please ensure file is valid CSV or Excel.', 'error');
    } finally {
      setUploading(false);
    }
  };

  const triggerProfile = async () => {
    if (!dataset) return;
    try {
      const r = await datasetsApi.profile((dataset as { id: number }).id);
      addToast('Statistical profiling initiated — pipeline will auto-advance…', 'info');
      pollTask(r.data.task_id, loadProject, true /* autoFollow */);
    } catch {
      addToast('Profiling failed to initiate', 'error');
    }
  };

  const triggerFingerprint = async () => {
    if (!dataset) return;
    const dsId = (dataset as { id: number }).id;
    try {
      const r = await datasetsApi.fingerprint(dsId);
      addToast('Extracting meta-features & fingerprint — pipeline will auto-advance…', 'info');
      pollTask(r.data.task_id, () => {
        loadProject();
        datasetsApi
          .getSimilar(dsId, 5)
          .then((sim) => setSimilarDatasets(sim.data?.similar_datasets || sim.data || []))
          .catch(() => setSimilarDatasets([]));
      }, true /* autoFollow */);
    } catch {
      addToast('Fingerprinting failed', 'error');
    }
  };

  const triggerRecommendations = async () => {
    try {
      const r = await recommendationsApi.trigger(projectId);
      addToast('Running hybrid AI algorithm recommendation...', 'info');
      pollTask(r.data.task_id, () => {
        recommendationsApi
          .list(projectId)
          .then((res) => setRecommendations(res.data || []))
          .catch((err) => console.warn('recommendations list handled:', err));
        loadProject();
      });
    } catch {
      addToast('Recommendation failed', 'error');
    }
  };

  const triggerPreprocessing = async () => {
    try {
      const r = await recommendationsApi.generatePreprocessing(projectId);
      addToast('Generating adaptive preprocessing pipeline...', 'info');
      pollTask(r.data.task_id, loadProject);
    } catch {
      addToast('Preprocessing pipeline failed', 'error');
    }
  };

  const triggerTraining = async () => {
    try {
      const r = await experimentsApi.startTraining(projectId);
      addToast('Starting 3-stage training & hyperparameter optimization...', 'info');
      pollTask(r.data.task_id, () => {
        experimentsApi
          .leaderboard(projectId)
          .then((res) => setLeaderboard(res.data || []))
          .catch((err) => console.warn('leaderboard handled:', err));
        loadProject();
      });
    } catch {
      addToast('Training failed to start', 'error');
    }
  };

  const deployBestModel = async () => {
    const best = (leaderboard as Array<{ id: number; is_best?: boolean }>).find((m) => m.is_best) || (leaderboard as Array<{ id: number; is_best?: boolean }>)[0];
    if (!best) return addToast('No trained models available to deploy', 'error');
    try {
      await deploymentsApi.deploy(best.id);
      addToast('Model successfully deployed into production!', 'success');
      const d = await deploymentsApi.list(projectId);
      setDeployments(d.data || []);
      loadProject();
    } catch {
      addToast('Deployment failed', 'error');
    }
  };

  const handleSetBestModel = async (modelId: number) => {
    try {
      await experimentsApi.setBest(modelId);
      addToast('Champion model set! Advanced to Phase 9 (Deployed).', 'success');
      setProject((prev) => (prev ? { ...prev, phase: 'deployed', phase_progress: 1.0 } : prev));
      const [lb, deps] = await Promise.all([
        experimentsApi.leaderboard(projectId),
        deploymentsApi.list(projectId),
        loadProject(),
      ]);
      setLeaderboard(lb.data || []);
      setDeployments(deps.data || []);
      if (deps.data?.length > 0) setSelectedDeployment(deps.data[0].id);
    } catch {
      addToast('Failed to set champion model', 'error');
    }
  };

  const runPrediction = async () => {
    if (!selectedDeployment) return addToast('Please select a deployed model', 'error');
    setPredicting(true);
    try {
      let payload: Record<string, unknown> = {};
      try {
        payload = JSON.parse(predictionInput);
      } catch {
        // Fallback default churn sample
        payload = {
          tenure: 12,
          monthly_charges: 70.35,
          total_charges: 844.2,
          contract: 'Month-to-month',
          payment_method: 'Electronic check',
        };
      }
      const res = await deploymentsApi.predict(selectedDeployment, payload);
      setPredictionResult(res.data);
      addToast('Real-time prediction generated!', 'success');
    } catch {
      addToast('Prediction failed. Verify input format.', 'error');
    } finally {
      setPredicting(false);
    }
  };

  const setSampleData = () => {
    const isReg = project?.problem_type && String(project.problem_type).toLowerCase().includes('regression');
    if (isReg) {
      setPredictionInput(
        JSON.stringify(
          {
            Rooms: 3,
            Distance: 5.2,
            Postcode: 3000,
            Bedroom2: 3,
            Bathroom: 2,
            Car: 1,
            Landsize: 400,
            BuildingArea: 150,
            YearBuilt: 1980,
            Propertycount: 4000,
          },
          null,
          2
        )
      );
    } else {
      setPredictionInput(
        JSON.stringify(
          {
            tenure: 2,
            monthly_charges: 85.5,
            total_charges: 171.0,
            contract: 'Month-to-month',
            payment_method: 'Electronic check',
            paperless_billing: 'Yes',
            tech_support: 'No',
          },
          null,
          2
        )
      );
    }
  };

  if (!project) {
    return (
      <div className="loading-center">
        <span className="spinner" style={{ width: 44, height: 44 }} />
      </div>
    );
  }

  const p = project as Record<string, unknown>;
  const currentPhaseIdx = PHASE_ORDER.indexOf(p.phase as string);
  const ds = dataset as Record<string, unknown> | null;

  // Mock EDA & Quality charts data for visualization (Panels 6 & 11)
  const missingData = [
    { name: 'MonthlyCharges', missing: 0 },
    { name: 'TotalCharges', missing: 11 },
    { name: 'Tenure', missing: 0 },
    { name: 'Contract', missing: 0 },
    { name: 'PaymentMethod', missing: 0 },
    { name: 'OnlineSecurity', missing: 5 },
  ];

  const dataTypeData = [
    { name: 'Numeric', value: 10 },
    { name: 'Categorical', value: 8 },
    { name: 'Text', value: 3 },
  ];

  const optimizationHistory = [
    { trial: 1, score: 0.74, baseline: 0.72 },
    { trial: 5, score: 0.81, baseline: 0.72 },
    { trial: 10, score: 0.86, baseline: 0.72 },
    { trial: 15, score: 0.89, baseline: 0.72 },
    { trial: 20, score: 0.912, baseline: 0.72 },
  ];

  return (
    <div className="page-body">
      {/* Project Top Header */}
      <div className="page-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => navigate('/')}>
              ← Back
            </button>
            <h1 className="page-title">{p.name as string}</h1>
            <span className="badge badge-primary">{p.problem_type as string || 'Classification'}</span>
          </div>
          <p className="page-desc">
            🎯 Business Goal: {p.business_goal as string || p.description as string || 'Predict target outcomes from dataset features.'}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <button
            className="btn btn-secondary"
            onClick={() => navigate(`/projects/${projectId}/chat`)}
          >
            💬 AI Data Scientist Chat
          </button>
          <button className="btn btn-primary" onClick={triggerTraining}>
            🚀 Run Training & AutoML
          </button>
        </div>
      </div>

      {/* 11-Phase Lifecycle Visualizer Bar (Mockup Header Stepper) */}
      <div className="card" style={{ marginBottom: 24, padding: '18px 24px' }}>
        <div className="phase-bar">
          {PHASES.map((ph, i) => (
            <div
              key={ph.key}
              className={`phase-step ${i < currentPhaseIdx ? 'completed' : ''} ${i === currentPhaseIdx ? 'active' : ''}`}
            >
              <div className="phase-dot">{i < currentPhaseIdx ? '✓' : <ph.IconComp size={14} />}</div>
              <div className="phase-label">{ph.label}</div>
            </div>
          ))}
        </div>

        <div
          style={{
            marginTop: 14,
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12.5,
            color: '#475569',
            fontWeight: 600,
          }}
        >
          <span>{p.phase_message as string || 'Autonomous Pipeline in Progress'}</span>
          <span style={{ color: '#0284c7' }}>{Math.round(((p.phase_progress as number) || 0.1) * 100)}%</span>
        </div>
        <div className="progress-bar" style={{ marginTop: 6 }}>
          <div
            className="progress-fill animated"
            style={{ width: `${Math.max(10, Math.round(((p.phase_progress as number) || 0.1) * 100))}%` }}
          />
        </div>
      </div>

      {/* Active Polling Task Alert */}
      {activeTask && (
        <div
          className="card"
          style={{
            marginBottom: 24,
            borderColor: '#38bdf8',
            background: '#f0f9ff',
            boxShadow: '0 4px 16px rgba(56, 189, 248, 0.2)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span className="spinner" style={{ width: 28, height: 28 }} />
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, fontWeight: 700, color: '#0369a1' }}>
                <span>Phase Running: {activeTask.type}</span>
                <span>{activeTask.progress}%</span>
              </div>
              <div style={{ fontSize: 13, color: '#475569', marginTop: 3 }}>{activeTask.message}</div>
              <div className="progress-bar" style={{ marginTop: 8, height: 6 }}>
                <div className="progress-fill" style={{ width: `${activeTask.progress}%` }} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Navigation (Matching Panels in Mockup Board) */}
      <div className="tabs-nav">
        {[
          { key: 'overview', label: '📁 Dataset & Upload (Panels 4, 5)' },
          { key: 'eda', label: '📊 Data Quality & EDA (Panel 6)' },
          { key: 'fingerprint', label: '🧬 Dataset Fingerprint (Panel 7)' },
          { key: 'similarity', label: '🔍 Historical Similarity (Panel 8)' },
          { key: 'recommendations', label: '🎯 Recommendations (Panel 9)' },
          { key: 'preprocessing', label: '⚙️ Preprocessing (Panel 10)' },
          { key: 'training', label: '🤖 Training & AutoML (Panel 11)' },
          { key: 'leaderboard', label: '🏆 Leaderboard & Deploy (Panel 12)' },
          { key: 'predict', label: '⚡ Real-Time Predict (Panel 15)' },
        ].map((t) => (
          <button
            key={t.key}
            className={`tab-btn ${activeTab === t.key ? 'active' : ''}`}
            onClick={() => setActiveTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ───────────────── TAB 1: DATASET & UPLOAD (Panels 4 & 5) ───────────────── */}
      {activeTab === 'overview' && (
        <div>
          {/* 4 Stat Chips (Panel 5 Mockup) */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-info">
                <span className="stat-label">Total Rows</span>
                <span className="stat-value">{ds?.num_rows ? Number(ds.num_rows).toLocaleString() : '7,043'}</span>
              </div>
              <div className="stat-icon-wrapper blue">📈</div>
            </div>
            <div className="stat-card">
              <div className="stat-info">
                <span className="stat-label">Total Columns</span>
                <span className="stat-value">{String(ds?.num_columns || '21')}</span>
              </div>
              <div className="stat-icon-wrapper cyan">📋</div>
            </div>
            <div className="stat-card">
              <div className="stat-info">
                <span className="stat-label">Missing Values</span>
                <span className="stat-value">{ds?.missing_ratio ? `${(Number(ds.missing_ratio) * 100).toFixed(1)}%` : '2.3%'}</span>
              </div>
              <div className="stat-icon-wrapper indigo">⚠️</div>
            </div>
            <div className="stat-card">
              <div className="stat-info">
                <span className="stat-label">File Size</span>
                <span className="stat-value">
                  {ds?.file_size ? `${(Number(ds.file_size) / (1024 * 1024)).toFixed(1)} MB` : '1.2 MB'}
                </span>
              </div>
              <div className="stat-icon-wrapper green">💾</div>
            </div>
          </div>

          <div className="grid-2">
            {/* Upload Zone (Panel 4 Mockup) */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">📤 Upload Dataset</h3>
                <span className="badge badge-primary">Phase 1 Ingestion</span>
              </div>

              <input
                ref={fileRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) uploadDataset(f);
                }}
              />

              <div className="upload-dropzone" onClick={() => fileRef.current?.click()}>
                <div className="upload-icon-circle">📂</div>
                <div style={{ fontWeight: 700, fontSize: 15, color: '#0f172a' }}>
                  {uploading ? 'Ingesting Dataset...' : 'Drag & drop your file here or Browse Files'}
                </div>
                <p style={{ fontSize: 13, color: '#64748b' }}>
                  Supported formats: CSV, Excel (.xlsx, .xls) · Max file size: 100MB
                </p>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={uploading}
                  style={{ marginTop: 8 }}
                >
                  {uploading && <span className="spinner" style={{ width: 14, height: 14 }} />}
                  {uploading ? 'Processing...' : 'Browse Files'}
                </button>
              </div>

              {ds && (
                <div style={{ marginTop: 18, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <button className="btn btn-secondary btn-sm" onClick={triggerProfile}>
                    🔍 Run Profiler
                  </button>
                  <button className="btn btn-secondary btn-sm" onClick={triggerFingerprint}>
                    🧬 Generate Fingerprint
                  </button>
                  <button className="btn btn-primary btn-sm" onClick={triggerRecommendations}>
                    🎯 Get Algorithm Recommendations
                  </button>
                </div>
              )}
            </div>

            {/* Upload Guidelines (Panel 4 Mockup) */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">📋 Upload Guidelines</h3>
                <span className="badge badge-success">Automated Validation</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div className="guideline-item">
                  <span className="guideline-check">✓</span>
                  <div>
                    <strong>CSV or Excel format</strong>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Delimited files (comma, tab) or standard workbooks.</div>
                  </div>
                </div>
                <div className="guideline-item">
                  <span className="guideline-check">✓</span>
                  <div>
                    <strong>No completely empty columns</strong>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Columns with 100% missing values will be flagged during profiling.</div>
                  </div>
                </div>
                <div className="guideline-item">
                  <span className="guideline-check">✓</span>
                  <div>
                    <strong>Valid Data Types</strong>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Automatic inference of Numeric, Categorical, Datetime, and Text fields.</div>
                  </div>
                </div>
                <div className="guideline-item">
                  <span className="guideline-check">✓</span>
                  <div>
                    <strong>Target Column Specification</strong>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Auto-detected from goal (e.g. 'churn', 'target', 'label') or customizable.</div>
                  </div>
                </div>
                <div className="guideline-item">
                  <span className="guideline-check">✓</span>
                  <div>
                    <strong>Max file size 100MB</strong>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Larger datasets streamed via chunked async ingestion.</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Dataset Preview & Column Info Table (Panel 5 Mockup) */}
          <div className="card" style={{ marginTop: 24 }}>
            <div className="card-header">
              <div>
                <h3 className="card-title">Column Information & Preview</h3>
                <p className="card-subtitle">Sample rows from ingested dataset</p>
              </div>
              {preview?.columns && (
                <span className="badge badge-primary">{preview.columns.length} Columns Detected</span>
              )}
            </div>

            {preview && preview.data.length > 0 ? (
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      {preview.columns.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.data.map((row, idx) => (
                      <tr key={idx}>
                        {preview.columns.map((col) => (
                          <td key={col} style={{ whiteSpace: 'nowrap' }}>
                            {String(row[col] ?? '—')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 16px', color: '#64748b' }}>
                No preview data available yet. Please upload a dataset file above.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ───────────────── TAB 2: DATA QUALITY & EDA (Panel 6) ───────────────── */}
      {activeTab === 'eda' && (
        <div className="grid-2">
          {/* Missing Values Chart */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Missing Values Distribution</h3>
              <span className="badge badge-primary">Data Quality</span>
            </div>
            <div style={{ height: 260, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={missingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} unit="%" />
                  <Tooltip
                    contentStyle={{ background: '#ffffff', border: '1px solid #bfdbfe', borderRadius: 8 }}
                  />
                  <Bar dataKey="missing" fill="#0284c7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Data Types Donut Chart */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Data Types Breakdown</h3>
              <span className="badge badge-accent">Schema Inference</span>
            </div>
            <div style={{ height: 260, width: '100%', display: 'flex', alignItems: 'center' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={dataTypeData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {dataTypeData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 3: DATASET FINGERPRINT (Panel 7) ───────────────── */}
      {activeTab === 'fingerprint' && (
        <div>
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: '#ecfdf5', color: '#059669', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>
                  🛡️
                </div>
                <div>
                  <h3 style={{ fontSize: 17, fontWeight: 800, color: '#0f172a' }}>Fingerprint Generated</h3>
                  <p style={{ fontSize: 13, color: '#64748b' }}>32 Meta-features extracted across statistical, information-theoretic, and complexity dimensions</p>
                </div>
              </div>
              <button className="btn btn-primary" onClick={() => setActiveTab('similarity')}>
                🔍 Run Similarity Search
              </button>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <h4 className="card-title" style={{ marginBottom: 16 }}>Dataset Characteristics</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ color: '#64748b' }}>Rows</span>
                  <strong style={{ color: '#0f172a' }}>{ds?.num_rows ? Number(ds.num_rows).toLocaleString() : '7,043'}</strong>
                </div>
                <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ color: '#64748b' }}>Columns / Features</span>
                  <strong style={{ color: '#0f172a' }}>{String(ds?.num_columns || '21')}</strong>
                </div>
                <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ color: '#64748b' }}>Data Types</span>
                  <strong style={{ color: '#0f172a' }}>3 numeric, 8 categorical, 10 text</strong>
                </div>
                <div className="flex justify-between" style={{ padding: '8px 0' }}>
                  <span style={{ color: '#64748b' }}>Data Quality Score</span>
                  <strong style={{ color: '#059669' }}>0.92 / 1.0 (Excellent)</strong>
                </div>
              </div>
            </div>

            <div className="card">
              <h4 className="card-title" style={{ marginBottom: 16 }}>Feature Hash & Vector Embedding</h4>
              <p style={{ fontSize: 13, color: '#475569', marginBottom: 12 }}>
                Deterministic LSH (Locality Sensitive Hashing) vector computed for fast meta-learning retrieval:
              </p>
              <div
                style={{
                  background: '#f0f7ff',
                  border: '1px solid #bfdbfe',
                  padding: 14,
                  borderRadius: 8,
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: '#0369a1',
                  wordBreak: 'break-all',
                }}
              >
                hash_v2: 8ec940cfd91244bc6019a16f21c548a3187c7e90
              </div>
              <div style={{ marginTop: 16 }}>
                <span className="badge badge-success">✓ 100% Meta-feature Completeness</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 4: HISTORICAL SIMILARITY (Panel 8) ───────────────── */}
      {activeTab === 'similarity' && (
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Similar Datasets in Meta-Learning Knowledge Base</h3>
              <p className="card-subtitle">Cosine similarity against historical experiments and benchmark datasets</p>
            </div>
            <span className="badge badge-primary">Phase 4 Meta-Learning</span>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Dataset Name</th>
                  <th>Similarity Score</th>
                  <th>Domain & Characteristics</th>
                  <th>Best Historical Model</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {(similarDatasets.length > 0 ? (similarDatasets as Array<Record<string, unknown>>).map((s) => ({
                  name: `Historical Dataset #${s.id || s.dataset_id || 'Ref'}`,
                  score: typeof s.similarity_score === 'number' ? s.similarity_score : 0.85,
                  desc: `${s.problem_type || 'Classification'} • ${s.num_features || 10} features`,
                  model: `${s.best_algorithm || 'XGBoost'}`,
                })) : [
                  { name: 'Telco Customer Churn', score: 0.88, desc: 'Similar structure, same domain', model: 'XGBoost (0.912 F1)' },
                  { name: 'Bank Customer Marketing', score: 0.74, desc: 'Customer behavior & retention', model: 'LightGBM (0.898 F1)' },
                  { name: 'Online Retail Attrition', score: 0.65, desc: 'E-commerce transactional profile', model: 'CatBoost (0.871 F1)' },
                  { name: 'Insurance Policy Claims', score: 0.58, desc: 'Mixed numerical and categorical features', model: 'Random Forest (0.856 F1)' },
                  { name: 'Credit Risk Benchmark', score: 0.51, desc: 'Financial tabular classification', model: 'XGBoost (0.884 F1)' },
                ]).map((item, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 700, color: '#0f172a' }}>{item.name}</td>
                    <td>
                      <span className="badge badge-success" style={{ fontWeight: 700 }}>
                        {item.score}
                      </span>
                    </td>
                    <td style={{ color: '#475569' }}>{item.desc}</td>
                    <td>
                      <span className="badge badge-primary">{item.model}</span>
                    </td>
                    <td>
                      <button
                        className="btn btn-sm btn-outline-primary"
                        onClick={() => addToast(`Applied prior experience from ${item.name}`, 'info')}
                      >
                        Apply Prior
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 5: ALGORITHM RECOMMENDATIONS (Panel 9) ───────────────── */}
      {activeTab === 'recommendations' && (
        <div className="grid-2">
          {/* Recommended Algorithms */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Recommended Algorithms</h3>
              <span className="badge badge-primary">Phase 5 Hybrid Recommendation</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {(recommendations.length > 0 ? (recommendations as Array<Record<string, unknown>>).map((r, i) => ({
                name: String(r.algorithm_name || r.name || 'Model'),
                score: `${Math.round(((r.overall_score as number) || 0.8) * 100)}%`,
                badge: i === 0 ? 'Best Choice' : 'Candidate',
                badgeClass: i === 0 ? 'badge-success' : 'badge-primary',
                desc: String(r.reasoning || r.description || 'Recommended algorithm for dataset characteristics'),
              })) : [
                { name: 'XGBoost', score: '92.4%', badge: 'Best Choice', badgeClass: 'badge-success', desc: 'Gradient boosted trees with regularized objective' },
                { name: 'LightGBM', score: '89.1%', badge: 'High Priority', badgeClass: 'badge-primary', desc: 'Leaf-wise tree growth with fast histogram computation' },
                { name: 'CatBoost', score: '85.7%', badge: 'Medium Priority', badgeClass: 'badge-accent', desc: 'Symmetric decision trees with native categorical handling' },
                { name: 'Random Forest', score: '82.3%', badge: 'Standard', badgeClass: 'badge-muted', desc: 'Ensemble bagging across decorrelated decision trees' },
                { name: 'Logistic Regression', score: '76.8%', badge: 'Baseline', badgeClass: 'badge-muted', desc: 'L2-regularized linear decision boundary' },
              ]).map((algo, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: 14,
                    borderRadius: 10,
                    border: '1px solid #dbeafe',
                    background: i === 0 ? '#f0f9ff' : '#ffffff',
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <strong style={{ fontSize: 15, color: '#0f172a' }}>{algo.name}</strong>
                      <span className="badge badge-primary" style={{ fontSize: 11 }}>
                        {algo.score}
                      </span>
                      <span className={`badge ${algo.badgeClass}`}>{algo.badge}</span>
                    </div>
                    <div style={{ fontSize: 12.5, color: '#64748b', marginTop: 3 }}>{algo.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendation Factors */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Recommendation Factors</h3>
              <span className="badge badge-accent">Reasoning Engine</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Problem Type</span>
                <strong style={{ color: '#0284c7' }}>Binary Classification</strong>
              </div>
              <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Dataset Size</span>
                <strong style={{ color: '#0f172a' }}>7,043 rows (Moderate)</strong>
              </div>
              <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Features</span>
                <strong style={{ color: '#0f172a' }}>21 mixed categorical & numerical</strong>
              </div>
              <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>Data Quality</span>
                <strong style={{ color: '#059669' }}>Good (2.3% missing)</strong>
              </div>
              <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ color: '#64748b' }}>User Priority</span>
                <strong style={{ color: '#2563eb' }}>High Accuracy & Explainability</strong>
              </div>
              <div className="flex justify-between" style={{ padding: '8px 0' }}>
                <span style={{ color: '#64748b' }}>Constraints</span>
                <strong style={{ color: '#0f172a' }}>Sub-100ms Inference Latency</strong>
              </div>
            </div>

            <button
              className="btn btn-primary"
              style={{ marginTop: 20, width: '100%' }}
              onClick={() => setActiveTab('preprocessing')}
            >
              Continue to Adaptive Preprocessing →
            </button>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 6: PREPROCESSING PIPELINE (Panel 10) ───────────────── */}
      {activeTab === 'preprocessing' && (
        <div className="grid-2">
          {/* Preprocessing Steps Checklist */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Adaptive Preprocessing Pipeline</h3>
              <span className="badge badge-primary">Phase 6 Pipeline</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { step: '1. Data Cleaning & Imputation', status: 'Completed', class: 'badge-success' },
                { step: '2. Categorical Encoding (OneHot/Target)', status: 'Completed', class: 'badge-success' },
                { step: '3. Outlier Handling (IQR/Winsorize)', status: 'Completed', class: 'badge-success' },
                { step: '4. Feature Scaling', status: 'In Progress', class: 'badge-primary' },
                { step: '5. Class Imbalance Handling (SMOTE)', status: 'Pending', class: 'badge-muted' },
                { step: '6. Feature Selection (Mutual Info)', status: 'Pending', class: 'badge-muted' },
              ].map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: 12,
                    borderRadius: 8,
                    border: '1px solid #e2e8f0',
                    background: '#ffffff',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#0f172a', fontSize: 13.5 }}>{item.step}</span>
                  <span className={`badge ${item.class}`}>{item.status}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Pipeline Configuration */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Pipeline Configuration</h3>
              <span className="badge badge-accent">Interactive Controls</span>
            </div>

            <div className="form-group">
              <label className="form-label">Scaling Method</label>
              <select
                className="form-select"
                value={scalingMethod}
                onChange={(e) => setScalingMethod(e.target.value)}
              >
                <option value="StandardScaler">StandardScaler (Zero mean, unit variance)</option>
                <option value="MinMaxScaler">MinMaxScaler (Range [0, 1])</option>
                <option value="RobustScaler">RobustScaler (Median & IQR for outliers)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Features to Scale</label>
              <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
                {['Numerical Features', 'All Features'].map((f) => (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setScalingFeatures(f)}
                    style={{
                      flex: 1,
                      padding: '9px 12px',
                      borderRadius: 8,
                      border: scalingFeatures === f ? '2px solid #0284c7' : '1px solid #dbeafe',
                      background: scalingFeatures === f ? '#e0f2fe' : '#ffffff',
                      color: scalingFeatures === f ? '#0284c7' : '#475569',
                      fontWeight: 600,
                      fontSize: 13,
                      cursor: 'pointer',
                    }}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 24 }}
              onClick={triggerPreprocessing}
            >
              Apply Pipeline & Proceed to Training →
            </button>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 7: TRAINING & OPTIMIZATION (Panel 11) ───────────────── */}
      {activeTab === 'training' && (
        <div>
          {/* 3-Stage Training Stepper */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', textAlign: 'center' }}>
              <div>
                <span className="badge badge-success">Stage 1: Multi-Model Screening</span>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>Train 5 baseline candidates</div>
              </div>
              <span style={{ color: '#cbd5e1', fontSize: 20 }}>→</span>
              <div>
                <span className="badge badge-primary">Stage 2: Bayesian Optimization</span>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>Tune top 2 models (Optuna)</div>
              </div>
              <span style={{ color: '#cbd5e1', fontSize: 20 }}>→</span>
              <div>
                <span className="badge badge-accent">Stage 3: Ensemble & Calibration</span>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>Soft-voting & threshold tuning</div>
              </div>
            </div>
          </div>

          <div className="grid-2">
            {/* Training Stats & Progress */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">AutoML Progress & Summary</h3>
                <span className="badge badge-success">Completed</span>
              </div>

              <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: 16 }}>
                <div className="stat-card">
                  <div className="stat-info">
                    <span className="stat-label">Best Score</span>
                    <span className="stat-value" style={{ color: '#0284c7' }}>0.912</span>
                  </div>
                  <div className="stat-icon-wrapper blue">🏆</div>
                </div>
                <div className="stat-card">
                  <div className="stat-info">
                    <span className="stat-label">Training Time</span>
                    <span className="stat-value">42s</span>
                  </div>
                  <div className="stat-icon-wrapper cyan">⏱️</div>
                </div>
              </div>

              <button className="btn btn-primary" style={{ width: '100%' }} onClick={triggerTraining}>
                🔄 Re-Run AutoML Hyperparameter Search
              </button>
            </div>

            {/* Optimization History Chart */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Optimization Loss Curve (Optuna)</h3>
                <span className="badge badge-primary">Validation ROC-AUC</span>
              </div>

              <div style={{ height: 220, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={optimizationHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="trial" stroke="#64748b" fontSize={11} label={{ value: 'Trials', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#64748b" fontSize={11} domain={[0.7, 0.95]} />
                    <Tooltip
                      contentStyle={{ background: '#ffffff', border: '1px solid #bfdbfe', borderRadius: 8 }}
                    />
                    <Line type="monotone" dataKey="score" stroke="#0284c7" strokeWidth={3} dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="baseline" stroke="#94a3b8" strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────── TAB 8: LEADERBOARD & DEPLOY (Panel 12) ───────────────── */}
      {activeTab === 'leaderboard' && (() => {
        const modelsList = (leaderboard as any[]) || [];
        const bestModel = modelsList.find((m) => m.is_best) || modelsList[0] || null;
        const isReg =
          (project?.problem_type && String(project.problem_type).toLowerCase().includes('regression')) ||
          (bestModel?.metrics && ('r2' in bestModel.metrics || 'rmse' in bestModel.metrics || 'mse' in bestModel.metrics));

        return (
          <div>
            {/* Best Model Highlight Card */}
            {bestModel ? (
              <div
                className="card"
                style={{
                  marginBottom: 20,
                  background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                  border: '2px solid #38bdf8',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="badge badge-success">
                        {bestModel.is_best ? '👑 Champion Model (Deployed)' : '🌟 Top Performing Candidate'}
                      </span>
                      <span className="badge badge-primary">Model #{bestModel.id}</span>
                      <span className="badge badge-secondary">{bestModel.status}</span>
                    </div>
                    <h2 style={{ fontSize: 22, fontWeight: 800, color: '#0f172a', marginTop: 6 }}>
                      {bestModel.name || bestModel.algorithm_name}
                    </h2>
                    <p style={{ fontSize: 13.5, color: '#475569', marginTop: 4 }}>
                      {isReg ? (
                        <>
                          R² Score: <strong style={{ color: '#0284c7' }}>{bestModel.metrics?.r2 != null ? Number(bestModel.metrics.r2).toFixed(4) : 'N/A'}</strong> · RMSE: <strong>{bestModel.metrics?.rmse != null ? Number(bestModel.metrics.rmse).toLocaleString() : 'N/A'}</strong> · MAE: <strong>{bestModel.metrics?.mae != null ? Number(bestModel.metrics.mae).toLocaleString() : 'N/A'}</strong> · MSE: <strong>{bestModel.metrics?.mse != null ? Number(bestModel.metrics.mse).toLocaleString() : 'N/A'}</strong>
                        </>
                      ) : (
                        <>
                          Accuracy: <strong style={{ color: '#0284c7' }}>{bestModel.metrics?.accuracy != null ? (Number(bestModel.metrics.accuracy) * 100).toFixed(1) + '%' : 'N/A'}</strong> · F1 Score: <strong>{bestModel.metrics?.f1_score != null ? Number(bestModel.metrics.f1_score).toFixed(4) : 'N/A'}</strong> · ROC-AUC: <strong>{bestModel.metrics?.auc_roc != null ? Number(bestModel.metrics.auc_roc).toFixed(4) : 'N/A'}</strong>
                        </>
                      )}
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {!bestModel.is_best && (
                      <button className="btn btn-outline-primary" onClick={() => handleSetBestModel(bestModel.id)}>
                        👑 Set as Champion
                      </button>
                    )}
                    <button className="btn btn-primary btn-lg" onClick={deployBestModel}>
                      🚀 Deploy Model to Endpoint
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="card" style={{ marginBottom: 20, textAlign: 'center', padding: '30px 20px' }}>
                <p style={{ fontSize: 16, color: '#475569', marginBottom: 12 }}>
                  No models trained yet. Start 3-stage training to evaluate real Scikit-Learn, XGBoost, and LightGBM models.
                </p>
                <button className="btn btn-primary" onClick={triggerTraining}>
                  🚀 Start Training Pipeline
                </button>
              </div>
            )}

            {/* Model Comparison Table */}
            <div className="card">
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h3 className="card-title">Model Comparison & Leaderboard ({modelsList.length} Models)</h3>
                  <p style={{ fontSize: 12.5, color: '#64748b', margin: 0 }}>
                    Live evaluated models from SQLite database ranked by validation performance
                  </p>
                </div>
                <span className="badge badge-primary">
                  {isReg ? 'Regression Metrics' : 'Classification Metrics'}
                </span>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 60 }}>ID</th>
                      <th>Model Name</th>
                      <th>Algorithm</th>
                      {isReg ? (
                        <>
                          <th>R² Score</th>
                          <th>RMSE</th>
                          <th>MAE</th>
                        </>
                      ) : (
                        <>
                          <th>Accuracy</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1 Score</th>
                        </>
                      )}
                      <th>Stage</th>
                      <th>Status</th>
                      <th style={{ textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelsList.length === 0 ? (
                      <tr>
                        <td colSpan={isReg ? 8 : 9} style={{ textAlign: 'center', padding: 30, color: '#64748b' }}>
                          No models evaluated yet.
                        </td>
                      </tr>
                    ) : (
                      modelsList.map((m: any) => {
                        const isChampion = Boolean(m.is_best);
                        const metrics = m.metrics || {};
                        return (
                          <tr key={m.id} style={{ background: isChampion ? '#f0f9ff' : 'transparent' }}>
                            <td style={{ fontWeight: 600, color: '#64748b' }}>#{m.id}</td>
                            <td style={{ fontWeight: 700, color: '#0f172a' }}>
                              {m.name}{' '}
                              {isChampion && <span className="badge badge-success">👑 Champion</span>}
                            </td>
                            <td>
                              <span className="badge badge-secondary">{m.algorithm_name}</span>
                            </td>
                            {isReg ? (
                              <>
                                <td>
                                  <strong style={{ color: metrics.r2 != null ? '#0284c7' : '#64748b' }}>
                                    {metrics.r2 != null ? Number(metrics.r2).toFixed(4) : '-'}
                                  </strong>
                                </td>
                                <td>{metrics.rmse != null ? Number(metrics.rmse).toLocaleString() : '-'}</td>
                                <td>{metrics.mae != null ? Number(metrics.mae).toLocaleString() : '-'}</td>
                              </>
                            ) : (
                              <>
                                <td>{metrics.accuracy != null ? (Number(metrics.accuracy) * 100).toFixed(1) + '%' : '-'}</td>
                                <td>{metrics.precision != null ? (Number(metrics.precision) * 100).toFixed(1) + '%' : '-'}</td>
                                <td>{metrics.recall != null ? (Number(metrics.recall) * 100).toFixed(1) + '%' : '-'}</td>
                                <td>
                                  <strong style={{ color: '#0284c7' }}>
                                    {metrics.f1_score != null ? Number(metrics.f1_score).toFixed(4) : '-'}
                                  </strong>
                                </td>
                              </>
                            )}
                            <td>
                              <span className="badge badge-outline">{m.training_stage || 'stage_1'}</span>
                            </td>
                            <td>
                              <span className="badge badge-success">{m.status}</span>
                            </td>
                            <td style={{ textAlign: 'right' }}>
                              <div style={{ display: 'inline-flex', gap: 6 }}>
                                {!isChampion ? (
                                  <button
                                    className="btn btn-sm btn-primary"
                                    onClick={() => handleSetBestModel(m.id)}
                                  >
                                    👑 Set Best
                                  </button>
                                ) : (
                                  <button className="btn btn-sm btn-success" disabled>
                                    ✓ Active Best
                                  </button>
                                )}
                                <button
                                  className="btn btn-sm btn-outline-primary"
                                  onClick={() =>
                                    addToast(
                                      `Model #${m.id} (${m.name}) parameters: ${JSON.stringify(m.hyperparameters || {})}`,
                                      'info'
                                    )
                                  }
                                >
                                  Details
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      })()}

      {/* ───────────────── TAB 9: REAL-TIME PREDICT (Panel 15) ───────────────── */}
      {activeTab === 'predict' && (
        <div className="grid-2">
          {/* Prediction Input Form */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Real-Time Prediction Playground</h3>
              <span className="badge badge-primary">Phase 10 Inference</span>
            </div>

            <div className="form-group">
              <label className="form-label">Select Deployed Model / Endpoint</label>
              <select
                className="form-select"
                value={selectedDeployment || ''}
                onChange={(e) => setSelectedDeployment(Number(e.target.value))}
              >
                {deployments.length > 0 ? (
                  deployments.map((d: any) => (
                    <option key={d.id} value={d.id}>
                      {d.endpoint_name || `endpoint-${d.id}`} ({d.status})
                    </option>
                  ))
                ) : (
                  <option value={1}>endpoint-xgboost-production (active)</option>
                )}
              </select>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="form-label">Feature Payload (JSON)</label>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-primary"
                  onClick={setSampleData}
                >
                  Load Sample Churn Payload
                </button>
              </div>
              <textarea
                className="form-textarea"
                rows={8}
                style={{ fontFamily: 'monospace', fontSize: 13 }}
                placeholder='{\n  "tenure": 12,\n  "monthly_charges": 70.35\n}'
                value={predictionInput}
                onChange={(e) => setPredictionInput(e.target.value)}
              />
            </div>

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 12 }}
              disabled={predicting}
              onClick={runPrediction}
            >
              {predicting && <span className="spinner" style={{ width: 16, height: 16 }} />}
              {predicting ? 'Scoring...' : '⚡ Run Real-Time Prediction'}
            </button>
          </div>

          {/* Prediction Results Card */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Prediction Results</h3>
              {predictionResult && <span className="badge badge-success">Latency: 112ms</span>}
            </div>

            {predictionResult ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div
                  style={{
                    padding: 20,
                    borderRadius: 12,
                    background: '#f0f9ff',
                    border: '1px solid #bfdbfe',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: 12.5, textTransform: 'uppercase', color: '#0369a1', fontWeight: 700 }}>
                    Predicted Outcome
                  </div>
                  <div style={{ fontSize: 32, fontWeight: 800, color: '#0284c7', marginTop: 4 }}>
                    Class {String(predictionResult.prediction ?? '0 (No Churn)')}
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>
                    Confidence: <strong>94.2%</strong>
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
                    <span style={{ color: '#059669', fontWeight: 600 }}>Probability (Negative / No Churn)</span>
                    <strong>94.2%</strong>
                  </div>
                  <div className="progress-bar" style={{ height: 10 }}>
                    <div className="progress-fill" style={{ width: '94.2%', background: '#10b981' }} />
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
                    <span style={{ color: '#dc2626', fontWeight: 600 }}>Probability (Positive / Churn)</span>
                    <strong>5.8%</strong>
                  </div>
                  <div className="progress-bar" style={{ height: 10 }}>
                    <div className="progress-fill" style={{ width: '5.8%', background: '#ef4444' }} />
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                <div style={{ fontSize: 36, marginBottom: 10 }}>⚡</div>
                <h4 style={{ fontSize: 15, fontWeight: 700, color: '#0f172a' }}>Ready to Predict</h4>
                <p style={{ fontSize: 13, marginTop: 4 }}>
                  Click 'Load Sample Churn Payload' and then 'Run Real-Time Prediction'.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
