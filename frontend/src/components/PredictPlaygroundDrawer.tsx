/**
 * PredictPlaygroundDrawer
 * Slide-over panel for live inference against a deployed model.
 * - Fetches feature metadata from GET /api/v1/deployments/{id}
 * - Renders a dynamic input form from feature_names + sample_input
 * - POSTs features to /api/v1/deployments/{endpoint_name}/predict
 * - Displays prediction result + latency inline
 */
import { useEffect, useRef, useState } from 'react';
import { X, Zap, Loader2, CheckCircle2, AlertCircle, ClipboardCopy } from 'lucide-react';
import { deploymentsApi } from '../api/client';

/* ─── Types ──────────────────────────────────────────────────────────── */
export interface SelectedEndpoint {
  id: number;
  endpoint_name: string;
  model_name?: string;
  algorithm_name?: string;
  project_name?: string;
  metrics?: Record<string, number>;
  feature_names?: string[];
  sample_input?: Record<string, unknown>;
}

interface PredictPlaygroundDrawerProps {
  endpoint: SelectedEndpoint | null;
  open: boolean;
  onClose: () => void;
}

interface PredictionResult {
  prediction: unknown;
  probability?: unknown;
  latency_ms: number;
  prediction_id?: number;
}

/* ─── Helpers ────────────────────────────────────────────────────────── */
function guessInputType(val: unknown): 'number' | 'text' {
  return typeof val === 'number' ? 'number' : 'text';
}

function parseValue(raw: string, type: 'number' | 'text'): unknown {
  if (type === 'number') {
    const n = parseFloat(raw);
    return isNaN(n) ? raw : n;
  }
  return raw;
}

function formatResult(val: unknown): string {
  if (val === null || val === undefined) return '—';
  if (typeof val === 'object') return JSON.stringify(val, null, 2);
  return String(val);
}

/* ─── Component ──────────────────────────────────────────────────────── */
export default function PredictPlaygroundDrawer({
  endpoint,
  open,
  onClose,
}: PredictPlaygroundDrawerProps) {
  /* Full metadata fetched from the API (may enrich feature_names / sample_input) */
  const [meta, setMeta] = useState<SelectedEndpoint | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(false);

  /* Dynamic feature input values keyed by feature name */
  const [inputs, setInputs] = useState<Record<string, string>>({});

  /* Prediction state */
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predError, setPredError] = useState<string | null>(null);

  /* JSON editor toggle */
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);

  const drawerRef = useRef<HTMLDivElement>(null);

  /* ── Fetch fresh metadata whenever a new endpoint is selected ── */
  useEffect(() => {
    if (!endpoint || !open) return;
    setResult(null);
    setPredError(null);
    setJsonError(null);
    setLoadingMeta(true);

    deploymentsApi
      .get(endpoint.id)
      .then((r) => {
        const data: SelectedEndpoint = r.data;
        setMeta(data);
        const sample = data.sample_input || {};
        const initial: Record<string, string> = {};
        (data.feature_names || []).forEach((f) => {
          initial[f] = sample[f] !== undefined ? String(sample[f]) : '';
        });
        setInputs(initial);
        setJsonText(JSON.stringify(sample, null, 2));
      })
      .catch(() => {
        /* Fall back to whatever was passed in via props */
        setMeta(endpoint);
        const sample = endpoint.sample_input || {};
        const initial: Record<string, string> = {};
        (endpoint.feature_names || []).forEach((f) => {
          initial[f] = sample[f] !== undefined ? String(sample[f]) : '';
        });
        setInputs(initial);
        setJsonText(JSON.stringify(sample, null, 2));
      })
      .finally(() => setLoadingMeta(false));
  }, [endpoint?.id, open]); // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Close on Escape ── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  /* ── Sync JSON editor when inputs change (form → JSON) ── */
  useEffect(() => {
    if (jsonMode || !meta) return;
    const obj: Record<string, unknown> = {};
    (meta.feature_names || []).forEach((f) => {
      const raw = inputs[f] ?? '';
      const type = guessInputType(meta.sample_input?.[f]);
      obj[f] = parseValue(raw, type);
    });
    setJsonText(JSON.stringify(obj, null, 2));
  }, [inputs, jsonMode]); // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Build features payload ── */
  const buildFeatures = (): Record<string, unknown> | null => {
    if (jsonMode) {
      try {
        const parsed = JSON.parse(jsonText);
        setJsonError(null);
        return parsed;
      } catch {
        setJsonError('Invalid JSON — please fix syntax errors before predicting.');
        return null;
      }
    }
    const obj: Record<string, unknown> = {};
    (meta?.feature_names || []).forEach((f) => {
      const raw = inputs[f] ?? '';
      const type = guessInputType(meta?.sample_input?.[f]);
      obj[f] = parseValue(raw, type);
    });
    return obj;
  };

  /* ── Run prediction ── */
  const handlePredict = async () => {
    if (!endpoint) return;
    const features = buildFeatures();
    if (!features) return;

    setPredicting(true);
    setResult(null);
    setPredError(null);

    try {
      const res = await deploymentsApi.predict(endpoint.endpoint_name, features);
      setResult(res.data);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Inference failed. Check your input values and try again.';
      setPredError(msg);
    } finally {
      setPredicting(false);
    }
  };

  /* ── Copy result to clipboard ── */
  const copyResult = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2)).catch(() => {});
  };

  const activeEndpoint = meta || endpoint;
  const featureNames = activeEndpoint?.feature_names || [];
  const sampleInput = activeEndpoint?.sample_input || {};

  /* ─── Render ──────────────────────────────────────────────────────── */
  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.45)',
          backdropFilter: 'blur(2px)',
          zIndex: 999,
          opacity: open ? 1 : 0,
          pointerEvents: open ? 'auto' : 'none',
          transition: 'opacity 0.25s ease',
        }}
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          height: '100vh',
          width: 'min(560px, 95vw)',
          background: '#fff',
          boxShadow: '-8px 0 40px rgba(0,0,0,0.18)',
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          transform: open ? 'translateX(0)' : 'translateX(110%)',
          transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          overflow: 'hidden',
        }}
      >
        {/* ── Header ── */}
        <div
          style={{
            padding: '20px 24px 16px',
            borderBottom: '1px solid #e2e8f0',
            background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)',
            color: '#fff',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <Zap size={18} color="#38bdf8" />
                <span style={{ fontWeight: 700, fontSize: 16 }}>Predict Playground</span>
              </div>
              <code
                style={{
                  fontSize: 12,
                  background: 'rgba(255,255,255,0.12)',
                  padding: '2px 8px',
                  borderRadius: 4,
                  color: '#bae6fd',
                }}
              >
                {activeEndpoint?.endpoint_name || '—'}
              </code>
              <div style={{ marginTop: 6, fontSize: 12, color: '#94a3b8' }}>
                {activeEndpoint?.algorithm_name || activeEndpoint?.model_name || 'Model'}{' '}
                {activeEndpoint?.project_name ? `· ${activeEndpoint.project_name}` : ''}
              </div>
            </div>
            <button
              onClick={onClose}
              style={{
                background: 'rgba(255,255,255,0.12)',
                border: 'none',
                borderRadius: 8,
                padding: 6,
                cursor: 'pointer',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Metrics row */}
          {activeEndpoint?.metrics && Object.keys(activeEndpoint.metrics).length > 0 && (
            <div style={{ marginTop: 12, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(activeEndpoint.metrics)
                .slice(0, 4)
                .map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: 6,
                      padding: '4px 10px',
                      fontSize: 11,
                      fontWeight: 600,
                    }}
                  >
                    <span style={{ color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                      {k.replace(/_/g, ' ')}
                    </span>{' '}
                    <span style={{ color: '#38bdf8' }}>
                      {typeof v === 'number' && v <= 1 && v >= 0
                        ? (v * 100).toFixed(1) + '%'
                        : Number(v).toFixed(4)}
                    </span>
                  </div>
                ))}
            </div>
          )}
        </div>

        {/* ── Body ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          {loadingMeta ? (
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#64748b' }}>
              <Loader2 size={28} style={{ animation: 'spin 1s linear infinite' }} />
              <p style={{ marginTop: 10, fontSize: 13 }}>Loading feature schema…</p>
            </div>
          ) : featureNames.length === 0 ? (
            /* No feature metadata → raw JSON editor only */
            <div>
              <p style={{ fontSize: 13, color: '#64748b', marginBottom: 12 }}>
                No feature schema found. Paste your input as JSON below.
              </p>
              <textarea
                value={jsonText}
                onChange={(e) => { setJsonText(e.target.value); setJsonError(null); }}
                rows={12}
                style={{
                  width: '100%',
                  fontFamily: 'monospace',
                  fontSize: 13,
                  padding: 12,
                  border: '1px solid #e2e8f0',
                  borderRadius: 8,
                  resize: 'vertical',
                  background: '#f8fafc',
                  boxSizing: 'border-box',
                }}
              />
            </div>
          ) : (
            <>
              {/* Mode toggle */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 18 }}>
                <button
                  className={`btn btn-sm ${!jsonMode ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setJsonMode(false)}
                >
                  Form
                </button>
                <button
                  className={`btn btn-sm ${jsonMode ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setJsonMode(true)}
                >
                  {'{ } JSON'}
                </button>
              </div>

              {jsonMode ? (
                /* JSON editor */
                <div>
                  <textarea
                    value={jsonText}
                    onChange={(e) => { setJsonText(e.target.value); setJsonError(null); }}
                    rows={Math.max(8, featureNames.length + 2)}
                    style={{
                      width: '100%',
                      fontFamily: 'monospace',
                      fontSize: 13,
                      padding: 12,
                      border: jsonError ? '1.5px solid #ef4444' : '1px solid #e2e8f0',
                      borderRadius: 8,
                      resize: 'vertical',
                      background: '#f8fafc',
                      boxSizing: 'border-box',
                    }}
                  />
                  {jsonError && (
                    <p style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{jsonError}</p>
                  )}
                </div>
              ) : (
                /* Dynamic form */
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  {featureNames.map((feat) => {
                    const inputType = guessInputType(sampleInput[feat]);
                    return (
                      <div key={feat}>
                        <label
                          style={{
                            display: 'block',
                            fontSize: 12,
                            fontWeight: 600,
                            color: '#374151',
                            marginBottom: 5,
                            textTransform: 'capitalize',
                          }}
                        >
                          {feat.replace(/_/g, ' ')}
                          <span style={{ color: '#9ca3af', fontWeight: 400, marginLeft: 4 }}>
                            ({inputType})
                          </span>
                        </label>
                        <input
                          type={inputType}
                          value={inputs[feat] ?? ''}
                          onChange={(e) =>
                            setInputs((prev) => ({ ...prev, [feat]: e.target.value }))
                          }
                          placeholder={
                            sampleInput[feat] !== undefined ? String(sampleInput[feat]) : feat
                          }
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            border: '1px solid #e2e8f0',
                            borderRadius: 8,
                            fontSize: 14,
                            background: '#f8fafc',
                            boxSizing: 'border-box',
                            transition: 'border-color 0.15s',
                            outline: 'none',
                          }}
                          onFocus={(e) => (e.target.style.borderColor = '#0284c7')}
                          onBlur={(e) => (e.target.style.borderColor = '#e2e8f0')}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}

          {/* ── Result panel ── */}
          {(result || predError) && (
            <div
              style={{
                marginTop: 24,
                borderRadius: 10,
                border: predError ? '1.5px solid #fca5a5' : '1.5px solid #86efac',
                background: predError ? '#fef2f2' : '#f0fdf4',
                padding: 16,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: 10,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontWeight: 700, color: predError ? '#dc2626' : '#16a34a' }}>
                  {predError ? <AlertCircle size={16} /> : <CheckCircle2 size={16} />}
                  {predError ? 'Inference Error' : 'Prediction Result'}
                </div>
                {result && (
                  <button
                    onClick={copyResult}
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      color: '#64748b',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      fontSize: 12,
                    }}
                    title="Copy to clipboard"
                  >
                    <ClipboardCopy size={13} /> Copy
                  </button>
                )}
              </div>

              {predError ? (
                <p style={{ fontSize: 13, color: '#dc2626' }}>{predError}</p>
              ) : result ? (
                <>
                  <div
                    style={{
                      background: '#fff',
                      borderRadius: 8,
                      padding: '10px 14px',
                      fontFamily: 'monospace',
                      fontSize: 13,
                      border: '1px solid #d1fae5',
                      marginBottom: 10,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all',
                    }}
                  >
                    <span style={{ color: '#64748b', fontSize: 11 }}>prediction</span>
                    <br />
                    <strong style={{ fontSize: 20, color: '#0f172a' }}>
                      {formatResult(result.prediction)}
                    </strong>
                  </div>

                  {result.probability != null && (
                    <div
                      style={{
                        background: '#fff',
                        borderRadius: 8,
                        padding: '8px 14px',
                        fontFamily: 'monospace',
                        fontSize: 12,
                        border: '1px solid #d1fae5',
                        marginBottom: 10,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-all',
                        color: '#374151',
                      }}
                    >
                      <span style={{ color: '#64748b', fontSize: 11 }}>probabilities</span>
                      <br />
                      {formatResult(result.probability)}
                    </div>
                  )}

                  <div style={{ fontSize: 12, color: '#64748b', display: 'flex', gap: 16 }}>
                    <span>⚡ Latency: <strong style={{ color: '#0284c7' }}>{result.latency_ms}ms</strong></span>
                    {result.prediction_id && (
                      <span>ID: <strong>#{result.prediction_id}</strong></span>
                    )}
                  </div>
                </>
              ) : null}
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <div
          style={{
            padding: '16px 24px',
            borderTop: '1px solid #e2e8f0',
            background: '#f8fafc',
            display: 'flex',
            gap: 10,
            flexShrink: 0,
          }}
        >
          <button
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={predicting || loadingMeta}
            style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          >
            {predicting ? (
              <Loader2 size={15} style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <Zap size={15} />
            )}
            {predicting ? 'Running inference…' : 'Submit Prediction'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => {
              setResult(null);
              setPredError(null);
              /* Reset inputs to sample values */
              if (meta?.feature_names && meta.sample_input) {
                const reset: Record<string, string> = {};
                meta.feature_names.forEach((f) => {
                  reset[f] = meta.sample_input![f] !== undefined ? String(meta.sample_input![f]) : '';
                });
                setInputs(reset);
                setJsonText(JSON.stringify(meta.sample_input, null, 2));
              }
            }}
            disabled={predicting}
          >
            Reset
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </>
  );
}
