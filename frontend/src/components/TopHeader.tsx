import { Link, useLocation } from 'react-router-dom';

const LIFECYCLE_STEPS = [
  { num: 1, key: 'upload', label: 'Upload', icon: '📤' },
  { num: 2, key: 'understand', label: 'Understand', icon: '🔍' },
  { num: 3, key: 'fingerprint', label: 'Fingerprint', icon: '🧬' },
  { num: 4, key: 'learn', label: 'Learn', icon: '💡' },
  { num: 5, key: 'recommend', label: 'Recommend', icon: '🎯' },
  { num: 6, key: 'preprocess', label: 'Preprocess', icon: '⚙️' },
  { num: 7, key: 'train', label: 'Train', icon: '🤖' },
  { num: 8, key: 'evaluate', label: 'Evaluate', icon: '📊' },
  { num: 9, key: 'deploy', label: 'Deploy', icon: '🚀' },
  { num: 10, key: 'predict', label: 'Predict', icon: '⚡' },
  { num: 11, key: 'monitor', label: 'Monitor', icon: '📡' },
  { num: 12, key: 'knowledge', label: 'Learn', icon: '🔄' },
];

export default function TopHeader() {
  const location = useLocation();

  // Determine current active step based on path
  const getActiveStep = () => {
    const path = location.pathname;
    if (path.includes('/predict')) return 10;
    if (path.includes('/monitoring')) return 11;
    if (path.includes('/knowledge')) return 12;
    if (path.includes('/model/')) return 8;
    if (path.includes('/experiments')) return 7;
    if (path.includes('/datasets')) return 1;
    return 6; // Default active midpoint for dashboard / project view
  };

  const activeNum = getActiveStep();

  return (
    <header className="top-header">
      <Link to="/" className="top-brand">
        <div className="brand-icon">🧠</div>
        <div>
          <div className="brand-title">AI Data Scientist</div>
          <div className="brand-subtitle">From Data to Decisions — Powered by AI</div>
        </div>
      </Link>

      <div className="top-stepper">
        {LIFECYCLE_STEPS.map((step, idx) => {
          const isCompleted = step.num < activeNum;
          const isActive = step.num === activeNum;
          return (
            <div key={`${step.key}-${idx}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                className={`top-step-pill ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                title={`Phase ${step.num}: ${step.label}`}
              >
                <span>{isCompleted ? '✓' : step.icon}</span>
                <span>{step.label}</span>
              </div>
              {idx < LIFECYCLE_STEPS.length - 1 && <span className="top-step-arrow">→</span>}
            </div>
          );
        })}
      </div>
    </header>
  );
}
