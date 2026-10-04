import { Link, useLocation } from 'react-router-dom';
import {
  Upload,
  Search,
  Fingerprint,
  Sparkles,
  Target,
  SlidersHorizontal,
  Bot,
  BarChart3,
  Rocket,
  Zap,
  Radio,
  RefreshCw,
  BrainCircuit,
  Check,
  ChevronRight,
} from 'lucide-react';

const LIFECYCLE_STEPS = [
  { num: 1,  key: 'upload',      label: 'Upload',      icon: Upload },
  { num: 2,  key: 'understand',  label: 'Understand',  icon: Search },
  { num: 3,  key: 'fingerprint', label: 'Fingerprint', icon: Fingerprint },
  { num: 4,  key: 'learn',       label: 'Learn',       icon: Sparkles },
  { num: 5,  key: 'recommend',   label: 'Recommend',   icon: Target },
  { num: 6,  key: 'preprocess',  label: 'Preprocess',  icon: SlidersHorizontal },
  { num: 7,  key: 'train',       label: 'Train',       icon: Bot },
  { num: 8,  key: 'evaluate',    label: 'Evaluate',    icon: BarChart3 },
  { num: 9,  key: 'deploy',      label: 'Deploy',      icon: Rocket },
  { num: 10, key: 'predict',     label: 'Predict',     icon: Zap },
  { num: 11, key: 'monitor',     label: 'Monitor',     icon: Radio },
  { num: 12, key: 'knowledge',   label: 'Learn',       icon: RefreshCw },
];

export default function TopHeader() {
  const location = useLocation();

  const getActiveStep = () => {
    const path = location.pathname;
    if (path.includes('/predict')) return 10;
    if (path.includes('/monitoring')) return 11;
    if (path.includes('/knowledge')) return 12;
    if (path.includes('/model/')) return 8;
    if (path.includes('/experiments')) return 7;
    if (path.includes('/datasets')) return 1;
    return 6;
  };

  const activeNum = getActiveStep();

  return (
    <header className="top-header">
      <Link to="/" className="top-brand">
        <div className="brand-icon">
          <BrainCircuit size={22} color="#38bdf8" />
        </div>
        <div>
          <div className="brand-title">AI Data Scientist</div>
          <div className="brand-subtitle">From Data to Decisions — Powered by AI</div>
        </div>
      </Link>

      <div className="top-stepper">
        {LIFECYCLE_STEPS.map((step, idx) => {
          const isCompleted = step.num < activeNum;
          const isActive = step.num === activeNum;
          const IconComp = step.icon;
          return (
            <div key={`${step.key}-${idx}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                className={`top-step-pill ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                title={`Phase ${step.num}: ${step.label}`}
              >
                {isCompleted
                  ? <Check size={12} strokeWidth={3} />
                  : <IconComp size={12} />
                }
                <span>{step.label}</span>
              </div>
              {idx < LIFECYCLE_STEPS.length - 1 && (
                <ChevronRight size={12} style={{ color: '#475569', flexShrink: 0 }} />
              )}
            </div>
          );
        })}
      </div>
    </header>
  );
}
