import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useToast } from '../contexts/ToastContext';

export default function AuthPage() {
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [loading, setLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const { login, register } = useAuth();
  const { addToast } = useToast();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: 'admin@aidatascientist.io',
    password: 'admin123',
    username: 'admin',
    full_name: 'Admin Data Scientist',
  });

  const handle = (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (tab === 'login') {
        await login(form.email, form.password);
      } else {
        await register({
          email: form.email,
          password: form.password,
          username: form.username,
          full_name: form.full_name,
        });
      }
      addToast(tab === 'login' ? 'Welcome back!' : 'Account registered successfully!', 'success');
      navigate('/');
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Authentication failed';
      addToast(msg, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-split-wrapper">
      {/* Left Hero Panel (Panel 1 Mockup) */}
      <div className="auth-hero-side">
        <div className="auth-hero-glow" />
        <div style={{ maxWidth: 460, zIndex: 2 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
            <div className="brand-icon" style={{ width: 48, height: 48, fontSize: 26 }}>
              🧠
            </div>
            <div>
              <div style={{ fontSize: 26, fontWeight: 800, color: '#ffffff', letterSpacing: -0.5 }}>
                AI Data Scientist
              </div>
              <div style={{ fontSize: 13, color: '#7dd3fc', fontWeight: 600 }}>
                Smart ML Pipelines. Better Decisions.
              </div>
            </div>
          </div>

          <h2 style={{ fontSize: 28, fontWeight: 800, color: '#ffffff', lineHeight: 1.3, marginBottom: 16 }}>
            Autonomous Machine Learning Lifecycle from Raw Data to Deployed Decisions
          </h2>

          <p style={{ fontSize: 14, color: '#bae6fd', lineHeight: 1.6, marginBottom: 32 }}>
            Empower your organization with full 11-phase automated intelligence: dataset fingerprinting,
            meta-learning recommendations, adaptive preprocessing, multi-model hyperparameter training,
            and real-time drift monitoring.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              'Automated 11-phase ML pipeline orchestration',
              'Meta-learning & hybrid algorithm recommendations',
              'Self-healing adaptive preprocessing pipelines',
              'Sub-second real-time inference & statistical drift detection',
            ].map((feat, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13.5, color: '#e0f2fe' }}>
                <span style={{ color: '#38bdf8', fontWeight: 800 }}>✓</span>
                <span>{feat}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Form Card (Panel 1 Mockup) */}
      <div className="auth-form-side">
        <div className="auth-card">
          <div style={{ marginBottom: 24 }}>
            <h1 style={{ fontSize: 24, fontWeight: 800, color: '#0f172a', letterSpacing: -0.5 }}>
              {tab === 'login' ? 'Welcome Back' : 'Create an Account'}
            </h1>
            <p style={{ fontSize: 13.5, color: '#64748b', marginTop: 4 }}>
              {tab === 'login'
                ? 'Sign in to access your AI Data Scientist platform'
                : 'Get started with autonomous machine learning'}
            </p>
          </div>

          {/* Quick tab switcher */}
          <div
            style={{
              display: 'flex',
              background: '#f0f7ff',
              padding: 4,
              borderRadius: 10,
              marginBottom: 20,
              border: '1px solid #dbeafe',
            }}
          >
            <button
              type="button"
              onClick={() => setTab('login')}
              style={{
                flex: 1,
                padding: '8px 12px',
                border: 'none',
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
                background: tab === 'login' ? '#ffffff' : 'transparent',
                color: tab === 'login' ? '#0284c7' : '#64748b',
                boxShadow: tab === 'login' ? '0 2px 6px rgba(2, 132, 199, 0.12)' : 'none',
                transition: 'all 0.2s',
              }}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setTab('register')}
              style={{
                flex: 1,
                padding: '8px 12px',
                border: 'none',
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
                background: tab === 'register' ? '#ffffff' : 'transparent',
                color: tab === 'register' ? '#0284c7' : '#64748b',
                boxShadow: tab === 'register' ? '0 2px 6px rgba(2, 132, 199, 0.12)' : 'none',
                transition: 'all 0.2s',
              }}
            >
              Register
            </button>
          </div>

          <form onSubmit={submit}>
            {tab === 'register' && (
              <>
                <div className="form-group">
                  <label className="form-label">Full Name</label>
                  <input
                    className="form-input"
                    name="full_name"
                    placeholder="e.g. John Doe"
                    value={form.full_name}
                    onChange={handle}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Username</label>
                  <input
                    className="form-input"
                    name="username"
                    placeholder="e.g. jdoe"
                    value={form.username}
                    onChange={handle}
                    required
                  />
                </div>
              </>
            )}

            <div className="form-group">
              <label className="form-label">Email Address</label>
              <input
                className="form-input"
                type="email"
                name="email"
                placeholder="you@company.com"
                value={form.email}
                onChange={handle}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <input
                className="form-input"
                type="password"
                name="password"
                placeholder="••••••••"
                value={form.password}
                onChange={handle}
                required
              />
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: 20,
                fontSize: 13,
              }}
            >
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#475569', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  style={{ accentColor: '#0284c7' }}
                />
                Remember me
              </label>
              <a
                href="#forgot"
                onClick={(e) => {
                  e.preventDefault();
                  addToast('Default credentials loaded: admin@aidatascientist.io / admin123', 'info');
                }}
                style={{ color: '#0284c7', textDecoration: 'none', fontWeight: 600 }}
              >
                Forgot password?
              </a>
            </div>

            <button
              className="btn btn-primary"
              type="submit"
              disabled={loading}
              style={{ width: '100%', padding: '11px 16px', fontSize: 14 }}
            >
              {loading && <span className="spinner" style={{ width: 18, height: 18 }} />}
              {loading ? 'Authenticating...' : tab === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </form>

          <div
            style={{
              textAlign: 'center',
              marginTop: 24,
              fontSize: 13,
              color: '#64748b',
              borderTop: '1px solid #e2e8f0',
              paddingTop: 16,
            }}
          >
            {tab === 'login' ? (
              <>
                Don't have an account?{' '}
                <button
                  type="button"
                  onClick={() => setTab('register')}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#0284c7',
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                >
                  Register
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button
                  type="button"
                  onClick={() => setTab('login')}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#0284c7',
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                >
                  Sign In
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
