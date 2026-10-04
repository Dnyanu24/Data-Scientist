import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useToast } from '../contexts/ToastContext';

export default function SettingsPage() {
  const { user } = useAuth();
  const { addToast } = useToast();
  const [activeTab, setActiveTab] = useState<'profile' | 'llm' | 'notifications'>('llm');

  const [llmProvider, setLlmProvider] = useState('Gemini (Default)');
  const [apiKey, setApiKey] = useState('••••••••••••••••••••••••••••••••');
  const [modelName, setModelName] = useState('gemini-1.5-pro');
  const [temperature, setTemperature] = useState(0.2);

  const saveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    addToast('Settings saved successfully!', 'success');
  };

  return (
    <div className="page-body">
      <div className="page-header">
        <h1 className="page-title">Platform Settings</h1>
        <p className="page-desc">Configure AI providers, LLM API keys, and workspace preferences</p>
      </div>

      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          Profile
        </button>
        <button
          className={`tab-btn ${activeTab === 'llm' ? 'active' : ''}`}
          onClick={() => setActiveTab('llm')}
        >
          LLM Provider Configuration
        </button>
        <button
          className={`tab-btn ${activeTab === 'notifications' ? 'active' : ''}`}
          onClick={() => setActiveTab('notifications')}
        >
          Notifications & Alerts
        </button>
      </div>

      {activeTab === 'llm' && (
        <div className="card" style={{ maxWidth: 680 }}>
          <div className="card-header">
            <div>
              <h3 className="card-title">LLM Provider Configuration</h3>
              <p className="card-subtitle">Select and configure the language model powering Phase 10 AI Chat & Reasoning</p>
            </div>
            <span className="badge badge-primary">Active</span>
          </div>

          <form onSubmit={saveSettings}>
            <div className="form-group">
              <label className="form-label">Provider</label>
              <select
                className="form-select"
                value={llmProvider}
                onChange={(e) => setLlmProvider(e.target.value)}
              >
                <option value="Gemini (Default)">Google Gemini (Gemini 1.5 Pro / Flash)</option>
                <option value="OpenAI">OpenAI (GPT-4o / GPT-4o-mini)</option>
                <option value="Anthropic">Anthropic (Claude 3.5 Sonnet)</option>
                <option value="Local LLM">Local LLM (Ollama / vLLM / llama.cpp)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">API Key</label>
              <input
                className="form-input"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
              />
              <span style={{ fontSize: 12, color: '#64748b' }}>
                You can also configure GEMINI_API_KEY or OPENAI_API_KEY in backend .env
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Model Name</label>
              <input
                className="form-input"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              />
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <label className="form-label">Temperature: {temperature}</label>
                <span style={{ fontSize: 12, color: '#64748b' }}>Lower = more deterministic</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                style={{ accentColor: '#0284c7', width: '100%', marginTop: 6 }}
              />
            </div>

            <button type="submit" className="btn btn-primary" style={{ marginTop: 12 }}>
              Save Changes
            </button>
          </form>
        </div>
      )}

      {activeTab === 'profile' && (
        <div className="card" style={{ maxWidth: 560 }}>
          <h3 className="card-title" style={{ marginBottom: 16 }}>User Information</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
              <span style={{ color: '#64748b' }}>Full Name</span>
              <strong>{user?.full_name || 'Admin Data Scientist'}</strong>
            </div>
            <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
              <span style={{ color: '#64748b' }}>Username</span>
              <strong>{user?.username || 'admin'}</strong>
            </div>
            <div className="flex justify-between" style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
              <span style={{ color: '#64748b' }}>Email</span>
              <strong>{user?.email || 'admin@aidatascientist.io'}</strong>
            </div>
            <div className="flex justify-between" style={{ padding: '8px 0' }}>
              <span style={{ color: '#64748b' }}>Theme Preference</span>
              <span className="badge badge-primary">White & Light Blue</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'notifications' && (
        <div className="card" style={{ maxWidth: 560 }}>
          <h3 className="card-title" style={{ marginBottom: 16 }}>Drift Alerts & Pipeline Notifications</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
              <input type="checkbox" defaultChecked style={{ accentColor: '#0284c7' }} />
              <span>Email notification upon Training Stage completion</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
              <input type="checkbox" defaultChecked style={{ accentColor: '#0284c7' }} />
              <span>Alert when statistical feature drift exceeds threshold (p &lt; 0.05)</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
              <input type="checkbox" defaultChecked style={{ accentColor: '#0284c7' }} />
              <span>Weekly continuous learning meta-summary</span>
            </label>
          </div>
          <button
            className="btn btn-primary"
            style={{ marginTop: 20 }}
            onClick={() => addToast('Notification preferences updated!', 'success')}
          >
            Update Preferences
          </button>
        </div>
      )}
    </div>
  );
}
