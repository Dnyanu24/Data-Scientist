import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

interface NavItem {
  icon: string;
  label: string;
  path: string;
}

const NAV_ITEMS: NavItem[] = [
  { icon: '🏠', label: 'Dashboard', path: '/' },
  { icon: '📁', label: 'Projects', path: '/projects' },
  { icon: '📊', label: 'Datasets', path: '/datasets' },
  { icon: '🔬', label: 'Experiments', path: '/experiments' },
  { icon: '🧠', label: 'Models', path: '/models' },
  { icon: '📡', label: 'Monitoring', path: '/monitoring' },
  { icon: '💡', label: 'Knowledge', path: '/knowledge' },
  { icon: '⚙️', label: 'Settings', path: '/settings' },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isItemActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        <div className="nav-section-label">Platform</div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.path}
            className={`nav-item ${isItemActive(item.path) ? 'active' : ''}`}
            onClick={() => navigate(item.path)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}

        <div className="nav-section-label" style={{ marginTop: 20 }}>Documentation</div>
        <a className="nav-item" href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
          <span className="nav-icon">📚</span>
          Swagger Docs
        </a>
        <a className="nav-item" href="http://localhost:8000/redoc" target="_blank" rel="noopener noreferrer">
          <span className="nav-icon">📖</span>
          ReDoc
        </a>
      </nav>

      <div className="sidebar-footer">
        <div className="user-chip" onClick={handleLogout} title="Click to sign out">
          <div className="user-avatar">
            {user?.username?.[0]?.toUpperCase() || 'U'}
          </div>
          <div className="user-info">
            <div className="user-name">{user?.full_name || user?.username || 'Data Scientist'}</div>
            <div className="user-email">{user?.email || 'admin@aidatascientist.io'}</div>
          </div>
          <span style={{ fontSize: 13, color: '#94a3b8' }} title="Logout">🚪</span>
        </div>
      </div>
    </aside>
  );
}
