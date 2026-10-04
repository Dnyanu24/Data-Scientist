import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
  LayoutDashboard,
  FolderOpen,
  Database,
  FlaskConical,
  BrainCircuit,
  Activity,
  Lightbulb,
  Settings,
  BookOpen,
  FileText,
  LogOut,
} from 'lucide-react';

interface NavItem {
  icon: React.ReactNode;
  label: string;
  path: string;
}

const NAV_ITEMS: NavItem[] = [
  { icon: <LayoutDashboard size={16} />, label: 'Dashboard', path: '/' },
  { icon: <FolderOpen size={16} />, label: 'Projects', path: '/projects' },
  { icon: <Database size={16} />, label: 'Datasets', path: '/datasets' },
  { icon: <FlaskConical size={16} />, label: 'Experiments', path: '/experiments' },
  { icon: <BrainCircuit size={16} />, label: 'Models', path: '/models' },
  { icon: <Activity size={16} />, label: 'Monitoring', path: '/monitoring' },
  { icon: <Lightbulb size={16} />, label: 'Knowledge', path: '/knowledge' },
  { icon: <Settings size={16} />, label: 'Settings', path: '/settings' },
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
        <a className="nav-item" href="http://localhost:8001/docs" target="_blank" rel="noopener noreferrer">
          <span className="nav-icon"><BookOpen size={16} /></span>
          Swagger Docs
        </a>
        <a className="nav-item" href="http://localhost:8001/redoc" target="_blank" rel="noopener noreferrer">
          <span className="nav-icon"><FileText size={16} /></span>
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
          <LogOut size={14} style={{ color: '#94a3b8', flexShrink: 0 }} title="Logout" />
        </div>
      </div>
    </aside>
  );
}
