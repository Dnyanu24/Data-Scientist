import { Link } from 'react-router-dom';
import { BrainCircuit } from 'lucide-react';

export default function TopHeader() {
  return (
    <header className="top-header">
      <Link to="/" className="top-brand">
        <div className="brand-icon">
          <BrainCircuit size={22} color="#ffffff" />
        </div>
        <div className="brand-title" style={{ fontSize: '18px', fontWeight: 800, letterSpacing: '-0.3px', color: '#ffffff' }}>
          AI Data Scientist
        </div>
      </Link>
    </header>
  );
}
