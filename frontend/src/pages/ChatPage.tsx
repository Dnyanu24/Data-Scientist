import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { chatApi, projectsApi } from '../api/client';
import { useToast } from '../contexts/ToastContext';
import { ArrowLeft, Bot, User, Send } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const SUGGESTIONS = [
  'Explain my dataset and suggest the best ML approach',
  'What preprocessing steps do you recommend?',
  'Which algorithm should I use and why?',
  'Explain the model evaluation metrics',
  'How can I improve my model performance?',
  'What does the feature importance tell us?',
  'Are there any data quality issues I should fix?',
];

export default function ChatPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [project, setProject] = useState<Record<string, unknown> | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    projectsApi.get(Number(projectId)).then((r) => setProject(r.data)).catch(() => {});
    chatApi.history(Number(projectId))
      .then((r) => {
        const hist = r.data?.history || [];
        setMessages(hist);
      })
      .catch(() => {});
  }, [projectId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async (text?: string) => {
    const msg = text || input.trim();
    if (!msg || loading) return;
    setInput('');
    const userMsg: Message = { role: 'user', content: msg };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);
    try {
      const r = await chatApi.send(Number(projectId), msg, messages);
      const aiMsg: Message = { role: 'assistant', content: r.data.response || r.data.message || 'No response' };
      setMessages((m) => [...m, aiMsg]);
    } catch {
      addToast('Chat failed', 'error');
      setMessages((m) => [...m, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }]);
    } finally { setLoading(false); }
  }, [input, loading, messages, projectId, addToast]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="page-body" style={{ padding: 0, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 0px)' }}>
      {/* Header */}
      <div style={{ padding: '20px 32px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
        <div className="flex items-center gap-4">
          <button className="btn btn-secondary btn-sm" onClick={() => navigate(`/projects/${projectId}`)}
            style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <ArrowLeft size={14} /> Back
          </button>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Bot size={18} color="#0284c7" /> AI Data Science Assistant
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {project ? `Project: ${project.name as string}` : 'Loading...'}
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages" style={{ flex: 1 }}>
        {messages.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 40 }}>
            <div style={{ marginBottom: 12, color: '#0284c7' }}>
              <Bot size={52} strokeWidth={1.4} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>How can I help?</div>
            <div style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 28 }}>
              Ask me anything about your data, models, or ML strategy
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 700 }}>
              {SUGGESTIONS.map((s) => (
                <button key={s} className="btn btn-secondary btn-sm" onClick={() => send(s)}
                  style={{ fontSize: 12 }}>{s}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-message ${m.role}`}>
            <div className={`chat-avatar ${m.role}`}>
              {m.role === 'assistant'
                ? <Bot size={16} />
                : <User size={16} />
              }
            </div>
            <div className="chat-bubble" style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
          </div>
        ))}

        {loading && (
          <div className="chat-message ai">
            <div className="chat-avatar ai"><Bot size={16} /></div>
            <div className="chat-bubble">
              <span className="spinner" style={{ width: 16, height: 16, display: 'inline-block' }} />
              <span style={{ marginLeft: 8 }}>Thinking...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-row">
        <input
          className="chat-input"
          placeholder="Ask about your data, models, or get recommendations..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button className="btn btn-primary" onClick={() => send()} disabled={loading || !input.trim()}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
          {loading ? <span className="spinner" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  );
}
