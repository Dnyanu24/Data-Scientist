import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 globally
api.interceptors.response.use(
  (r) => r,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

/* ─── Auth ─────────────────────────────────────────────────────────── */
export const authApi = {
  register: (data: { email: string; username: string; full_name: string; password: string }) =>
    api.post('/auth/register', data),
  login: (data: { email: string; password: string }) =>
    api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
};

/* ─── Projects ──────────────────────────────────────────────────────── */
export const projectsApi = {
  list: () => api.get('/projects/'),
  create: (data: { name: string; description?: string; business_goal?: string; target_column?: string }) =>
    api.post('/projects/', data),
  get: (id: number) => api.get(`/projects/${id}`),
  update: (id: number, data: Record<string, unknown>) => api.put(`/projects/${id}`, data),
  delete: (id: number) => api.delete(`/projects/${id}`),
  understandGoal: (id: number, data: { business_goal: string }) =>
    api.post(`/projects/${id}/understand-goal`, data),
};

/* ─── Datasets ──────────────────────────────────────────────────────── */
export const datasetsApi = {
  listByProject: (projectId: number) => api.get(`/datasets/project/${projectId}`),
  upload: (formData: FormData) =>
    api.post('/datasets/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  get: (id: number) => api.get(`/datasets/${id}`),
  preview: (id: number, page = 1, pageSize = 20) =>
    api.get(`/datasets/${id}/preview?page=${page}&page_size=${pageSize}`),
  profile: (id: number) => api.post(`/datasets/${id}/profile`),
  getProfile: (id: number) => api.get(`/datasets/${id}/profile`),
  fingerprint: (id: number) => api.post(`/datasets/${id}/fingerprint`),
  getFingerprint: (id: number) => api.get(`/datasets/${id}/fingerprint`),
  getSimilar: (id: number, topK = 5) => api.get(`/datasets/${id}/similar?top_k=${topK}`),
};

/* ─── Recommendations ───────────────────────────────────────────────── */
export const recommendationsApi = {
  trigger: (projectId: number) => api.post(`/recommendations/projects/${projectId}/recommend`),
  list: (projectId: number) => api.get(`/recommendations/projects/${projectId}/recommendations`),
  generatePreprocessing: (projectId: number) =>
    api.post(`/recommendations/projects/${projectId}/preprocessing/generate`),
  getPreprocessing: (projectId: number) =>
    api.get(`/recommendations/projects/${projectId}/preprocessing`),
};

/* ─── Experiments / Training ────────────────────────────────────────── */
export const experimentsApi = {
  startTraining: (projectId: number, config?: Record<string, unknown>) =>
    api.post(`/experiments/projects/${projectId}/train`, config || {}),
  listExperiments: (projectId: number) => api.get(`/experiments/projects/${projectId}/experiments`),
  leaderboard: (projectId: number) => api.get(`/experiments/projects/${projectId}/leaderboard`),
  getModel: (modelId: number) => api.get(`/experiments/models/${modelId}`),
  getEvaluation: (modelId: number) => api.get(`/experiments/models/${modelId}/evaluation`),
  getExplainability: (modelId: number) => api.get(`/experiments/models/${modelId}/explain`),
  setBest: (modelId: number) => api.post(`/experiments/models/${modelId}/set-best`),
};

/* ─── Deployments ───────────────────────────────────────────────────── */
export const deploymentsApi = {
  deploy: (modelId: number, endpointName?: string) =>
    api.post('/deployments/', { model_id: modelId, endpoint_name: endpointName }),
  list: (projectId?: number) =>
    api.get(`/deployments/${projectId ? `?project_id=${projectId}` : ''}`),
  get: (id: number) => api.get(`/deployments/${id}`),
  deactivate: (id: number) => api.delete(`/deployments/${id}`),
  predict: (deploymentId: number, features: Record<string, unknown> | Record<string, unknown>[]) =>
    api.post(`/deployments/${deploymentId}/predict`, { features }),
  history: (deploymentId: number) => api.get(`/deployments/${deploymentId}/predictions`),
  feedback: (predictionId: number, data: { is_correct: boolean; correct_value?: string; comment?: string }) =>
    api.post(`/deployments/predictions/${predictionId}/feedback`, data),
};

/* ─── Monitoring ────────────────────────────────────────────────────── */
export const monitoringApi = {
  get: (deploymentId: number) => api.get(`/monitoring/deployments/${deploymentId}`),
  checkDrift: (deploymentId: number) => api.post(`/monitoring/deployments/${deploymentId}/check-drift`),
};

/* ─── Chat ──────────────────────────────────────────────────────────── */
export const chatApi = {
  send: (projectId: number, message: string, history: { role: string; content: string }[]) =>
    api.post('/chat/', { project_id: projectId, message, conversation_history: history }),
  // history is maintained client-side; backend has no separate history endpoint
  history: (_projectId: number) => Promise.resolve({ data: { history: [] } }),
};

/* ─── Tasks ─────────────────────────────────────────────────────────── */
export const tasksApi = {
  get: (taskId: number) => api.get(`/tasks/${taskId}`),
  listByProject: (projectId: number) => api.get(`/tasks/project/${projectId}`),
};

/* ─── Knowledge & Experience ────────────────────────────────────────── */
export const knowledgeApi = {
  stats: () => api.get('/knowledge/stats'),
  algorithms: () => api.get('/knowledge/algorithms'),
  experiences: (limit = 50) => api.get(`/knowledge/experiences?limit=${limit}`),
  triggerSeed: () => api.post('/knowledge/seed'),
};

