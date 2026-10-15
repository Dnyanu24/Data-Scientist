import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8001/api/v1', // Bypasses Vite proxy and connects straight to FastAPI
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
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
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

/* ─── Datasets ─── */
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
  getSimilar: (id: number, topK = 5) => api.get(`/datasets/${id}/similar`, { params: { top_k: topK } }),
};

/* ─── Recommendations & Preprocessing ─── */
export const recommendationsApi = {
  // POST /api/v1/projects/{project_id}/recommend
  trigger: (projectId: number) =>
    api.post(`/projects/${projectId}/recommend`),

  // GET /api/v1/projects/{project_id}/recommendations
  list: (projectId: number) =>
    api.get(`/projects/${projectId}/recommendations`),

  // POST /api/v1/projects/{project_id}/preprocessing/generate
  generatePreprocessing: (projectId: number) =>
    api.post(`/projects/${projectId}/preprocessing/generate`),

  // GET /api/v1/projects/{project_id}/preprocessing
  getPreprocessing: (projectId: number) =>
    api.get(`/projects/${projectId}/preprocessing`),

  // PUT /api/v1/projects/{project_id}/preprocessing
  updatePreprocessing: (projectId: number, data: Record<string, unknown>) =>
    api.put(`/projects/${projectId}/preprocessing`, data),
};

/* ─── Experiments / Training ─── */
export const experimentsApi = {
  // If Swagger (/docs) shows /api/v1/projects/{projectId}/leaderboard:
  leaderboard: (projectId: number) =>
    api.get(`/projects/${projectId}/leaderboard`),

  // If Swagger shows /api/v1/projects/{projectId}/train:
  startTraining: (projectId: number, config?: Record<string, unknown>) =>
    api.post(`/projects/${projectId}/train`, config || {}),

  // If Swagger shows /api/v1/projects/{projectId}/experiments:
  listExperiments: (projectId: number) =>
    api.get(`/projects/${projectId}/experiments`),

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
  predict: (deploymentId: number | string, features: Record<string, unknown> | Record<string, unknown>[]) =>
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
export interface TaskStatusResponse {
  id: number;
  project_id?: number;
  celery_task_id?: string;
  task_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  project_phase?: string;
  result?: any;
  error?: string;
  created_at?: string;
  completed_at?: string;
}

export interface PollTaskOptions {
  onProgress?: (task: TaskStatusResponse) => void;
  onDone?: (task?: TaskStatusResponse) => void;
  loadProject?: () => Promise<any> | void;
  onError?: (error: any) => void;
  maxConsecutiveErrors?: number;
  intervalMs?: number;
}

export const pollTask = (
  taskId: number,
  callbacksOrDone?: (() => void) | PollTaskOptions,
  onProgressCallback?: (task: TaskStatusResponse) => void
): (() => void) => {
  const options: PollTaskOptions =
    typeof callbacksOrDone === 'function'
      ? { onDone: callbacksOrDone, onProgress: onProgressCallback }
      : callbacksOrDone || {};

  let isSubscribed = true;
  let consecutiveErrors = 0;
  const maxErrors = options.maxConsecutiveErrors ?? 5;
  const intervalMs = options.intervalMs ?? 2000;

  const intervalId = setInterval(async () => {
    if (!isSubscribed) {
      clearInterval(intervalId);
      return;
    }

    try {
      const r = await tasksApi.get(taskId);
      consecutiveErrors = 0; // Reset consecutive glitches
      const t: TaskStatusResponse = r.data;

      if (options.onProgress) {
        try {
          options.onProgress(t);
        } catch (err) {
          console.warn('[pollTask] onProgress error:', err);
        }
      }

      if (t.status === 'completed' || t.status === 'failed') {
        clearInterval(intervalId);
        isSubscribed = false;

        if (t.status === 'completed') {
          // 1. Automatically trigger project state reload if provided
          if (options.loadProject) {
            try {
              await options.loadProject();
            } catch (err) {
              console.warn('[pollTask] loadProject failed gracefully:', err);
            }
          }

          // 2. Safely execute onDone callback (ignoring/logging any 400 from dependent calls)
          if (options.onDone) {
            try {
              await options.onDone(t);
            } catch (err) {
              console.warn('[pollTask] onDone dependent call failed gracefully (e.g., /similar 400):', err);
            }
          }
        } else if (t.status === 'failed') {
          if (options.onError) {
            options.onError(new Error(t.message || 'Task failed'));
          }
        }
      }
    } catch (err) {
      consecutiveErrors++;
      console.warn(`[pollTask] Transient network glitch polling task ${taskId} (${consecutiveErrors}/${maxErrors}):`, err);
      // Do NOT stop polling on minor network glitches until max errors reached
      if (consecutiveErrors >= maxErrors) {
        clearInterval(intervalId);
        isSubscribed = false;
        if (options.onError) {
          options.onError(err);
        }
      }
    }
  }, intervalMs);

  return () => {
    isSubscribed = false;
    clearInterval(intervalId);
  };
};

export const tasksApi = {
  get: (taskId: number) => api.get(`/tasks/${taskId}`),
  listByProject: (projectId: number) => api.get(`/tasks/project/${projectId}`),
  poll: pollTask,
};

/* ─── Knowledge & Experience ────────────────────────────────────────── */
export const knowledgeApi = {
  stats: () => api.get('/knowledge/stats'),
  algorithms: () => api.get('/knowledge/algorithms'),
  experiences: (limit = 50) => api.get(`/knowledge/experiences?limit=${limit}`),
  triggerSeed: () => api.post('/knowledge/seed'),
};

