import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? ''
  : 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  login: (credentials) => api.post('/login', credentials),
  signup: (userData) => api.post('/signup', userData),
  getProfile: () => api.get('/profile'),
  logout: () => api.post('/logout'),
};

export const profileAPI = {
  update: (data) => api.post('/update_profile', data),
};

export const chatAPI = {
  query: (question) => api.post('/query', { query: question }),
  getTopics: () => api.get('/topics'),
};

export const hrAPI = {
  generateQuestions: () => api.post('/hr_questions'),
  saveSession: (data) => api.post('/save_interview_session', data),
  getHistory: (type) => api.get(`/interview_history?type=${type}`),
};

export const resumeAPI = {
  upload: (formData) => api.post('/upload_resume', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  generateResumeBasedQuestions: (data) => api.post('/resume_based_questions', data),
};

export const statsAPI = {
  getUserStats: () => api.get('/user_stats'),
};

export const healthAPI = {
  check: () => api.get('/health'),
};

export const audioAPI = {
  uploadAudio: (formData) => api.post('/process_audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
};

// ✅ UPDATED - Progress tracking API with all subtopic methods
export const progressAPI = {
  // Basic progress
  getUserProgress: () => api.get('/user/progress'),
  getTopicDetails: (topic) => api.get(`/user/topics/${topic}/details`),

  // 🔥 NEW: Subtopic mastery endpoints
  getSubtopicStats: () => api.get('/user/subtopic_stats'),
  getTopicSubtopics: (topic) => api.get(`/user/subtopics/${topic}`),
  resetMastery: (topic = null) => api.post('/user/reset_mastery', { topic }),

  // 🔥 NEW: Get questions for a specific subtopic
  getSubtopicQuestions: (topic, subtopic) =>
    api.get(`/user/subtopic/${topic}/${encodeURIComponent(subtopic)}/questions`),
};

export default api;