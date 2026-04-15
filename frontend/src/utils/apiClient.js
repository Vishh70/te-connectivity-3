import axios from 'axios';

// In development, use the Vite proxy (empty base URL = same origin).
// In production, reads from VITE_API_BASE_URL env variable.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

// Create a centralized Axios instance
const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor to attach the JWT token to every request
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Interceptor to handle global authentication errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // If the backend says unauthorized, clear the token and force login
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('jwt_token');
      // Dispatch a custom event so App.jsx knows to show the login screen
      window.dispatchEvent(new Event('auth-unauthorized'));
    }
    return Promise.reject(error);
  }
);

export default apiClient;

export const getWsUrl = (path) => {
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const token = localStorage.getItem("jwt_token");

  // Senior Pro Fix: Use relative origin to leverage the Vite proxy (stabilizes WebSocket connectivity)
  const host = window.location.host;
  
  if (!token) return `${wsProtocol}//${host}${path}`;
  
  const separator = path.includes("?") ? "&" : "?";
  return `${wsProtocol}//${host}${path}${separator}token=${encodeURIComponent(token)}`;
};
