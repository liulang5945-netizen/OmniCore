/**
 * 认证管理 composable
 * 提供登录、登出、Token 管理、自动附加认证头等功能
 */
import { ref, computed, onMounted } from 'vue';
import { API_BASE } from './useApi.js';

const token = ref(localStorage.getItem('jwt_token') || '');
const authEnabled = ref(false);
const username = ref('');
const authLoaded = ref(false);

export function useAuth() {
  const isAuthenticated = computed(() => !authEnabled.value || !!token.value);

  async function checkAuthStatus() {
    try {
      const r = await fetch(`${API_BASE}/api/auth/status`);
      if (r.ok) {
        const d = await r.json();
        authEnabled.value = d.enabled || false;
        username.value = d.username || '';
      }
    } catch (e) { /* ignore */ }
    authLoaded.value = true;
  }

  async function login(user, password) {
    const r = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password })
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || '登录失败');
    }
    const d = await r.json();
    token.value = d.token;
    localStorage.setItem('jwt_token', d.token);
    return d;
  }

  function logout() {
    token.value = '';
    localStorage.removeItem('jwt_token');
  }

  function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (token.value) {
      headers['Authorization'] = `Bearer ${token.value}`;
    }
    return headers;
  }

  async function enableAuth(user, password) {
    const r = await fetch(`${API_BASE}/api/auth/enable`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ username: user, password })
    });
    if (r.ok) {
      authEnabled.value = true;
      username.value = user;
    }
    return r.ok;
  }

  async function disableAuth() {
    const r = await fetch(`${API_BASE}/api/auth/disable`, {
      method: 'POST',
      headers: getAuthHeaders()
    });
    if (r.ok) {
      authEnabled.value = false;
      logout();
    }
    return r.ok;
  }

  async function changePassword(oldPwd, newPwd) {
    const r = await fetch(`${API_BASE}/api/auth/change_password`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ old_password: oldPwd, new_password: newPwd })
    });
    return r.ok;
  }

  // 处理 401 响应（全局拦截）
  function handleAuthError(response) {
    if (response.status === 401) {
      logout();
      return true;
    }
    return false;
  }

  // 初始化时检查
  checkAuthStatus();

  return {
    token,
    authEnabled,
    username,
    authLoaded,
    isAuthenticated,
    login,
    logout,
    getAuthHeaders,
    enableAuth,
    disableAuth,
    changePassword,
    handleAuthError,
    checkAuthStatus,
  };
}