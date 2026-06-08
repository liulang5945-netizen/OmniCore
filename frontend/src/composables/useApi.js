/**
 * API 连接与健康检查 composable
 * 状态提取到模块级别，确保所有调用方（App.vue / ChatView.vue）共享同一份响应式状态
 *
 * 子模块拆分:
 *   - useModelSwitch.js → 模型切换轮询逻辑
 *   - useSettings.js    → 设置持久化逻辑
 */
import { ref, computed } from 'vue';
import { useModelSwitch } from './useModelSwitch.js';
import { useSettings } from './useSettings.js';
import { useAppStore } from '../stores/appStore.js';

/**
 * 动态 API 基地址：
 * - 开发环境：Vite proxy 拦截 /api，使用空字符串（相对路径）
 * - 生产环境：自动使用当前页面的 host，支持局域网/公网访问
 *   （不再硬编码 127.0.0.1，避免远程访问时跨域指向浏览器本机）
 */
export const API_BASE = import.meta.env.DEV
  ? ''  // Vite dev server proxy handles /api → 127.0.0.1:8000
  : `${window.location.protocol}//${window.location.hostname}:8000`;

// === 模块级共享状态 ===
const connectionState = ref('unknown');
const connectionErrorMsg = ref('');
const retryCountdown = ref(0);
const downloadProgress = ref(null);
const currentLang = ref('zh');
const taijiAvailable = ref(false); // 态极模块是否可用（公众版本为 false）
const modelLoaded = ref(false); // 模型是否已加载
let healthCheckTimer = null;
let retryTimer = null;
let consecutiveFailures = 0;
// 聊天/推理进行中标志
let isChatReceiving = false;

/**
 * 模块级函数：设置聊天接收状态
 * 可直接 import { setChatReceiving } from '@/composables/useApi.js'
 */
export function setChatReceiving(receiving) {
  isChatReceiving = receiving;
}

export function useApi() {
  const appStore = useAppStore();
  const t = (key, params = {}) => appStore.t(key, params);

  const connectionClass = computed(() => {
    if (connectionState.value === 'connected') return 'connected';
    if (connectionState.value === 'downloading') return 'downloading';
    if (connectionState.value === 'loading') return 'loading';
    if (connectionState.value === 'connecting') return 'connecting';
    return 'error';
  });

  const connectionStatus = computed(() => {
    if (connectionState.value === 'connected') return modelLoaded.value ? t('status_connected') : t('status_connected_no_model');
    if (connectionState.value === 'downloading') {
      const dp = downloadProgress.value;
      if (dp && dp.percent > 0) {
        const dl = dp.downloadedMb || 0;
        const total = dp.totalMb || 0;
        const sizeStr = total > 0 ? ` ${dl.toFixed(0)}/${total.toFixed(0)} MB` : '';
        return `📥 下载模型 ${dp.percent.toFixed(0)}%${sizeStr}`;
      }
      return '📥 正在下载模型...';
    }
    if (connectionState.value === 'connecting') return t('status_connecting');
    if (connectionState.value === 'loading') {
      if (retryCountdown.value > 0) return t('retry', { n: retryCountdown.value });
      return t('status_model_loading');
    }
    return t('status_error');
  });

  // 同步 appStore 连接状态（ChatView.vue 读取 appStore.connectionStatus）
  function syncAppState(state, msg = '') {
    connectionState.value = state;
    connectionErrorMsg.value = msg;
    appStore.setConnectionState(state, msg, modelLoaded.value);
  }

  async function checkHealth() {
    try {
      const controller = new AbortController();
      // 健康检查超时 15 秒：后端加载大模型或执行批量 I/O 时 5 秒容易误判
      const timeout = 15000;
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      const resp = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!resp.ok) {
        syncAppState('error', `后端返回错误 (HTTP ${resp.status})`);
        return false;
      }

      const ctype = resp.headers.get('content-type') || '';
      if (!ctype.includes('application/json')) {
        syncAppState('error', '后端返回了非JSON响应，可能正在启动中...');
        return false;
      }

      const data = await resp.json();
      consecutiveFailures = 0;
      if (data.status === 'ok') {
        modelLoaded.value = !!data.model_loaded;
        syncAppState('connected');
        downloadProgress.value = null;
        retryCountdown.value = 0;
        taijiAvailable.value = !!data.taiji_available;
        clearRetryTimer();
        return true;
      } else if (data.status === 'downloading') {
        syncAppState('downloading', data.message || '正在下载模型...');
        downloadProgress.value = {
          percent: data.percent || 0,
          message: data.message || '正在下载模型...',
          totalMb: data.total_mb || 0,
          downloadedMb: data.downloaded_mb || 0,
        };
        return false;
      } else if (data.status === 'loading') {
        consecutiveFailures = 0;
        if (connectionState.value !== 'connected' && connectionState.value !== 'downloading') {
          syncAppState('loading', data.message || '模型正在加载中...');
        } else {
          connectionErrorMsg.value = data.message || '模型正在加载中...';
        }
        return false;
      } else {
        syncAppState('error', data.message || '后端报告错误');
        return false;
      }
    } catch (err) {
      if (connectionState.value === 'loading') {
        consecutiveFailures = 0;
        return false;
      }
      if (connectionState.value === 'unknown' || connectionState.value === 'connecting') {
        syncAppState('connecting', '正在连接后端服务...');
        consecutiveFailures = 0;
        return false;
      }
      if (isChatReceiving) {
        return false;
      }
      consecutiveFailures++;
      if (consecutiveFailures >= 5) {
        syncAppState('error', err.message);
      }
      return false;
    }
  }

  function startHealthCheck() {
    consecutiveFailures = 0;
    checkHealth();
    healthCheckTimer = setInterval(async () => {
      const connected = await checkHealth();
      if (connected) {
        clearInterval(healthCheckTimer);
        healthCheckTimer = setInterval(checkHealth, 10000);
      }
    }, 2000);
  }

  function stopHealthCheck() {
    if (healthCheckTimer) {
      clearInterval(healthCheckTimer);
      healthCheckTimer = null;
    }
    clearRetryTimer();
  }

  function clearRetryTimer() {
    if (retryTimer) { clearInterval(retryTimer); retryTimer = null; }
  }

  // 委托给子模块（向后兼容：通过 useApi() 仍可访问）
  const modelSwitch = useModelSwitch();
  const settings = useSettings();

  return {
    API_BASE,
    connectionState, connectionErrorMsg, retryCountdown, downloadProgress,
    connectionClass, connectionStatus, currentLang,
    // 模型切换（来自 useModelSwitch）
    switchingState: modelSwitch.switchingState,
    switchingMessage: modelSwitch.switchingMessage,
    switchingError: modelSwitch.switchingError,
    switchModel: modelSwitch.switchModel,
    pollSwitchStatus: modelSwitch.pollSwitchStatus,
    startSwitchPolling: modelSwitch.startSwitchPolling,
    stopSwitchPolling: modelSwitch.stopSwitchPolling,
    resetSwitchState: modelSwitch.resetSwitchState,
    // 设置持久化（来自 useSettings）
    saveSettingsToServer: settings.saveSettingsToServer,
    debouncedSaveSettings: settings.debouncedSaveSettings,
    loadSettingsFromServer: settings.loadSettingsFromServer,
    // 健康检查
    t,
    checkHealth, startHealthCheck, stopHealthCheck,
    setChatReceiving,
    taijiAvailable, modelLoaded,
  };
}