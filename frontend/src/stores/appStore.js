import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { locales } from '@/locales/index.js'

// 内联 API_BASE（避免与 useApi.js 循环依赖）
const _API_BASE = import.meta.env.DEV
  ? ''
  : `${window.location.protocol}//${window.location.hostname}:8000`

// 防抖保存 UI 设置到后端
let _uiSaveTimer = null
function _debouncedSaveUI(data) {
  if (_uiSaveTimer) clearTimeout(_uiSaveTimer)
  _uiSaveTimer = setTimeout(async () => {
    try {
      await fetch(`${_API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
    } catch (e) { /* silent fail */ }
  }, 1000)
}

export const useAppStore = defineStore('app', () => {
  // === State ===
  const currentEngine = ref(localStorage.getItem('omnicore_engine') || 'local-chat')
  const currentModel = ref(localStorage.getItem('omnicore_model') || '')
  const currentTheme = ref(localStorage.getItem('omnicore_theme') || 'auto')
  const currentAccent = ref(localStorage.getItem('omnicore_accent') || '')
  const currentBgImage = ref(localStorage.getItem('omnicore_bg_image') || '')
  const currentLang = ref('zh')
  const showWorkspace = ref(false)

  // 连接状态
  const connectionState = ref('unknown')
  const connectionErrorMsg = ref('')
  const downloadProgress = ref(null)
  const retryCountdown = ref(0)
  const modelLoaded = ref(false)
  let consecutiveFailures = 0

  // 模型切换状态
  const switchingState = ref('idle')
  const switchingMessage = ref('')
  const switchingError = ref('')

  // === Getters ===
  const connectionClass = computed(() => {
    if (connectionState.value === 'connected') return 'connected'
    if (connectionState.value === 'downloading') return 'downloading'
    if (connectionState.value === 'loading') return 'loading'
    if (connectionState.value === 'connecting') return 'connecting'
    return 'error'
  })

  const connectionStatus = computed(() => {
    if (connectionState.value === 'connected') return modelLoaded.value ? t('status_connected') : t('status_connected_no_model')
    if (connectionState.value === 'downloading') {
      const dp = downloadProgress.value
      if (dp && dp.percent > 0) {
        const dl = dp.downloadedMb || 0
        const total = dp.totalMb || 0
        const sizeStr = total > 0 ? ` ${dl.toFixed(0)}/${total.toFixed(0)} MB` : ''
        return `📥 下载模型 ${dp.percent.toFixed(0)}%${sizeStr}`
      }
      return '📥 正在下载模型...'
    }
    if (connectionState.value === 'connecting') return t('status_connecting')
    if (connectionState.value === 'loading') {
      if (retryCountdown.value > 0) return t('retry', { n: retryCountdown.value })
      return t('status_model_loading')
    }
    return t('status_error')
  })

  // === Helpers ===
  function t(key, params = {}) {
    let text = locales[currentLang.value][key] || locales['zh'][key] || key
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, v)
    }
    return text
  }

  // === Actions ===
  function setEngine(engine) {
    currentEngine.value = engine
    localStorage.setItem('omnicore_engine', engine)
  }

  function setModel(path, modelType) {
    currentModel.value = path
    localStorage.setItem('omnicore_model', path)
    if (modelType) {
      localStorage.setItem('omnicore_model_type', modelType)
    } else {
      const isGGUF = path.toLowerCase().endsWith('.gguf')
      localStorage.setItem('omnicore_model_type', isGGUF ? 'gguf' : 'huggingface')
    }
  }

  function toggleWorkspace() {
    showWorkspace.value = !showWorkspace.value
  }

  function setConnectionState(state, errorMsg = '', loaded = false) {
    connectionState.value = state
    connectionErrorMsg.value = errorMsg
    modelLoaded.value = loaded
  }

  function setDownloadProgress(progress) {
    downloadProgress.value = progress
  }

  function resetSwitchState() {
    switchingState.value = 'idle'
    switchingMessage.value = ''
    switchingError.value = ''
  }

  // 预设主题色
  const accentPresets = [
    { name: '默认灰蓝', color: '#5b7a8a' },
    { name: '靛蓝', color: '#4f46e5' },
    { name: '翠绿', color: '#059669' },
    { name: '琥珀', color: '#d97706' },
    { name: '玫瑰', color: '#e11d48' },
    { name: '紫色', color: '#7c3aed' },
    { name: '青色', color: '#0891b2' },
    { name: '石墨', color: '#475569' },
  ]

  function applyTheme() {
    const r = document.documentElement
    r.classList.remove('theme-dark', 'theme-light')
    if (currentTheme.value === 'dark') {
      r.classList.add('theme-dark')
      r.setAttribute('data-theme', 'dark')
    } else if (currentTheme.value === 'light') {
      r.classList.add('theme-light')
      r.setAttribute('data-theme', 'light')
    } else {
      r.removeAttribute('data-theme')
    }
    // 应用自定义主题色
    applyAccent()
    // 应用背景图
    applyBgImage()
  }

  function applyAccent() {
    const hex = currentAccent.value
    if (!hex) return
    const r = document.documentElement
    // 计算衍生色
    const rgb = hexToRgb(hex)
    if (!rgb) return
    r.style.setProperty('--primary', hex)
    r.style.setProperty('--primary-hover', darken(hex, 15))
    r.style.setProperty('--primary-light', `rgba(${rgb.r},${rgb.g},${rgb.b},0.08)`)
    r.style.setProperty('--primary-subtle', `rgba(${rgb.r},${rgb.g},${rgb.b},0.04)`)
    r.style.setProperty('--primary-gradient', `linear-gradient(135deg, ${hex} 0%, ${lighten(hex, 20)} 100%)`)
  }

  function applyBgImage() {
    const wrapper = document.querySelector('.app-wrapper')
    if (!wrapper) return
    if (currentBgImage.value) {
      wrapper.style.backgroundImage = `url(${currentBgImage.value})`
      wrapper.style.backgroundSize = 'cover'
      wrapper.style.backgroundPosition = 'center'
      wrapper.style.backgroundAttachment = 'fixed'
    } else {
      wrapper.style.backgroundImage = ''
    }
  }

  function hexToRgb(hex) {
    const m = hex.replace('#', '').match(/^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)
    return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : null
  }

  function darken(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = 1 - pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c * f).toString(16).padStart(2, '0')).join('')
  }

  function lighten(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c + (255 - c) * f).toString(16).padStart(2, '0')).join('')
  }

  function setTheme(theme) {
    currentTheme.value = theme
    localStorage.setItem('omnicore_theme', theme)
    applyTheme()
    _debouncedSaveUI({ theme })
  }

  function setAccent(color) {
    currentAccent.value = color
    localStorage.setItem('omnicore_accent', color)
    applyAccent()
    _debouncedSaveUI({ accent: color })
  }

  function setBgImage(dataUrl) {
    currentBgImage.value = dataUrl
    if (dataUrl) {
      localStorage.setItem('omnicore_bg_image', dataUrl)
    } else {
      localStorage.removeItem('omnicore_bg_image')
    }
    applyBgImage()
    // 背景图是 base64 data URL（可能几MB），不存后端，仅存 localStorage
  }

  /**
   * 从后端设置中恢复 UI 相关配置
   * 在 App.vue onMounted 中调用，将后端持久化的设置同步到 Pinia store
   */
  function restoreUISettings(serverSettings) {
    if (!serverSettings || typeof serverSettings !== 'object') return
    let needsApplyTheme = false

    // 恢复主题色
    if (serverSettings.accent !== undefined) {
      const accentVal = serverSettings.accent
      currentAccent.value = accentVal
      localStorage.setItem('omnicore_accent', accentVal)
      needsApplyTheme = true
    }

    // 恢复主题
    if (serverSettings.theme !== undefined) {
      currentTheme.value = serverSettings.theme
      localStorage.setItem('omnicore_theme', serverSettings.theme)
      needsApplyTheme = true
    }

    // 恢复语言
    if (serverSettings.lang !== undefined) {
      currentLang.value = serverSettings.lang
      localStorage.setItem('omnicore_lang', serverSettings.lang)
    }

    // 恢复搜索引擎配置
    if (serverSettings.search_engine !== undefined) {
      localStorage.setItem('omnicore_search_engine', serverSettings.search_engine)
    }
    if (serverSettings.search_key !== undefined) {
      localStorage.setItem('omnicore_search_key', serverSettings.search_key)
      // 同时存引擎专属 key
      const eng = serverSettings.search_engine || localStorage.getItem('omnicore_search_engine')
      if (eng) localStorage.setItem(`omnicore_search_key_${eng}`, serverSettings.search_key)
    }

    // 恢复云端API配置
    if (serverSettings.cloud_profiles !== undefined) {
      localStorage.setItem('omnicore_cloud_profiles', JSON.stringify(serverSettings.cloud_profiles))
    }
    if (serverSettings.cloud_active !== undefined) {
      localStorage.setItem('omnicore_cloud_active', serverSettings.cloud_active)
    }
    if (serverSettings.cloud_type !== undefined) {
      localStorage.setItem('omnicore_cloud_type', serverSettings.cloud_type)
    }
    if (serverSettings.cloud_base !== undefined) {
      localStorage.setItem('omnicore_cloud_base', serverSettings.cloud_base)
    }
    if (serverSettings.cloud_key !== undefined) {
      localStorage.setItem('omnicore_cloud_key', serverSettings.cloud_key)
    }
    if (serverSettings.cloud_model !== undefined) {
      localStorage.setItem('omnicore_cloud_model', serverSettings.cloud_model)
    }

    // 统一应用主题（如果有变化）
    if (needsApplyTheme) {
      applyTheme()
    }
  }

  // 初始化主题
  applyTheme()

  return {
    // State
    currentEngine,
    currentModel,
    currentTheme,
    currentAccent,
    currentBgImage,
    currentLang,
    showWorkspace,
    connectionState,
    connectionErrorMsg,
    downloadProgress,
    retryCountdown,
    modelLoaded,
    switchingState,
    switchingMessage,
    switchingError,
    // Getters
    connectionClass,
    connectionStatus,
    // Actions
    t,
    setEngine,
    setModel,
    toggleWorkspace,
    setConnectionState,
    setDownloadProgress,
    resetSwitchState,
    applyTheme,
    setTheme,
    setAccent,
    setBgImage,
    accentPresets,
    restoreUISettings,
  }
})