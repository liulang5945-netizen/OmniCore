/**
 * 设置持久化 composable
 * 从 useApi.js 中提取的设置保存/加载逻辑
 */
import { API_BASE } from './useApi.js'

let _settingsSaveTimer = null

export function useSettings() {
  const saveSettingsToServer = async (settings) => {
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      })
    } catch (e) { /* silent fail */ }
  }

  const debouncedSaveSettings = (settings) => {
    if (_settingsSaveTimer) clearTimeout(_settingsSaveTimer)
    _settingsSaveTimer = setTimeout(() => saveSettingsToServer(settings), 2000)
  }

  const loadSettingsFromServer = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`)
      if (res.ok) {
        return await res.json()
      }
    } catch (e) {
      console.warn('[Settings] 服务端加载失败:', e.message)
    }
    return null
  }

  return {
    saveSettingsToServer,
    debouncedSaveSettings,
    loadSettingsFromServer,
  }
}
