/**
 * 模型切换 composable
 * 从 useApi.js 中提取的模型切换轮询逻辑
 */
import { ref } from 'vue'
import { API_BASE } from './useApi.js'

// 模块级共享状态
const switchingState = ref('idle')  // idle / switching / success / error
const switchingMessage = ref('')
const switchingError = ref('')
let switchPollTimer = null

export function useModelSwitch() {
  async function switchModel(payload) {
    /** 发起异步模型切换，并开始轮询切换状态 */
    try {
      const res = await fetch(`${API_BASE}/api/system/switch_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()

      if (data.status === 'switching_in_progress') {
        switchingState.value = 'switching'
        switchingMessage.value = data.message || '正在切换模型中...'
        startSwitchPolling()
        return false
      }

      if (data.status !== 'ok') {
        switchingState.value = 'error'
        switchingMessage.value = ''
        switchingError.value = data.message || '启动切换失败'
        return false
      }

      switchingState.value = 'switching'
      switchingMessage.value = data.message || '正在切换模型...'
      startSwitchPolling()
      return true
    } catch (err) {
      switchingState.value = 'error'
      switchingMessage.value = ''
      switchingError.value = err.message || '网络错误'
      return false
    }
  }

  async function pollSwitchStatus() {
    /** 轮询获取当前切换状态 */
    try {
      const res = await fetch(`${API_BASE}/api/system/switch_status`)
      const data = await res.json()

      if (data.status === 'idle' || !data.status) {
        if (switchingState.value !== 'idle') {
          switchingState.value = 'idle'
          switchingMessage.value = ''
          switchingError.value = ''
          stopSwitchPolling()
        }
        return
      }

      switchingState.value = data.status
      switchingMessage.value = data.message || ''

      if (data.status === 'error') {
        switchingError.value = data.error || '切换失败'
        stopSwitchPolling()
        setTimeout(() => {
          switchingState.value = 'idle'
          switchingMessage.value = ''
          switchingError.value = ''
        }, 8000)
      } else if (data.status === 'success') {
        switchingError.value = ''
        stopSwitchPolling()
        setTimeout(() => {
          switchingState.value = 'idle'
          switchingMessage.value = ''
          switchingError.value = ''
        }, 5000)
      }
    } catch (err) {
      // 轮询失败忽略，会继续重试
    }
  }

  function startSwitchPolling() {
    stopSwitchPolling()
    pollSwitchStatus()
    switchPollTimer = setInterval(pollSwitchStatus, 1500)
  }

  function stopSwitchPolling() {
    if (switchPollTimer) {
      clearInterval(switchPollTimer)
      switchPollTimer = null
    }
  }

  function resetSwitchState() {
    switchingState.value = 'idle'
    switchingMessage.value = ''
    switchingError.value = ''
    stopSwitchPolling()
  }

  return {
    switchingState,
    switchingMessage,
    switchingError,
    switchModel,
    pollSwitchStatus,
    startSwitchPolling,
    stopSwitchPolling,
    resetSwitchState,
  }
}
