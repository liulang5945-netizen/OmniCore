/**
 * 模型下载 composable
 * 从 ModelMarket.vue 中提取的下载/暂停/恢复/取消/轮询逻辑
 */
import { ref } from 'vue'
import { API_BASE } from './useApi.js'

export function useModelDownload(allModels, loadDownloadedModels, loadModels) {
  const downloading = ref({})

  // HF 下载状态使用独立 key，避免与 GGUF 下载冲突
  function hfKey(model) { return 'hf_' + model.key }

  function stopPolling(key) {
    if (downloading.value[key] && downloading.value[key]._pollInterval) {
      clearInterval(downloading.value[key]._pollInterval)
      downloading.value[key]._pollInterval = null
    }
  }

  function keepFor(key, ms) {
    setTimeout(() => {
      if (downloading.value[key] && (downloading.value[key]._isCancelled || downloading.value[key]._isError)) {
        delete downloading.value[key]
      }
    }, ms)
  }

  // GGUF 下载
  async function downloadModel(model, emit) {
    const key = model.key
    if (downloading.value[key] && downloading.value[key]._taskId) return

    if (downloading.value[key] && downloading.value[key]._pollInterval) {
      clearInterval(downloading.value[key]._pollInterval)
    }

    downloading.value[key] = { progress: '0%', _taskId: null, _isError: false, _isCancelled: false, _isPaused: false, _isHF: false, _errorDetail: '', _pollInterval: null, _percent: 0 }
    try {
      const res = await fetch(`${API_BASE}/api/models/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: key,
          quant: model.quant,
          save_dir: '',
          verify_ssl: true
        })
      })
      const data = await res.json()
      if (data.status !== 'ok') {
        const errMsg = data.message || data.detail || 'Download failed'
        downloading.value[key]._isError = true
        downloading.value[key]._errorDetail = errMsg
        downloading.value[key].progress = '❌ ' + (errMsg.length > 200 ? errMsg.substring(0, 200) + '...' : errMsg)
        setTimeout(() => { delete downloading.value[key] }, 30000)
        return
      }
      const taskId = data.task_id
      downloading.value[key]._taskId = taskId

      const pollInterval = setInterval(async () => {
        if (downloading.value[key]._isPaused) return
        try {
          const pRes = await fetch(`${API_BASE}/api/models/download_progress?task_id=${encodeURIComponent(taskId)}`)
          const pData = await pRes.json()
          const prog = pData.progress
          if (!prog) return

          const pct = prog.percent || 0
          const totalMb = prog.total_mb || 0
          const dlMb = prog.downloaded_mb || 0
          const spd = typeof prog.speed_mbps === 'number' ? prog.speed_mbps : 0
          const speed = spd > 0 ? ` (${spd.toFixed(1)} MB/s)` : (dlMb > 0 ? ' (计算中...)' : '')
          const eta = prog.eta_seconds > 0 ? ` | 剩余 ${Math.round(prog.eta_seconds / 60)} 分钟` : ''
          const sizeInfo = totalMb > 0 ? ` | ${dlMb.toFixed(1)}/${totalMb.toFixed(1)} MB` : ''
          const statusText = prog.status === 'paused' ? ' (已暂停)' : (prog.status === 'downloading' && pct === 0 && dlMb === 0 ? ' (准备中...)' : '')
          downloading.value[key].progress = `${pct}%${sizeInfo}${speed}${eta}${statusText}`
          downloading.value[key]._percent = pct

          if (prog.status === 'paused') {
            downloading.value[key]._isPaused = true
          } else if (downloading.value[key]._isPaused && prog.status === 'downloading') {
            downloading.value[key]._isPaused = false
          }

          if (prog.status === 'completed') {
            clearInterval(pollInterval)
            downloading.value[key]._pollInterval = null
            downloading.value[key].progress = '✅ 下载完成'
            downloading.value[key]._isError = false; downloading.value[key]._isCancelled = false; downloading.value[key]._isPaused = false
            const found = allModels.value.find(m => m.key === key)
            if (found) found.downloaded = true
            if (pData.progress.file_path && emit) emit('select-model', pData.progress.file_path)
            loadDownloadedModels()
            setTimeout(() => { delete downloading.value[key] }, 5000)
          } else if (prog.status === 'error') {
            clearInterval(pollInterval)
            downloading.value[key]._pollInterval = null
            const errMsg = prog.error_message || '下载失败'
            downloading.value[key]._isError = true; downloading.value[key]._isPaused = false
            downloading.value[key]._errorDetail = errMsg
            downloading.value[key].progress = '❌ ' + (errMsg.length > 200 ? errMsg.substring(0, 200) + '...' : errMsg)
            const found = allModels.value.find(m => m.key === key)
            if (found) found.downloaded = false
            keepFor(key, 30000)
          } else if (prog.status === 'cancelled') {
            clearInterval(pollInterval)
            downloading.value[key]._pollInterval = null
            downloading.value[key]._isCancelled = true; downloading.value[key]._isError = false; downloading.value[key]._isPaused = false
            downloading.value[key].progress = '⏹ 已取消'
            keepFor(key, 10000)
          }
        } catch { /* 轮询异常忽略 */ }
      }, 1000)

      downloading.value[key]._pollInterval = pollInterval

      // 安全超时（60分钟后强制停止轮询）
      setTimeout(() => {
        clearInterval(pollInterval)
        if (downloading.value[key] && downloading.value[key]._taskId === taskId) {
          downloading.value[key]._isError = true
          downloading.value[key]._errorDetail = '下载超时（超过60分钟），请检查网络连接后重试'
          downloading.value[key].progress = '❌ 下载超时'
          downloading.value[key]._pollInterval = null
        }
      }, 60 * 60 * 1000)

    } catch (err) {
      downloading.value[key]._isError = true
      downloading.value[key]._errorDetail = err.message || '下载失败'
      downloading.value[key].progress = '❌ ' + (err.message || '下载失败').substring(0, 60)
      setTimeout(() => { delete downloading.value[key] }, 30000)
    }
  }

  // HF 下载
  async function downloadHFModel(model, t) {
    const repo = model.hf_train_repo
    if (!repo) return
    const key = hfKey(model)
    if (downloading.value[key] && downloading.value[key]._taskId) {
      alert((t ? t('downloading') : '下载中') + '... ' + (t ? t('wait_for_download') : '请等待'))
      return
    }
    stopPolling(key)
    downloading.value[key] = {
      _isError: false, _isCancelled: false, _isPaused: false, _isHF: true,
      _percent: 0, _taskId: null, _pollInterval: null,
      progress: '🔬 ' + (t ? t('downloading') : '下载中') + ' HF... 0%',
    }
    try {
      const res = await fetch(`${API_BASE}/api/models/download_hf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: model.name, quant: model.quant || 'Q4_K_M' }),
      })
      const data = await res.json()
      if (data.status !== 'ok') {
        downloading.value[key]._isError = true
        downloading.value[key].progress = '❌ HF: ' + (data.message || 'download failed')
        downloading.value[key]._errorDetail = data.message || ''
        return
      }
      const taskId = data.task_id
      downloading.value[key]._taskId = taskId

      const poll = setInterval(async () => {
        if (downloading.value[key]._isPaused) return
        try {
          const pRes = await fetch(`${API_BASE}/api/models/download_progress?task_id=${encodeURIComponent(taskId)}`)
          const pData = await pRes.json()
          const p = pData.progress || {}
          const pct = Math.min(100, Math.round((p.percent || 0) * 100) / 100)
          downloading.value[key]._percent = pct
          const dl = (p.downloaded_mb || 0).toFixed(0)
          const tot = (p.total_mb || 0).toFixed(0)
          const spd = typeof p.speed_mbps === 'number' ? p.speed_mbps : 0
          const speedStr = spd > 0 ? ' ' + spd.toFixed(1) + ' MB/s' : ''
          const eta = p.eta_seconds > 0 ? ' ~' + Math.round(p.eta_seconds / 60) + 'min' : ''
          downloading.value[key].progress = '🔬 HF: ' + dl + '/' + tot + ' MB' + speedStr + eta

          if (pData.status === 'completed' || p.status === 'completed') {
            clearInterval(poll)
            downloading.value[key]._pollInterval = null
            downloading.value[key]._percent = 100
            downloading.value[key].progress = '✅ HF model ready — ' + (t ? t('restart_to_finetune') : '重启后即可微调')
            loadDownloadedModels()
            setTimeout(() => { delete downloading.value[key]; loadModels() }, 3000)
          } else if (pData.status === 'error' || p.status === 'error') {
            clearInterval(poll)
            downloading.value[key]._pollInterval = null
            downloading.value[key]._isError = true
            downloading.value[key].progress = '❌ HF: ' + (pData.error_message || p.error_message || 'download error')
            downloading.value[key]._errorDetail = pData.error_message || p.error_message || ''
          } else if (pData.status === 'cancelled' || p.status === 'cancelled') {
            clearInterval(poll)
            downloading.value[key]._pollInterval = null
            downloading.value[key]._isCancelled = true
            downloading.value[key].progress = '⏹ ' + (t ? t('cancelled') : '已取消')
            keepFor(key, 10000)
          }
        } catch { /* continue */ }
      }, 2000)
      downloading.value[key]._pollInterval = poll

      // 安全超时
      setTimeout(() => {
        stopPolling(key)
        if (downloading.value[key] && downloading.value[key]._taskId === taskId && !downloading.value[key]._isError && !downloading.value[key]._isCancelled) {
          downloading.value[key]._isError = true
          downloading.value[key]._errorDetail = '下载超时（超过60分钟）'
          downloading.value[key].progress = '❌ 下载超时'
        }
      }, 60 * 60 * 1000)
    } catch (err) {
      downloading.value[key]._isError = true
      downloading.value[key].progress = '❌ HF: ' + err.message
      downloading.value[key]._errorDetail = err.message
    }
  }

  // 取消 GGUF
  async function cancelDownload(model) {
    const key = model.key
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    try {
      await fetch(`${API_BASE}/api/models/download_cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: dl._taskId })
      })
      stopPolling(key)
      dl._isCancelled = true; dl._isError = false; dl._isPaused = false
      dl.progress = '⏹ 已取消'
      keepFor(key, 10000)
    } catch (err) { console.warn('[ModelDownload] 取消下载失败:', err.message) }
  }

  // 取消 HF
  async function cancelHFDownload(model) {
    const key = hfKey(model)
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    try {
      await fetch(`${API_BASE}/api/models/download_cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: dl._taskId })
      })
      stopPolling(key)
      dl._isCancelled = true; dl._isError = false; dl._isPaused = false
      dl.progress = '⏹ 已取消'
      keepFor(key, 10000)
    } catch (err) { console.warn('[ModelDownload] 取消HF下载失败:', err.message) }
  }

  // 暂停 GGUF
  async function pauseDownload(model) {
    const key = model.key
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    await fetch(`${API_BASE}/api/models/download_pause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: dl._taskId })
    })
    dl._isPaused = true
    dl.progress = '⏸ ' + dl.progress.replace(/^⏸\s*/, '')
  }

  // 恢复 GGUF
  async function resumeDownload(model) {
    const key = model.key
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    await fetch(`${API_BASE}/api/models/download_resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: dl._taskId })
    })
    dl._isPaused = false
  }

  // 暂停 HF
  async function pauseHFDownload(model) {
    const key = hfKey(model)
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    await fetch(`${API_BASE}/api/models/download_pause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: dl._taskId })
    })
    dl._isPaused = true
    dl.progress = '⏸ ' + dl.progress.replace(/^⏸\s*/, '')
  }

  // 恢复 HF
  async function resumeHFDownload(model) {
    const key = hfKey(model)
    const dl = downloading.value[key]
    if (!dl || !dl._taskId) return
    await fetch(`${API_BASE}/api/models/download_resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: dl._taskId })
    })
    dl._isPaused = false
  }

  // 重试 GGUF
  async function retryDownload(model) {
    const key = model.key
    stopPolling(key)
    delete downloading.value[key]
    await downloadModel(model)
  }

  // 重试 HF
  async function retryHFDownload(model) {
    const key = hfKey(model)
    stopPolling(key)
    delete downloading.value[key]
    await downloadHFModel(model)
  }

  return {
    downloading,
    hfKey,
    downloadModel,
    downloadHFModel,
    cancelDownload,
    cancelHFDownload,
    pauseDownload,
    resumeDownload,
    pauseHFDownload,
    resumeHFDownload,
    retryDownload,
    retryHFDownload,
    stopPolling,
  }
}
