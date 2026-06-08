import { defineStore } from 'pinia'
import { ref, computed, nextTick } from 'vue'
import { API_BASE, setChatReceiving } from '@/composables/useApi.js'

export const useChatStore = defineStore('chat', () => {
  // === State ===
  const sessions = ref([])
  const currentSessionId = ref(null)
  const messages = ref([])
  const chatInput = ref('')
  const isLoading = ref(false)
  const isReceiving = ref(false)
  const lastEngineType = ref('')  // 记录最近一次使用的引擎类型
  const sessionsLoaded = ref(false) // 标记是否已从后端加载过
  let abortController = null

  // === Getters ===
  const currentSessionName = computed(() => {
    const s = sessions.value.find(s => s.id === currentSessionId.value)
    return s ? s.name : ''
  })

  // === Actions ===

  /**
   * 从后端加载所有历史会话
   */
  async function loadSessions() {
    try {
      const res = await fetch(`${API_BASE}/api/chat/sessions`)
      if (!res.ok) return
      const list = await res.json()
      if (!Array.isArray(list) || list.length === 0) {
        // 没有历史会话，标记已加载
        sessionsLoaded.value = true
        return
      }

      // 按 updated_at 降序排列
      list.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))

      const loaded = []
      for (const item of list) {
        try {
          const detailRes = await fetch(`${API_BASE}/api/chat/history/${item.session_id}`)
          if (!detailRes.ok) continue
          const detail = await detailRes.json()
          loaded.push({
            id: Number(item.session_id) || item.session_id,
            name: detail.name || item.name || '',
            messages: detail.messages || [],
          })
        } catch {
          // 单个会话加载失败，跳过
          continue
        }
      }

      if (loaded.length > 0) {
        sessions.value = loaded
        currentSessionId.value = loaded[0].id
        messages.value = [...(loaded[0].messages || [])]
      }
      sessionsLoaded.value = true
    } catch (e) {
      console.warn('[ChatStore] 加载历史会话失败:', e.message)
      sessionsLoaded.value = true
    }
  }

  function createNewSession() {
    const id = Date.now()
    sessions.value.unshift({
      id,
      name: `新对话 ${sessions.value.length + 1}`,
      messages: []
    })
    currentSessionId.value = id
    messages.value = []
  }

  function switchSession(id) {
    currentSessionId.value = id
    const s = sessions.value.find(s => s.id === id)
    messages.value = s ? (s.messages || []) : []
  }

  function deleteSession(id) {
    // 先从后端删除
    fetch(`${API_BASE}/api/chat/history/${id}`, { method: 'DELETE' }).catch(() => {})
    sessions.value = sessions.value.filter(s => s.id !== id)
    if (currentSessionId.value === id) {
      if (sessions.value.length) switchSession(sessions.value[0].id)
      else createNewSession()
    }
  }

  function clearCurrentChat() {
    messages.value = []
    const s = sessions.value.find(s => s.id === currentSessionId.value)
    if (s) s.messages = []
  }

  function setChatInput(val) {
    chatInput.value = val
  }

  function getSystemPrompt() {
    return localStorage.getItem('omnicore_system_prompt') || '你是一个全能助手。'
  }

  function getCloudBase() {
    return localStorage.getItem('omnicore_cloud_base') || ''
  }

  function getCloudKey() {
    return localStorage.getItem('omnicore_cloud_key') || ''
  }

  function getCloudModel() {
    return localStorage.getItem('omnicore_cloud_model') || ''
  }

  function getCloudType() {
    return localStorage.getItem('omnicore_cloud_type') || 'openai'
  }

  function getSearchEngine() {
    return localStorage.getItem('omnicore_search_engine') || 'DuckDuckGo'
  }

  function getSearchKey() {
    const engine = getSearchEngine()
    // 优先读取引擎专属 key，兼容旧版通用 key
    return localStorage.getItem(`omnicore_search_key_${engine}`) || localStorage.getItem('omnicore_search_key') || ''
  }

  async function sendMessage(engineType) {
    const input = chatInput.value.trim()
    if (!input || isLoading.value) return

    const s = sessions.value.find(s => s.id === currentSessionId.value)
    if (s && s.id !== sessions.value[0]?.id) {
      sessions.value = sessions.value.filter(x => x.id !== s.id)
      sessions.value.unshift(s)
    }

    messages.value.push({ role: 'user', content: input })
    if (s) s.messages = [...messages.value]
    chatInput.value = ''

    // 自动更新对话标题：如果是第一条用户消息且当前名称还是默认的"新对话"，则用用户输入前20个字符作为标题
    if (s && s.name && /^新对话\s*\d+$/.test(s.name) && messages.value.filter(m => m.role === 'user').length === 1) {
      const autoTitle = input.length > 20 ? input.slice(0, 20) + '…' : input
      s.name = autoTitle
      // 同步更新到后端（异步，不阻塞）
      fetch(`${API_BASE}/api/chat/history/${s.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: autoTitle, messages: s.messages }),
      }).catch(() => {})
    }
    isLoading.value = true
    isReceiving.value = false

    // 记录引擎类型供重新生成使用
    lastEngineType.value = engineType || 'local-chat'
    // 更新全局接收状态
    setChatReceiving(true)

    abortController = new AbortController()

    try {
      const history = []
      for (let i = 1; i < messages.value.length; i += 2) {
        if (messages.value[i]?.role === 'assistant') {
          history.push([messages.value[i - 1]?.content || '', messages.value[i].content || ''])
        }
      }

      const sysPrompt = getSystemPrompt()
      const cloudBase = getCloudBase()
      const cloudKey = getCloudKey()
      const cloudModel = getCloudModel()

      const res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: input,
          system_prompt: sysPrompt,
          history,
          engine: engineType || 'local-chat',
          api_base: cloudBase || undefined,
          api_key: cloudKey || undefined,
          api_model: cloudModel || undefined,
          api_type: getCloudType() || 'openai',
          search_engine: getSearchEngine(),
          search_key: getSearchKey() || undefined,
          agent_max_iterations: 10,
          agent_temperature: 0.7
        }),
        signal: abortController.signal,
      })

      if (!res.ok) {
        const e = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(e.detail || `HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      const aiMsg = { role: 'assistant', content: '' }
      messages.value.push(aiMsg)
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const p = line.slice(6)
            if (p === '[DONE]') {
              isReceiving.value = false
              break
            }
            if (p === '[START]') {
              aiMsg.content = ''
              continue
            }
            aiMsg.content += p.replace(/\\n/g, '\n')
            // 收到第一个数据 chunk 时标记正在接收
            if (!isReceiving.value) {
              isReceiving.value = true
            }
            messages.value[messages.value.length - 1] = { ...aiMsg }
          }
        }
      }

      if (s) s.messages = [...messages.value]
    } catch (err) {
      if (err.name !== 'AbortError') {
        const m = messages.value[messages.value.length - 1]
        if (m && m.role === 'assistant') {
          m.content += `\n\n❌ 错误: ${err.message}`
        } else {
          messages.value.push({ role: 'assistant', content: `❌ 错误: ${err.message}` })
        }
      }
    } finally {
      isLoading.value = false
      isReceiving.value = false
      abortController = null
      setChatReceiving(false)

      // 对话完成后自动保存到后端（异步，不阻塞）
      if (s && s.messages && s.messages.length > 0) {
        fetch(`${API_BASE}/api/chat/history/${s.id}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: s.name, messages: s.messages }),
        }).catch(() => {})
      }
    }
  }

  function stopGeneration() {
    if (abortController) abortController.abort()
    isLoading.value = false
    isReceiving.value = false
    setChatReceiving(false)
  }

  function regenerateMessage(idx) {
    if (idx > 0 && messages.value[idx - 1]?.role === 'user') {
      const m = messages.value[idx - 1].content
      messages.value.splice(idx - 1, 2)
      chatInput.value = m
      nextTick(() => sendMessage(lastEngineType.value || 'local-chat'))
    }
  }

  return {
    // State
    sessions,
    currentSessionId,
    messages,
    chatInput,
    isLoading,
    isReceiving,
    sessionsLoaded,
    // Getters
    currentSessionName,
    // Actions
    loadSessions,
    createNewSession,
    switchSession,
    deleteSession,
    clearCurrentChat,
    setChatInput,
    sendMessage,
    stopGeneration,
    regenerateMessage,
  }
})