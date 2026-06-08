<template>
  <main class="main-content">
    <header class="chat-header">
      <div class="header-left">
        <div class="header-title">{{ chatStore.currentSessionName || t('new_chat') }}</div>
      </div>
      <div class="header-controls">
        <select :value="engineModel" @change="setEngine($event.target.value)" class="engine-select">
          <option value="local-chat">
            <Monitor /> {{ t('engine_local') }}
          </option>
          <option value="cloud-chat">
            <Cloud /> {{ t('engine_cloud') }}
          </option>
          <option value="local-agent">
            <Cpu /> {{ t('engine_agent_local') }}
          </option>
          <option value="cloud-agent">
            <Globe /> {{ t('engine_agent_cloud') }}
          </option>
        </select>
        <select v-if="engineModel.includes('agent')" :value="searchEngine" @change="setSearchEngine($event.target.value)" class="engine-select search-select" :title="t('search_engine')">
          <option value="DuckDuckGo">DuckDuckGo</option>
          <option value="Baidu">Baidu</option>
          <option value="Bing">Bing</option>
          <option value="Serper">Serper</option>
          <option value="Tavily">Tavily</option>
          <option value="smart-multi">{{ t('multi_search') }}</option>
        </select>
        <span :class="['status', appStore.connectionClass]" :title="appStore.connectionErrorMsg">{{ appStore.connectionStatus }}</span>
        <select :value="appStore.currentLang" @change="setLang($event.target.value)" class="lang-select">
          <option value="zh">中文</option>
          <option value="en">EN</option>
        </select>
      </div>
    </header>

    <div class="messages-area" ref="messagesArea">
      <div class="chat-thread">
        <!-- Empty state -->
        <div v-if="chatStore.messages.length === 0" class="empty-state">
          <div class="empty-logo"><Brain :size="64" /></div>
          <h1>{{ t('welcome_title') }}</h1>
          <p>{{ t('welcome_desc') }}</p>
          <div class="quick-actions">
            <button class="quick-btn" @click="chatStore.setChatInput(t('quick_code_prompt'))">
              <Code :size="16" class="quick-icon" /> {{ t('quick_code') }}
            </button>
            <button class="quick-btn" @click="chatStore.setChatInput(t('quick_explain_prompt'))">
              <HelpCircle :size="16" class="quick-icon" /> {{ t('quick_explain') }}
            </button>
            <button class="quick-btn" @click="chatStore.setChatInput(t('quick_news_prompt'))">
              <Newspaper :size="16" class="quick-icon" /> {{ t('quick_news') }}
            </button>
          </div>
        </div>

        <!-- Messages -->
        <div v-for="(msg, index) in chatStore.messages" :key="index" :class="['message-wrapper', msg.role]">
          <div :class="['avatar', msg.role === 'user' ? 'user-avatar' : 'ai-avatar']">
            <User v-if="msg.role === 'user'" :size="20" />
            <Bot v-else :size="20" />
          </div>
          <div class="message-content">
            <div class="bubble">
              <div v-if="msg.role === 'user'" class="text-content">{{ msg.content }}</div>
              <div v-else>
                <!-- Assistant message with reasoning folding -->
                <div v-if="parsedMessages[index] && parsedMessages[index].reasoning" class="reasoning-block">
                  <button class="reasoning-toggle" @click="toggleReasoning(index)">
                    <ChevronDown v-if="expandedReasonings[index]" :size="14" class="reasoning-icon" />
                    <ChevronRight v-else :size="14" class="reasoning-icon" />
                    <span class="reasoning-label">{{ t('thinking') }}</span>
                    <span class="reasoning-hint">{{ expandedReasonings[index] ? t('collapse_thinking') : t('expand_thinking') }}</span>
                  </button>
                  <div v-show="expandedReasonings[index]" class="reasoning-content markdown-body" v-html="renderMarkdown(parsedMessages[index].reasoning)"></div>
                </div>
                <div class="markdown-body" v-html="renderMarkdown(parsedMessages[index] ? parsedMessages[index].content : msg.content)"></div>
              </div>
            </div>
            <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
              <button class="msg-action-btn" @click="copyMsg(msg.content)" :title="t('copy')">
                <Copy :size="14" />
              </button>
              <button class="msg-action-btn" @click="chatStore.regenerateMessage(index)" :title="t('regenerate')">
                <RotateCcw :size="14" />
              </button>
            </div>
          </div>
        </div>

        <!-- Loading indicator -->
        <div v-if="chatStore.isLoading && !chatStore.isReceiving" class="message-wrapper assistant">
          <div class="avatar ai-avatar">
            <Bot :size="20" />
          </div>
          <div class="bubble loading-bubble">
            <span class="typing-dots"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span>
          </div>
        </div>
      </div>
    </div>

    <!-- Input area -->
    <div class="input-area">
      <div class="stop-container" v-if="chatStore.isReceiving">
        <button class="stop-btn" @click="chatStore.stopGeneration()">
          <Square :size="14" fill="currentColor" />
          {{ t('stop_generation') }}
        </button>
      </div>
      <!-- Attached files preview -->
      <div v-if="chatAttachments.length" class="chat-attachments-row">
        <div v-for="(att, idx) in chatAttachments" :key="idx" class="chat-attachment-chip">
          <ImageIcon v-if="att.isImage" :size="14" class="att-icon" />
          <FileText v-else :size="14" class="att-icon" />
          <span class="att-name">{{ att.name }}</span>
          <button class="att-remove" @click="removeAttachment(idx)">
            <X :size="12" />
          </button>
        </div>
      </div>
      <div class="input-container">
        <label class="attach-btn" :title="t('attach_file')">
          <Paperclip :size="18" />
          <input type="file" multiple @change="onFileSelect" style="display:none" />
        </label>
        <textarea
          ref="inputRef"
          v-model="chatStore.chatInput"
          :placeholder="t('placeholder')"
          rows="1"
          @keydown="onKeydown"
          @input="autoResize"
        ></textarea>
        <button class="send-btn" @click="handleSend" :disabled="!chatStore.chatInput.trim() || chatStore.isLoading">
          <Send :size="18" />
        </button>
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, inject } from 'vue'
import {
  Brain, Plus, Copy, RotateCcw, Send, Square, User, Bot,
  ChevronDown, ChevronRight, HelpCircle, Newspaper, Code,
  Monitor, Cloud, Cpu, Globe, Paperclip, FileText, ImageIcon, X
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'
import { useChatUpload } from '@/composables/useChatUpload.js'
import { useSettings } from '@/composables/useSettings.js'

const chatStore = useChatStore()
const appStore = useAppStore()
const { renderMarkdown } = useMarkdown()
const toast = inject('toast', () => {})

const t = (key, params) => appStore.t(key, params)

// Refs
const messagesArea = ref(null)
const inputRef = ref(null)

// Engine state
const engineModel = computed(() => appStore.currentEngine)
const searchEngine = ref(localStorage.getItem('omnicore_search_engine') || 'DuckDuckGo')

function setEngine(val) {
  appStore.setEngine(val)
}

function setSearchEngine(val) {
  searchEngine.value = val
  localStorage.setItem('omnicore_search_engine', val)
  // 同步到后端持久化
  const { saveSettingsToServer } = useSettings()
  saveSettingsToServer({ search_engine: val }).catch(() => {})
}

function setLang(val) {
  appStore.currentLang = val
  localStorage.setItem('omnicore_lang', val)
  // 同步到后端持久化
  const { saveSettingsToServer } = useSettings()
  saveSettingsToServer({ lang: val }).catch(() => {})
}

// Reasoning parsing
const parsedMessages = ref({})
const expandedReasonings = ref({})

function parseMessageContent(content) {
  if (!content) return { content: '', reasoning: '' }
  // Support <think>...</think> and <thinking>...<</thinking> patterns
  const thinkMatch = content.match(/<(?:think|thinking)>([\s\S]*?)<\/(?:think|thinking)>/)
  if (thinkMatch) {
    const reasoning = thinkMatch[1].trim()
    const rest = content.replace(thinkMatch[0], '').trim()
    return { content: rest, reasoning }
  }
  return { content, reasoning: '' }
}

function toggleReasoning(index) {
  expandedReasonings.value[index] = !expandedReasonings.value[index]
}

// Watch messages to parse reasoning
watch(() => chatStore.messages, (msgs) => {
  const parsed = {}
  msgs.forEach((msg, i) => {
    if (msg.role === 'assistant') {
      parsed[i] = parseMessageContent(msg.content)
    }
  })
  parsedMessages.value = parsed
}, { deep: true, immediate: true })

// File attachments
const { chatAttachments, onFileSelect, removeAttachment } = useChatUpload()

// Auto scroll
watch(() => chatStore.messages.length, () => {
  nextTick(() => {
    if (messagesArea.value) {
      messagesArea.value.scrollTop = messagesArea.value.scrollHeight
    }
  })
})

watch(() => chatStore.isReceiving, () => {
  nextTick(() => {
    if (messagesArea.value) {
      messagesArea.value.scrollTop = messagesArea.value.scrollHeight
    }
  })
})

// Textarea auto resize
function autoResize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 150) + 'px'
}

// Send message
function handleSend() {
  const engine = engineModel.value
  chatStore.sendMessage(engine)
  nextTick(() => {
    autoResize()
    if (messagesArea.value) {
      messagesArea.value.scrollTop = messagesArea.value.scrollHeight
    }
  })
}

// Keyboard
function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

// Copy message
async function copyMsg(content) {
  try {
    await navigator.clipboard.writeText(content)
    toast(t('copy') + ' ✓', 'success')
  } catch {
    toast('Copy failed', 'error')
  }
}

// Scroll to bottom on mount
onMounted(() => {
  nextTick(() => {
    if (messagesArea.value) {
      messagesArea.value.scrollTop = messagesArea.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.quick-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 20px;
  max-width: 520px;
}

.quick-btn {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  padding: 14px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: 0.84rem;
  color: var(--text-secondary);
  transition: var(--transition);
  font-family: inherit;
  text-align: left;
}

.quick-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
  background: var(--primary-light);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.quick-icon {
  color: var(--primary);
}

.empty-logo {
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-gradient);
  border-radius: 20px;
  color: white;
  box-shadow: 0 8px 32px rgba(99,102,241,0.25);
  margin-bottom: 20px;
}

.empty-state h1 {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 8px;
  letter-spacing: -0.03em;
}

.empty-state p {
  color: var(--text-muted);
  font-size: 0.92rem;
  margin: 0;
  max-width: 400px;
  line-height: 1.7;
  text-align: center;
}

.header-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Avatar */
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.user-avatar {
  background: var(--primary-gradient);
  color: white;
  box-shadow: 0 3px 10px rgba(99,102,241,0.25);
}

.ai-avatar {
  background: var(--primary-light);
  border: 1px solid rgba(99,102,241,0.15);
  color: var(--primary);
  box-shadow: 0 0 12px rgba(99,102,241,0.1);
}

/* Reasoning block */
.reasoning-block {
  margin-bottom: 10px;
}

.reasoning-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--primary-subtle);
  border: 1px solid rgba(99,102,241,0.12);
  cursor: pointer;
  color: var(--primary);
  font-size: 0.78rem;
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  font-family: inherit;
  transition: var(--transition-fast);
  width: 100%;
}

.reasoning-toggle:hover {
  background: var(--primary-light);
}

.reasoning-icon {
  flex-shrink: 0;
}

.reasoning-label {
  font-weight: 600;
}

.reasoning-hint {
  font-size: 0.72rem;
  opacity: 0.6;
  margin-left: auto;
}

.reasoning-content {
  padding: 10px 14px;
  margin: 6px 0 10px;
  background: var(--bg-muted);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--primary);
  font-size: 0.82rem;
  color: var(--text-secondary);
  max-height: 300px;
  overflow-y: auto;
  line-height: 1.6;
  transition: max-height 0.3s ease;
}

/* Message actions */
.msg-actions {
  display: flex;
  gap: 4px;
  margin-top: 6px;
  opacity: 0;
  transform: translateY(2px);
  transition: opacity 0.25s ease, transform 0.25s ease;
}

.message-wrapper:hover .msg-actions {
  opacity: 0.7;
  transform: translateY(0);
}

.msg-actions:hover {
  opacity: 1 !important;
}

.msg-action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  cursor: pointer;
  color: var(--text-muted);
  padding: 5px;
  border-radius: 6px;
  transition: var(--transition-fast);
}

.msg-action-btn:hover {
  background: var(--primary-subtle);
  color: var(--primary);
}

/* Typing dots */
.typing-dots .dot {
  animation: blink 1.4s infinite both;
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--primary);
}

.typing-dots .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-dots .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  0%, 80%, 100% { opacity: 0; }
  40% { opacity: 1; }
}

.loading-bubble {
  padding: 12px 20px !important;
}

/* Search select */
.search-select {
  max-width: 120px;
}

/* Responsive */
@media (max-width: 640px) {
  .quick-actions {
    grid-template-columns: 1fr;
  }
}
</style>