<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <Brain class="logo-icon-svg" :size="22" />
        </div>
        <h2>{{ t('title') }}</h2>
        <MemoryStatusBar class="sidebar-memory-ring" @memory-warning="onMemoryWarning" />
      </div>
    </div>

    <button class="new-chat-btn" @click="handleNewChat">
      <Plus :size="16" />
      {{ t('new_chat') }}
    </button>

    <div class="session-list">
      <div v-for="session in chatStore.sessions" :key="session.id"
        :class="['session-item', { active: chatStore.currentSessionId === session.id }]"
        @click="chatStore.switchSession(session.id)">
        <span class="session-name">
          <MessageSquare :size="14" class="session-icon" />
          {{ session.name }}
        </span>
        <button class="session-del-btn" @click.stop="chatStore.deleteSession(session.id)" title="删除">
          <X :size="14" />
        </button>
      </div>
    </div>

    <div class="sidebar-footer">
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        custom v-slot="{ navigate, isActive }">
        <button class="settings-btn" :class="{ active: isActive }" @click="navigate">
          <span class="nav-icon-wrap">
            <component :is="item.icon" :size="14" class="nav-icon" />
          </span>
          <span class="nav-label">{{ item.label }}</span>
        </button>
      </router-link>

      <div class="sidebar-divider"></div>

      <button class="settings-btn restart-btn" @click="restartSystem">
        <span class="nav-icon-wrap">
          <RotateCcw :size="14" class="nav-icon" />
        </span>
        <span class="nav-label">{{ t('restart_system') }}</span>
      </button>
    </div>
  </aside>
</template>

<script setup>
import { computed, inject } from 'vue'
import {
  Brain, Plus, MessageSquare, X, RotateCcw,
  BookOpen, Zap, Cpu, Layout, Package, Settings
} from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { API_BASE } from '@/composables/useApi.js'
import MemoryStatusBar from './MemoryStatusBar.vue'

const toast = inject('toast', () => {})
let _memoryWarningTimeout = null

const chatStore = useChatStore()
const appStore = useAppStore()

const t = (key, params) => appStore.t(key, params)

const navItems = computed(() => [
  { path: '/', icon: MessageSquare, label: t('back') },
  { path: '/kb', icon: BookOpen, label: t('kb_management') },
  { path: '/train', icon: Zap, label: t('fine_tuning') },
  { path: '/agent', icon: Cpu, label: t('agent_config') },
  { path: '/workspace', icon: Layout, label: 'IDE' },
  { path: '/market', icon: Package, label: t('model_market') },
  { path: '/settings', icon: Settings, label: t('sys_settings') },
])

function handleNewChat() {
  chatStore.createNewSession()
}

async function restartSystem() {
  try {
    await fetch(`${API_BASE}/api/system/restart`, { method: 'POST' })
  } catch (e) {}
}

function onMemoryWarning(data) {
  if (_memoryWarningTimeout) return
  toast(
    `\u26A0\uFE0F 系统内存告急！可用仅 ${(data.available_pct * 100).toFixed(0)}% (${data.available_gb.toFixed(1)}GB)。请关闭其他应用或中断当前操作。`,
    'warning'
  )
  _memoryWarningTimeout = setTimeout(() => { _memoryWarningTimeout = null }, 30000)
}
</script>

<style scoped>
.logo-icon-wrap {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-gradient);
  border-radius: 12px;
  color: white;
  box-shadow: 0 4px 14px rgba(99,102,241,0.3);
  flex-shrink: 0;
  transition: var(--transition);
}

.logo-icon-wrap:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

.nav-icon-wrap {
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  flex-shrink: 0;
  transition: var(--transition-fast);
  background: var(--bg-muted);
  color: var(--text-muted);
}

.settings-btn.active .nav-icon-wrap {
  background: var(--primary);
  color: white;
  box-shadow: 0 3px 10px rgba(99,102,241,0.25);
}

.settings-btn:hover .nav-icon-wrap {
  background: var(--primary-light);
  color: var(--primary);
  transform: scale(1.08);
}

.settings-btn.active:hover .nav-icon-wrap {
  background: var(--primary);
  color: white;
}

.sidebar-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent);
  margin: 8px 0;
}

.nav-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>