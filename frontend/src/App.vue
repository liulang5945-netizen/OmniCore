<template>
  <div class="app-wrapper" @dragenter="onDragEnter" @dragleave="onDragLeave" @dragover.prevent @drop.prevent="onDrop">
    <ToastManager ref="toastRef" />
    <ConfirmDialog ref="confirmRef" />

    <!-- === Sidebar === -->
    <AppSidebar />

    <!-- === 全局模型切换状态指示条 === -->
    <div v-if="appStore.switchingState === 'switching'" class="global-switch-bar switching">
      <span class="spinner"></span> ⏳ {{ appStore.switchingMessage || '正在切换模型...' }}
    </div>
    <div v-else-if="appStore.switchingState === 'success'" class="global-switch-bar success">
      ✅ {{ appStore.switchingMessage || '模型切换成功' }}
    </div>
    <div v-else-if="appStore.switchingState === 'error'" class="global-switch-bar error">
      ❌ {{ appStore.switchingError || '模型切换失败' }}
    </div>

    <!-- === Router View === -->
    <div class="router-wrapper">
      <router-view v-slot="{ Component, route }">
        <keep-alive>
          <component :is="Component" :key="route.path" />
        </keep-alive>
      </router-view>
    </div>

    <UpgradeNotification />

    <div v-if="dragOver" class="global-drag-overlay">
      <div style="text-align:center;color:white;">
        <div style="font-size:3rem;margin-bottom:12px;">📥</div>
        <p style="font-size:1.2rem;color:rgba(255,255,255,0.9);">{{ appStore.t('drop_release') }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, provide } from 'vue'
import ToastManager from './components/ToastManager.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import AppSidebar from './components/AppSidebar.vue'
import UpgradeNotification from './components/UpgradeNotification.vue'
import { useAppStore } from './stores/appStore.js'
import { useChatStore } from './stores/chatStore.js'
import { useApi } from './composables/useApi.js'
import { useSettings } from './composables/useSettings.js'
import { loadCheckpoints, trainAbortController } from './composables/useTraining.js'

const appStore = useAppStore()
const chatStore = useChatStore()

// Toast & Confirm
const toastRef = ref(null)
const confirmRef = ref(null)
const toast = (msg, type = 'info') => { if (toastRef.value) toastRef.value.showToast(msg, type) }
const $confirm = (options) => confirmRef.value ? confirmRef.value.show(options) : Promise.resolve(false)
provide('toast', toast)
provide('$confirm', $confirm)

// API connection
const { startHealthCheck } = useApi()

// Drag
const dragOver = ref(false)
let dragCounter = 0
const onDragEnter = () => { dragCounter++; if (appStore.$route?.path !== '/kb' && appStore.$route?.path !== '/train') dragOver.value = true }
const onDragLeave = () => { dragCounter--; if (dragCounter <= 0) { dragCounter = 0; dragOver.value = false } }
const onDrop = () => { dragCounter = 0; dragOver.value = false }

// Model selection handler (exposed for child components)
provide('onModelSelected', (path) => {
  appStore.setModel(path)
  toast('✅ 模型已切换并生效', 'success')
})

// Lifecycle
onMounted(async () => {
  // 先从后端加载设置（系统提示词等），同步到 localStorage 和 Pinia store
  try {
    const { loadSettingsFromServer } = useSettings()
    const saved = await loadSettingsFromServer()
    if (saved && typeof saved === 'object') {
      for (const [key, value] of Object.entries(saved)) {
        const storageKey = `omnicore_${key}`
        if (value !== undefined && value !== null) {
          localStorage.setItem(storageKey, typeof value === 'string' ? value : JSON.stringify(value))
        }
      }
      // 恢复 UI 设置到 Pinia store（主题色、主题、搜索引擎、云端配置等）
      appStore.restoreUISettings(saved)
    }
  } catch (e) { /* 静默处理 */ }

  // 从后端加载历史会话，如果没有历史则创建新会话
  await chatStore.loadSessions()
  if (chatStore.sessions.length === 0) {
    chatStore.createNewSession()
  }

  startHealthCheck()
  loadCheckpoints()
})

onUnmounted(() => {
  if (trainAbortController) trainAbortController.abort()
})
</script>
<style>@import './assets/app.css';</style>