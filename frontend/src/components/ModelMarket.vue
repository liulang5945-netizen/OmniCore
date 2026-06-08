<template>
  <section class="dedicated-view">
  <div class="model-market view-body">
    <!-- 硬件检测面板 -->
    <div class="hw-panel">
      <div class="hw-header" @click="showHardware = !showHardware">
        <span class="hw-title"><MonitorIcon :size="16" /> {{ t('hardware_info') }}</span>
        <span class="hw-toggle">{{ showHardware ? '▼' : '▶' }}</span>
      </div>
      <div v-if="showHardware" class="hw-details">
        <div v-if="hwLoading" class="hw-loading">{{ t('detecting') }}</div>
        <div v-else-if="hwError" class="hw-error">{{ hwError }}</div>
        <div v-else class="hw-grid">
          <div class="hw-item"><span class="hw-label">CPU:</span><span>{{ hwInfo.cpu || 'N/A' }}</span></div>
          <div class="hw-item"><span class="hw-label">RAM:</span><span>{{ hwInfo.ram || 'N/A' }}</span></div>
          <div class="hw-item"><span class="hw-label">GPU:</span><span>{{ hwInfo.gpu || t('no_gpu') }}</span></div>
          <div class="hw-item"><span class="hw-label">VRAM:</span><span>{{ hwInfo.vram || 'N/A' }}</span></div>
          <div class="hw-item"><span class="hw-label">{{ t('recommend') }}:</span><span class="hw-recommend">{{ hwRecommend }}</span></div>
        </div>
        <button class="btn-detect" @click="detectHardware" :disabled="hwLoading">{{ hwLoading ? t('detecting') : t('redetect') }}</button>
        <button class="btn-detect" style="margin-left: 8px;" @click="runNetworkDiagnose" :disabled="networkDiagLoading">
          {{ networkDiagLoading ? t('diagnosing_network') : t('network_diagnose') }}
        </button>
        <div v-if="networkDiagResult" class="network-diag-result">
          <div class="diag-title">{{ t('network_diag_result') }}:</div>
          <pre class="diag-text">{{ networkDiagResult }}</pre>
        </div>
      </div>
    </div>

    <!-- 搜索和过滤 -->
    <div class="market-toolbar">
      <input 
        v-model="searchQuery" 
        class="market-search" 
        :placeholder="t('search_placeholder')"
      />
      <select v-model="sizeFilter" class="market-filter">
        <option value="all">{{ t('filter_size_all') }}</option>
        <option value="tiny">≤3B</option>
        <option value="small">3B-8B</option>
        <option value="medium">8B-14B</option>
        <option value="large">14B-34B</option>
        <option value="xl">≥34B</option>
      </select>
      <select v-model="quantFilter" class="market-filter">
        <option value="all">{{ t('filter_quant_all') }}</option>
        <option value="Q2_K">Q2_K (2-bit)</option>
        <option value="Q4_K_M">Q4_K_M (4-bit)</option>
        <option value="Q5_K_M">Q5_K_M (5-bit)</option>
        <option value="Q8_0">Q8_0 (8-bit)</option>
        <option value="F16">F16 (16-bit)</option>
      </select>
      <div class="market-tabs">
        <button :class="['tab-btn', { active: tab === 'recommended' }]" @click="tab = 'recommended'">{{ t('tab_recommended') }}</button>
        <button :class="['tab-btn', { active: tab === 'all' }]" @click="tab = 'all'">{{ t('tab_all') }}</button>
        <button :class="['tab-btn', { active: tab === 'downloaded' }]" @click="tab = 'downloaded'; loadDownloadedModels()">{{ t('tab_downloaded') }}</button>
      </div>
    </div>

    <!-- 模型列表 -->
    <div class="market-list" v-if="tab !== 'empty'">
      <div v-if="marketLoading" class="market-status">{{ t('loading_models') }}</div>
      <div v-else-if="filteredModels.length === 0 && allModels.length === 0" class="market-status market-empty">
        <Package class="empty-icon" :size="48" />
        <p>{{ t('no_models') }}</p>
        <button class="btn-retry" @click="loadModels">{{ t('retry') }}</button>
      </div>
      <div v-else-if="filteredModels.length === 0" class="market-status market-empty">
        <Search class="empty-icon" :size="48" />
        <p>{{ t('no_results') }}</p>
        <p class="empty-hint">{{ t('try_adjust_filter') }}</p>
      </div>
      <div v-else class="model-cards">
        <div 
          v-for="model in filteredModels" 
          :key="model.key" 
          :class="['model-card', { recommended: model.recommended, downloaded: model.downloaded }]"
          @mouseenter="model._hover = true" @mouseleave="model._hover = false"
        >
          <div class="mc-header">
            <span class="mc-name">{{ model.name }}</span>
            <span v-if="model.recommended" class="mc-badge rec">{{ t('rec_badge') }}</span>
            <span v-if="model.downloaded" class="mc-badge dl">✓ {{ t('downloaded_badge') }}</span>
          </div>
          <div class="mc-meta">
            <span class="mc-tag" v-if="model.file_size_display"><Package :size="12" /> {{ model.file_size_display }}</span>
            <span class="mc-tag">{{ model.params_b ? model.params_b + 'B' : formatSize(model.size) }}</span>
            <span class="mc-tag">{{ model.quant || 'BF16' }}</span>
            <span class="mc-tag">{{ model.family || '' }}</span>
            <span class="mc-tag">{{ model.task || '' }}</span>
          </div>
          <div class="mc-desc">{{ model.desc || '' }}</div>
          <div class="mc-storage-row" v-if="model.downloaded && model.path">
            <span class="mc-path">{{ model.path }}</span>
          </div>
          <div class="mc-actions">
            <button class="btn-model info-btn" @click="showModelDetail(model)">{{ t('detail') }}</button>
            
            <!-- GGUF 下载状态 (model.key) -->
            <template v-if="downloading[model.key] && !downloading[model.key]._isHF">
              <div class="mc-progress-bar-wrapper" v-if="!downloading[model.key]._isError && !downloading[model.key]._isCancelled && !downloading[model.key]._isPaused">
                <div class="mc-progress-bar">
                  <div class="mc-progress-fill" :style="{ width: (downloading[model.key]._percent || 0) + '%' }"></div>
                </div>
              </div>
              <div class="mc-dl-progress">
                <span class="spinner" v-if="!downloading[model.key]._isError && !downloading[model.key]._isCancelled && !downloading[model.key]._isPaused"></span>
                <span v-else class="mc-error-icon"><component :is="downloading[model.key]._isCancelled ? Square : downloading[model.key]._isPaused ? Pause : X" :size="14" /></span>
                <span class="dl-progress-text" :class="{ 'dl-error-text': downloading[model.key]._isError, 'dl-cancelled-text': downloading[model.key]._isCancelled, 'dl-paused-text': downloading[model.key]._isPaused }">
                  {{ downloading[model.key].progress }}
                </span>
              </div>
              <div class="mc-dl-buttons">
                <button v-if="downloading[model.key]._taskId && !downloading[model.key]._isPaused && !downloading[model.key]._isError && !downloading[model.key]._isCancelled"
                  class="btn-model pause-btn" @click="pauseDownload(model)"><Square :size="12" /> {{ t('pause_download') }}</button>
                <button v-if="downloading[model.key]._isPaused"
                  class="btn-model resume-btn" @click="resumeDownload(model)"><Play :size="12" /> {{ t('resume_download') }}</button>
                <button v-if="downloading[model.key]._taskId && !downloading[model.key]._isError && !downloading[model.key]._isCancelled"
                  class="btn-model cancel-btn" @click="cancelDownload(model)">{{ t('cancel_download') }}</button>
                <button v-if="downloading[model.key]._isError || downloading[model.key]._isCancelled"
                  class="btn-model retry-btn" @click="retryDownload(model)">{{ t('retry_download') }}</button>
              </div>
            </template>
            
            <!-- HF 可微调版下载状态 (hfKey) -->
            <template v-if="downloading[hfKey(model)]">
              <div class="mc-progress-bar-wrapper" v-if="!downloading[hfKey(model)]._isError && !downloading[hfKey(model)]._isCancelled && !downloading[hfKey(model)]._isPaused">
                <div class="mc-progress-bar">
                  <div class="mc-progress-fill" :style="{ width: (downloading[hfKey(model)]._percent || 0) + '%' }"></div>
                </div>
              </div>
              <div class="mc-dl-progress">
                <span class="spinner" v-if="!downloading[hfKey(model)]._isError && !downloading[hfKey(model)]._isCancelled && !downloading[hfKey(model)]._isPaused"></span>
                <span v-else class="mc-error-icon"><component :is="downloading[hfKey(model)]._isCancelled ? Square : downloading[hfKey(model)]._isPaused ? Pause : X" :size="14" /></span>
                <span class="dl-progress-text" :class="{ 'dl-error-text': downloading[hfKey(model)]._isError, 'dl-cancelled-text': downloading[hfKey(model)]._isCancelled, 'dl-paused-text': downloading[hfKey(model)]._isPaused }">
                  {{ downloading[hfKey(model)].progress }}
                </span>
              </div>
              <div class="mc-dl-buttons">
                <button v-if="downloading[hfKey(model)]._taskId && !downloading[hfKey(model)]._isPaused && !downloading[hfKey(model)]._isError && !downloading[hfKey(model)]._isCancelled"
                  class="btn-model pause-btn" @click="pauseHFDownload(model)"><Square :size="12" /> {{ t('pause_download') }}</button>
                <button v-if="downloading[hfKey(model)]._isPaused"
                  class="btn-model resume-btn" @click="resumeHFDownload(model)"><Play :size="12" /> {{ t('resume_download') }}</button>
                <button v-if="downloading[hfKey(model)]._taskId && !downloading[hfKey(model)]._isError && !downloading[hfKey(model)]._isCancelled"
                  class="btn-model cancel-btn" @click="cancelHFDownload(model)">{{ t('cancel_download') }}</button>
                <button v-if="downloading[hfKey(model)]._isError || downloading[hfKey(model)]._isCancelled"
                  class="btn-model retry-btn" @click="retryHFDownload(model)">{{ t('retry_download') }}</button>
              </div>
            </template>
            
            <template v-else-if="model.downloaded">
              <button class="btn-model done">✓ {{ t('already_downloaded') }}</button>
              <button v-if="model.path" class="btn-model apply-btn" @click="applyModel(model)">{{ t('apply_model') }}</button>
            </template>
            
            <button v-else class="btn-model download" @click="downloadModel(model)">{{ t('one_click_download') }}</button>

            <!-- HF 可微调版下载按钮 -->
            <button v-if="model.hf_train_repo" class="btn-model hf-btn" @click="downloadHFModel(model)"><Microscope :size="12" /> {{ t('download_for_finetune') }}</button>
          </div>
          
          <!-- GGUF 错误详情 -->
          <div v-if="downloading[model.key] && downloading[model.key]._isError && downloading[model.key]._errorDetail && model._hover" class="mc-error-detail">
            <div class="error-detail-title"><Search :size="12" /> {{ t('error_detail') }}:</div>
            <pre class="error-detail-text">{{ downloading[model.key]._errorDetail }}</pre>
          </div>
          <!-- HF 错误详情 -->
          <div v-if="downloading[hfKey(model)] && downloading[hfKey(model)]._isError && downloading[hfKey(model)]._errorDetail && model._hover" class="mc-error-detail">
            <div class="error-detail-title">🔍 {{ t('error_detail') }}:</div>
            <pre class="error-detail-text">{{ downloading[hfKey(model)]._errorDetail }}</pre>
          </div>
        </div>
      </div>
    </div>

    <!-- 已下载模型列表 -->
    <div v-if="tab === 'downloaded'" class="downloaded-list">
      <div v-if="downloadedModels.length === 0" class="market-status">{{ t('no_downloaded') }}</div>
      <div v-else>
        <div v-for="m in downloadedModels" :key="m.path" class="downloaded-item">
          <span class="dl-name">{{ m.model_name || m.name || m.path }}</span>
          <span class="dl-type" :class="m.model_type === 'self' ? 'type-self' : (m.model_type === 'huggingface' ? 'type-hf' : 'type-gguf')">
            {{ m.model_type === 'self' ? '🧬态极' : (m.model_type === 'huggingface' ? 'HF可微调' : 'GGUF推理') }}
          </span>
          <span class="dl-size" v-if="m.size_gb">({{ m.size_gb }} GB)</span>
          <span class="dl-path">{{ m.path }}</span>
          <button class="btn-model small" @click="selectModel(m.path, m.model_type)">{{ t('use_this') }}</button>
          <button class="btn-model small danger" @click="deleteModel(m)">{{ t('delete') }}</button>
        </div>
      </div>
    </div>

    <!-- 模型详情弹窗 -->
    <div v-if="detailModel" class="modal-overlay" @click.self="detailModel = null">
      <div class="modal-detail">
        <h3>{{ detailModel.name }}</h3>
        <div class="detail-grid">
          <div><strong>{{ t('param_count') }}:</strong> {{ detailModel.params_b ? detailModel.params_b + 'B' : formatSize(detailModel.size) }}</div>
          <div v-if="detailModel.file_size_display"><strong>{{ t('file_size') }}:</strong> {{ detailModel.file_size_display }}</div>
          <div><strong>{{ t('quantization') }}:</strong> {{ detailModel.quant || 'BF16' }}</div>
          <div><strong>{{ t('family') }}:</strong> {{ detailModel.family || '-' }}</div>
          <div><strong>{{ t('task_type') }}:</strong> {{ detailModel.task || '-' }}</div>
          <div><strong>{{ t('memory_estimate') }}:</strong> {{ detailModel.memory_estimate || '-' }}</div>
          <div><strong>HF:</strong> {{ detailModel.hf_id || detailModel.key }}</div>
        </div>
        <div class="detail-desc">{{ detailModel.desc || '' }}</div>
        <div class="detail-actions">
          <button v-if="!detailModel.downloaded" class="btn-model download" @click="downloadModel(detailModel)">{{ t('one_click_download') }}</button>
          <button class="btn-model" @click="detailModel = null">{{ t('close') }}</button>
        </div>
      </div>
    </div>
  </div>
  </section>
</template>

<script setup>
import { Package, Search, Square, Pause, X, Monitor as MonitorIcon, Microscope, Play, StopCircle, AlertCircle, ChevronDown, ChevronRight } from 'lucide-vue-next';

import { ref, computed, onMounted, onUnmounted, inject } from 'vue'
import { API_BASE, useApi } from '../composables/useApi.js'
import { useModelDownload } from '../composables/useModelDownload.js'
import { modelMarketLocales } from '../locales/modelMarket.js'
import { useAppStore } from '../stores/appStore.js'
const { switchingState, switchingMessage, switchingError, switchModel } = useApi();
const toast = inject('toast');
const $confirm = inject('$confirm');
const onModelSelected = inject('onModelSelected', null);
const appStore = useAppStore();

const locales = modelMarketLocales

const currentLang = computed(() => appStore.currentLang || 'zh')
const t = (key) => locales[currentLang.value][key] || locales['zh'][key] || key

// Hardware
const showHardware = ref(false)
const hwLoading = ref(false)
const hwError = ref('')
const hwInfo = ref({ cpu: '', ram: '', gpu: '', vram: '' })
const hwRecommend = ref('')

// Network diagnose
const networkDiagLoading = ref(false)
const networkDiagResult = ref('')

// Market
const searchQuery = ref('')
const sizeFilter = ref('all')
const quantFilter = ref('all')
const tab = ref('recommended')
const marketLoading = ref(false)
const allModels = ref([])
const downloadedModels = ref([])
const detailModel = ref(null)

// 下载逻辑委托给 composable
const {
  downloading, hfKey, downloadModel: dlModel, downloadHFModel: dlHFModel,
  cancelDownload: cancelDl, cancelHFDownload: cancelHFDl,
  pauseDownload: pauseDl, resumeDownload: resumeDl,
  pauseHFDownload: pauseHFDl, resumeHFDownload: resumeHFDl,
  retryDownload: retryDl, retryHFDownload: retryHFDl,
  stopPolling,
} = useModelDownload(allModels, loadDownloadedModels, loadModels)

function selectModel(path, modelType) {
  if (onModelSelected) onModelSelected(path)
  else appStore.setModel(path, modelType)
}

// 包装函数以匹配模板中的调用签名
function downloadModel(model) { dlModel(model, (p) => selectModel(p)) }
function downloadHFModel(model) { dlHFModel(model, t) }
function cancelDownload(model) { cancelDl(model) }
function cancelHFDownload(model) { cancelHFDl(model) }
function pauseDownload(model) { pauseDl(model) }
function resumeDownload(model) { resumeDl(model) }
function pauseHFDownload(model) { pauseHFDl(model) }
function resumeHFDownload(model) { resumeHFDl(model) }
function retryDownload(model) { retryDl(model) }
function retryHFDownload(model) { retryHFDl(model) }

const filteredModels = computed(() => {
  let list = allModels.value || []
  
  // Tab filter
  if (tab.value === 'recommended') list = list.filter(m => m.recommended)
  else if (tab.value === 'downloaded') list = list.filter(m => m.downloaded)
  
  // Search
  if (searchQuery.value) {
    const q = searchQuery.value.toLowerCase()
    list = list.filter(m => (m.name || '').toLowerCase().includes(q) || (m.family || '').toLowerCase().includes(q) || (m.key || '').toLowerCase().includes(q))
  }
  
  // Size filter
  if (sizeFilter.value !== 'all') {
    const ranges = { tiny: [0, 3], small: [3, 8], medium: [8, 14], large: [14, 34], xl: [34, Infinity] }
    const [lo, hi] = ranges[sizeFilter.value]
    list = list.filter(m => {
      const sz = m.size != null ? String(m.size) : ''
      const num = parseFloat(sz)
      if (isNaN(num)) {
        return true
      }
      return num >= lo && num < hi
    })
  }
  
  // Quant filter
  if (quantFilter.value !== 'all') {
    list = list.filter(m => (m.quant || '').startsWith(quantFilter.value))
  }
  
  return list
})

function formatSize(s) {
  if (!s) return ''
  return String(s).replace(/b$/i,'B')
}

// Hardware detection
async function detectHardware() {
  hwLoading.value = true
  hwError.value = ''
  try {
    const res = await fetch(`${API_BASE}/api/system/hardware`)
    const data = await res.json()
    if (data.status === 'ok') {
      hwInfo.value = {
        cpu: data.cpu || '',
        ram: data.ram || '',
        gpu: data.gpu || '',
        vram: data.vram || ''
      }
      hwRecommend.value = data.recommend || ''
      // After hardware detection, reload models with recommendations
      await loadModels()
    } else {
      hwError.value = data.message || 'Detection failed'
    }
  } catch (err) {
    hwError.value = err.message || 'Network error'
  } finally {
    hwLoading.value = false
  }
}

// Network diagnose
async function runNetworkDiagnose() {
  networkDiagLoading.value = true
  networkDiagResult.value = ''
  try {
    const res = await fetch(`${API_BASE}/api/network/diagnose`)
    const data = await res.json()
    if (data.status === 'ok' && data.diagnosis) {
      // 格式化诊断结果
      const d = data.diagnosis
      let text = ''
      if (d.system_proxy) text += `系统代理: ${d.system_proxy}\n`
      if (d.hf_official) text += `HF官方: ${d.hf_official}\n`
      if (d.hf_mirror) text += `HF镜像: ${d.hf_mirror}\n`
      if (d.recommended_source) text += `推荐源: ${d.recommended_source}\n`
      if (d.errors && d.errors.length > 0) text += `错误: ${d.errors.join('; ')}\n`
      if (!text) text = JSON.stringify(d, null, 2)
      networkDiagResult.value = text
    } else {
      networkDiagResult.value = data.message || '诊断失败'
    }
  } catch (err) {
    networkDiagResult.value = err.message || '网络错误'
  } finally {
    networkDiagLoading.value = false
  }
}

// Load model list from backend
async function loadModels() {
  marketLoading.value = true
  try {
    const res = await fetch(`${API_BASE}/api/models/list`)
    const data = await res.json()
    if (data.status === 'ok') {
      allModels.value = (data.models || []).map(m => ({
        key: m.key,
        name: m.name || m.key,
        hf_id: m.hf_id || m.key,
        size: m.size || '',
        quant: m.quant || '',
        family: m.family || '',
        task: m.task || '',
        desc: m.desc || m.description || '',
        recommended: m.recommended || false,
        downloaded: m.downloaded || false,
        path: m.path || '',
        memory_estimate: m.memory_estimate || '',
        url: m.url || '',
        file_size_mb: m.file_size_mb || 0,
        file_size_display: m.file_size_display || '',
        params_b: m.params_b || '',
        hf_train_repo: m.hf_train_repo || '',
      }))
    }
  } catch (err) {
    console.warn('[ModelMarket] 加载模型列表失败:', err.message)
  } finally {
    marketLoading.value = false
  }
}

// Load downloaded models
async function loadDownloadedModels() {
  try {
    const res = await fetch(`${API_BASE}/api/models/downloaded`)
    const data = await res.json()
    if (data.status === 'ok') {
      downloadedModels.value = data.models || []
    }
  } catch (err) {
    console.warn('[ModelMarket] 加载已下载模型失败:', err.message)
  }
}

// Delete downloaded model
async function deleteModel(model) {
  const ok = await $confirm({ title: t('delete'), message: t('confirm_delete_model'), type: 'danger' });
  if (!ok) return
  try {
    const res = await fetch(`${API_BASE}/api/models/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: model.path })
    })
    const data = await res.json()
    if (data.status === 'ok') {
      loadDownloadedModels()
      await loadModels()
    }
  } catch (err) {
    console.warn('[ModelMarket] 删除模型失败:', err.message)
  }
}

// Apply downloaded model — 使用异步切换，状态由全局 useApi 管理
async function applyModel(model) {
  if (!model.path) return

  // 防止重复点击
  if (switchingState.value === 'switching') {
    if (toast) toast('⏳ 正在切换模型中，请勿重复操作', 'warning');
    return
  }

  const type = model.model_type || (model.path.toLowerCase().endsWith('.gguf') ? 'gguf' : 'huggingface')
  let payload
  if (type === 'gguf') {
    payload = { model_type: 'gguf', gguf_path: model.path, model_name: model.path }
  } else if (type === 'self') {
    payload = { model_type: 'self', model_name: model.path, gguf_path: '' }
  } else {
    payload = { model_type: 'huggingface', model_name: model.path, gguf_path: '' }
  }

  const ok = await switchModel(payload)
  if (ok === false && switchingState.value !== 'switching') {
    if (toast) toast(`❌ ${switchingError.value || '切换失败'}`, 'error')
    return
  }

  // 切换已异步启动，成功后会通过 watch 通知
  selectModel(model.path, type)
}

function showModelDetail(model) {
  detailModel.value = model
}

onMounted(async () => {
  await detectHardware()
  await loadDownloadedModels()
})

onUnmounted(() => {
  // 组件卸载时，清理所有正在运行的轮询定时器，防止内存泄漏
  Object.values(downloading.value).forEach(dl => {
    if (dl && dl._pollInterval) {
      clearInterval(dl._pollInterval)
    }
  })
})
</script>

<style scoped>
.model-market {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: min(1000px, calc(100vw - 40px));
  padding-bottom: 48px;
}

/* Hardware Panel */
.hw-panel {
  background: var(--bg-muted);
  border-radius: 8px;
  padding: 10px 14px;
}
.hw-header {
  display: flex;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
}
.hw-title { font-weight: 600; font-size: 14px; }
.hw-toggle { color: var(--text-muted); font-size: 12px; }
.hw-details { margin-top: 10px; }
.hw-loading, .hw-error { font-size: 13px; color: var(--text-muted); }
.hw-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13px; }
.hw-item { display: flex; gap: 6px; }
.hw-label { color: var(--text-muted); }
.hw-recommend { color: var(--warning); font-weight: 600; }
.btn-detect {
  margin-top: 8px;
  padding: 4px 12px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-card);
  cursor: pointer;
}
.btn-detect:hover { background: var(--bg-hover); }
.btn-detect:disabled { opacity: 0.5; cursor: not-allowed; }

/* Network diagnose */
.network-diag-result {
  margin-top: 10px;
  padding: 8px 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 12px;
}
.diag-title { font-weight: 600; margin-bottom: 4px; color: var(--text); }
.diag-text { 
  white-space: pre-wrap; 
  word-break: break-all; 
  margin: 0; 
  font-size: 11px; 
  color: var(--text-muted);
  max-height: 200px;
  overflow-y: auto;
}

/* Toolbar */
.market-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.market-search {
  flex: 1;
  min-width: 140px;
  padding: 6px 10px;
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text);
}
.market-filter {
  padding: 6px 8px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text);
}
.market-tabs {
  display: flex;
  gap: 4px;
}
.tab-btn {
  padding: 5px 10px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-card);
  color: var(--text);
  cursor: pointer;
  white-space: nowrap;
}
.tab-btn.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

/* Model Cards */
.market-list { flex: 1; overflow-y: auto; }
.market-status { text-align: center; padding: 20px; color: var(--text-muted); font-size: 13px; }
.market-empty { padding: 40px 20px; }
.empty-icon { font-size: 48px; margin: 0 0 12px 0; }
.empty-hint { font-size: 12px; color: var(--text-muted); margin: 8px 0 0 0; }
.btn-retry {
  margin-top: 12px;
  padding: 6px 16px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-card);
  color: var(--text);
  cursor: pointer;
}
.btn-retry:hover { background: var(--bg-hover); }
.model-cards { display: flex; flex-direction: column; gap: 8px; }
.model-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  transition: all 0.2s;
}
.model-card.recommended { border-color: var(--warning); background: var(--warning-light); }
.model-card.downloaded { border-color: var(--success); }
.mc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.mc-name { font-weight: 600; font-size: 13px; }
.mc-badge {
  font-size: 11px;
  padding: 2px 10px;
  border-radius: 10px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  line-height: 1.4;
}
.mc-badge.rec { background: var(--warning-light); color: var(--warning); }
.mc-badge.dl { background: var(--success-light); color: var(--success); }
.mc-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.mc-tag {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 4px;
  background: var(--bg-muted);
  color: var(--text-muted);
}
.mc-desc { font-size: 12px; color: var(--text-muted); margin-bottom: 8px; line-height: 1.4; }
.mc-storage-row { margin-bottom: 6px; }
.mc-path { font-size: 11px; color: var(--text-muted); word-break: break-all; }
.mc-actions { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }

/* Progress bar */
.mc-progress-bar-wrapper {
  width: 100%;
  margin-bottom: 6px;
}
.mc-progress-bar {
  width: 100%;
  height: 6px;
  background: var(--bg-muted);
  border-radius: 3px;
  overflow: hidden;
}
.mc-progress-fill {
  height: 100%;
  background: var(--primary-gradient);
  border-radius: 3px;
  transition: width 0.3s ease;
  min-width: 0;
}

.mc-dl-progress {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--warning);
}
.mc-error-icon { font-size: 14px; }
.dl-progress-text { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dl-error-text { color: var(--danger); }

/* Error detail (shown on hover) */
.mc-error-detail {
  margin-top: 8px;
  padding: 8px 10px;
  background: var(--danger-light);
  border: 1px solid var(--danger);
  border-radius: 6px;
  font-size: 11px;
}
.error-detail-title { font-weight: 600; color: var(--danger); margin-bottom: 4px; }
.error-detail-text { 
  white-space: pre-wrap; 
  word-break: break-all; 
  margin: 0; 
  color: var(--danger); 
  font-size: 11px; 
  max-height: 120px; 
  overflow-y: auto; 
}

.btn-model {
  padding: 5px 12px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  cursor: pointer;
  background: var(--bg-card);
  color: var(--text);
  white-space: nowrap;
}
.btn-model.download { background: var(--primary); color: white; border-color: var(--primary); }
.btn-model.download:hover { background: var(--primary-hover); }
.btn-model.done { background: var(--success-light); color: var(--success); border-color: var(--success); cursor: default; }
.btn-model.downloading { background: var(--warning-light); color: var(--warning); border-color: var(--warning); cursor: wait; }
.btn-model.retry-btn { background: var(--danger); color: white; border-color: var(--danger); }
.btn-model.retry-btn:hover { opacity: 0.9; }
.btn-model.cancel-btn { background: var(--warning); color: white; border-color: var(--warning); }
.btn-model.cancel-btn:hover { opacity: 0.9; }
.btn-model.pause-btn { background: var(--purple); color: white; border-color: var(--purple); }
.btn-model.pause-btn:hover { opacity: 0.9; }
.btn-model.resume-btn { background: var(--success); color: white; border-color: var(--success); }
.btn-model.resume-btn:hover { opacity: 0.9; }
.btn-model.info-btn { padding: 5px 8px; }
.btn-model.small { padding: 3px 8px; font-size: 11px; }

.mc-dl-buttons {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.dl-cancelled-text { color: var(--text-muted); }
.dl-paused-text { color: var(--purple); }
.btn-model.danger { color: var(--danger); border-color: var(--danger); }
.btn-model.danger:hover { background: var(--danger-light); }
.btn-model.apply-btn { background: var(--success); color: white; border-color: var(--success); font-weight: 600; }
.btn-model.apply-btn:hover { opacity: 0.9; }

.spinner {
  display: inline-block;
  width: 10px; height: 10px;
  border: 2px solid var(--warning);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 4px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Downloaded List */
.downloaded-list { margin-top: 8px; }
.downloaded-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: var(--bg-muted);
  border-radius: 6px;
  margin-bottom: 4px;
  flex-wrap: wrap;
  font-size: 12px;
}
.dl-name { font-weight: 600; }
.dl-type { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; white-space: nowrap; }
.dl-type.type-gguf { background: var(--info-light); color: var(--info); }
.dl-type.type-hf { background: var(--danger-light); color: var(--danger); }
.dl-type.type-self { background: #ede9fe; color: #7c3aed; }
.dl-size { color: var(--text-muted); font-size: 11px; white-space: nowrap; }
.dl-path { color: var(--text-muted); flex: 1; min-width: 100px; word-break: break-all; }

/* Modal */
.modal-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999;
}
.modal-detail {
  background: var(--bg-card);
  border-radius: 12px;
  padding: 20px 24px;
  max-width: 480px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  color: var(--text);
}
.modal-detail h3 { margin: 0 0 12px; font-size: 16px; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; margin-bottom: 12px; }
.detail-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5; }
.detail-actions { display: flex; gap: 8px; }
/* HF 下载按钮 */
.hf-btn {
  background: linear-gradient(135deg, #ec4899 0%, #db2777 100%) !important;
  color: white !important;
  font-size: 0.72rem !important;
  padding: 4px 10px !important;
}
</style>