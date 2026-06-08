<template>
  <section class="dedicated-view">
    <div class="view-header"><h2><SettingsIcon :size="24" /> {{ t('sys_settings') }}</h2></div>
    <div class="view-body">
      <!-- Current loaded model -->
      <div class="panel-section">
        <div class="panel-header"><h3><BrainIcon :size="18" /> {{ t('current_loaded_model') }}</h3></div>
        <div class="panel-content">
          <div v-if="currentModelLoading" class="text-muted">{{ t('checking_model') }}</div>
          <div v-else-if="currentModelInfo && currentModelInfo.loaded">
            <div class="current-model-row">
              <span class="model-type-badge" :class="modelTypeClass(currentModelInfo)">{{ modelTypeLabel(currentModelInfo) }}</span>
              <span class="model-path-text">{{ currentModelInfo.model_path || currentModelInfo.effective_path || '' }}</span>
            </div>
            <div v-if="currentModelInfo.pending_settings?.pending_model_path" class="pending-model-row">
              <span><Clock :size="12" /> {{ t('apply_change') }}: {{ currentModelInfo.pending_settings.pending_model_path }}</span>
            </div>
          </div>
          <p v-else class="text-muted">{{ t('no_model_loaded') }}</p>
          <button class="primary-btn" style="margin-top:8px;" @click="loadCurrentModel">{{ t('reload') }}</button>
        </div>
      </div>
      <!-- UI Theme -->
      <div class="panel-section">
        <div class="panel-header"><h3><Palette :size="18" /> {{ t('ui_settings') }}</h3></div>
        <div class="panel-content params-grid">
          <label class="param-item"><span>{{ t('theme') }}</span><select v-model="appStore.currentTheme" class="theme-select" @change="appStore.setTheme($event.target.value)"><option value="light">{{ t('theme_light') }}</option><option value="dark">{{ t('theme_dark') }}</option><option value="auto">{{ t('theme_auto') }}</option></select></label>
        </div>
        <!-- 主题色选择 -->
        <div class="panel-content">
          <label class="param-item"><span>主题色</span></label>
          <div class="theme-accent-row">
            <button
              v-for="preset in appStore.accentPresets"
              :key="preset.color"
              class="accent-swatch"
              :class="{ active: appStore.currentAccent === preset.color }"
              :style="{ background: preset.color }"
              :title="preset.name"
              @click="appStore.setAccent(preset.color)"
            ></button>
            <button
              class="accent-swatch accent-reset"
              :class="{ active: !appStore.currentAccent }"
              title="恢复默认"
              @click="appStore.setAccent('')"
            >×</button>
            <input type="color" class="accent-custom" :value="appStore.currentAccent || '#5b7a8a'" @input="appStore.setAccent($event.target.value)" title="自定义颜色" />
          </div>
        </div>
        <!-- 背景图设置 -->
        <div class="panel-content">
          <label class="param-item"><span>背景图片</span></label>
          <div class="bg-upload-row">
            <label class="file-upload-area">
              <ImageIcon :size="14" /> 选择图片
              <input type="file" accept="image/*" @change="onBgImageSelect" />
            </label>
            <button v-if="appStore.currentBgImage" class="btn-secondary" @click="appStore.setBgImage('')">清除背景</button>
          </div>
          <p v-if="appStore.currentBgImage" class="text-muted" style="margin:4px 0 0;font-size:0.78rem;">✅ 背景图已生效，侧边栏和面板会自动适应</p>
        </div>
      </div>
      <!-- Hardware -->
      <div class="panel-section">
        <div class="panel-header"><h3><MonitorIcon :size="18" /> {{ t('hardware_settings') }}</h3></div>
        <div class="panel-content params-grid">
          <label class="param-item"><span>{{ t('inference_device') }}</span><select v-model="currentHardware" class="theme-select"><option value="auto">{{ t('device_auto') }}</option><option value="cpu">{{ t('device_cpu') }}</option></select></label>
        </div>
      </div>
      <!-- Model Path -->
      <div class="panel-section">
        <div class="panel-header"><h3><FolderOpen :size="18" /> {{ t('model_name_path') }}</h3></div>
        <div class="panel-content">
          <div style="display:flex;gap:8px;align-items:center;"><PathSelector v-model="currentModel" style="flex:1;" /></div>
          <div style="display:flex;gap:8px;margin-top:8px;">
            <select v-model="modelType" class="theme-select"><option value="auto">auto (自动)</option><option v-if="taijiAvailable" value="self">🧬 态极 (ModelSelf)</option><option value="huggingface">HuggingFace</option><option value="gguf">GGUF</option></select>
            <button class="primary-btn" @click="applyModelSettings" :disabled="switchingState === 'switching'">
              {{ switchingState === 'switching' ? '切换中...' : t('apply_and_switch') }}
            </button>
          </div>
          <!-- 切换状态指示 -->
          <div v-if="switchingState === 'switching'" class="switch-status-bar switching">
            <span class="spinner"></span> {{ switchingMessage || '正在切换模型...' }}
          </div>
          <div v-else-if="switchingState === 'success'" class="switch-status-bar success">
            <CheckCircle :size="14" /> {{ switchingMessage || '模型切换成功' }}
          </div>
          <div v-else-if="switchingState === 'error'" class="switch-status-bar error">
            <XCircle :size="14" /> {{ switchingError || '模型切换失败' }}
          </div>
          <p v-else style="font-size:0.75rem;color:var(--text-muted);margin:6px 0 0;"><Info :size="12" /> {{ t('apply_change') }}即时生效，无需重启</p>
        </div>
      </div>
      <!-- Cloud API -->
      <div class="panel-section">
        <div class="panel-header"><h3><Cloud :size="18" /> {{ t('cloud_api_settings') }}</h3></div>
        <!-- 配置列表选择 -->
        <div class="panel-content">
          <div class="cloud-profiles-bar">
            <select v-model="activeCloudProfileId" class="form-input cloud-profile-select" @change="onProfileSwitch">
              <option v-for="p in cloudProfiles" :key="p.id" :value="p.id">{{ p.name }}</option>
            </select>
            <button class="primary-btn btn-sm" @click="addCloudProfile" title="新增配置">+ 新增</button>
            <button v-if="cloudProfiles.length > 1" class="btn-secondary btn-sm" @click="removeCloudProfile" title="删除当前配置">删除</button>
          </div>
        </div>
        <!-- 当前配置编辑 -->
        <div class="panel-content params-grid">
          <label class="param-item"><span>配置名称</span><input v-model="cloudApiName" class="form-input" placeholder="如：OpenAI / Claude / DeepSeek" /></label>
          <label class="param-item"><span>{{ t('api_provider') }}</span><select v-model="cloudApiType" class="form-input"><option value="openai">{{ t('openai_compatible') }}</option><option value="anthropic">{{ t('anthropic_compatible') }}</option></select></label>
          <label class="param-item"><span>{{ t('api_base_url') }}</span><input v-model="cloudApiBase" class="form-input" :placeholder="cloudApiType === 'anthropic' ? 'https://api.anthropic.com' : 'https://api.openai.com/v1'" /></label>
          <label class="param-item"><span>{{ t('api_key') }}</span><input v-model="cloudApiKey" class="form-input" type="password" /></label>
          <label class="param-item"><span>{{ t('api_model') }}</span><input v-model="cloudApiModel" class="form-input" :placeholder="cloudApiType === 'anthropic' ? 'claude-sonnet-4-20250514' : 'gpt-4o'" /></label>
        </div>
        <div class="panel-content" style="flex-direction:row;"><button class="primary-btn" @click="saveCloudSettings">{{ t('save') }}</button></div>
      </div>
      <!-- System Prompt -->
      <div class="panel-section">
        <div class="panel-header"><h3><MessageSquareText :size="18" /> {{ t('system_prompt_settings') }}</h3></div>
        <div class="panel-content">
          <textarea v-model="currentSystemPrompt" class="form-input" rows="4" style="resize:vertical;min-height:80px;"></textarea>
          <div style="display:flex;gap:8px;"><button class="primary-btn" @click="saveSettings">{{ t('save') }}</button><button class="btn-secondary" @click="currentSystemPrompt='你是一个全能助手。';saveSettings()">{{ t('reset_default') }}</button></div>
        </div>
      </div>
      <!-- Update -->
      <div class="panel-section">
        <div class="panel-header"><h3><RefreshCw :size="18" /> {{ t('software_update') }}</h3></div>
        <div class="panel-content">
          <p class="text-muted" style="margin:0 0 8px;">{{ t('current_version') }}: {{ appVersion }}</p>
          <div style="display:flex;gap:8px;">
            <button class="primary-btn" @click="checkUpdate" :disabled="updateChecking">{{ updateChecking ? t('checking') : t('check_update') }}</button>
            <button v-if="updateAvailable" class="train-btn primary" @click="applyUpdate">{{ t('update_now') }}</button>
          </div>
          <p v-if="updateMsg" class="text-muted" style="margin-top:8px;">{{ updateMsg }}</p>
        </div>
      </div>
      <!-- Hot Update / Patch Management -->
      <div class="panel-section">
        <div class="panel-header"><h3><Wrench :size="18" /> 热更新管理</h3></div>
        <div class="panel-content">
          <!-- Installed patches list -->
          <div class="patch-header">
            <span class="label">已安装的补丁<span class="patch-count-badge">{{ installedPatches.length }}</span></span>
            <button class="primary-btn btn-sm" @click="loadPatches"><RefreshCw :size="12" /> 刷新</button>
          </div>
          <div v-if="installedPatches.length > 0" class="patches-list">
            <div v-for="p in installedPatches" :key="p.module" class="patch-item">
              <span class="patch-module">{{ p.module }}</span>
              <span class="patch-path">{{ p.path }}</span>
              <span class="patch-size">{{ (p.size / 1024).toFixed(1) }} KB</span>
            </div>
          </div>
          <div v-else class="empty-state">
            <Inbox class="empty-icon" :size="32" />
            <span>暂无已安装的补丁</span>
          </div>

          <!-- Upload patch -->
          <div class="section-divider">
            <div class="upload-section-label"><FileTextIcon :size="14" /> 上传 Python 补丁（支持子目录结构）</div>
            <div class="upload-row">
              <input v-model="patchTargetPath" class="form-input" placeholder="目标路径（如 api/routes_chat.py）" style="flex:1;min-width:200px;" />
              <label class="file-upload-area">
                <FolderOpen :size="14" /> 选择文件
                <input type="file" accept=".py" @change="onPatchFileSelected" />
              </label>
            </div>
            <div v-if="selectedPatchFile" class="file-pending">
              <span class="file-name"><FileTextIcon :size="14" /> {{ selectedPatchFile.name }}</span>
              <button class="primary-btn btn-sm" @click="uploadPatch" :disabled="patchUploading">
                {{ patchUploading ? '上传中...' : '部署补丁' }}
                <template v-if="!patchUploading"><Upload :size="12" /></template>
              </button>
            </div>
            <p v-if="patchUploadMsg" class="status-msg" :class="{ success: patchUploadMsg.startsWith('✅'), error: patchUploadMsg.startsWith('❌') }">{{ patchUploadMsg }}</p>
          </div>

          <!-- Upload update package (ZIP) -->
          <div class="section-divider">
            <div class="upload-section-label"><PackageIcon :size="14" /> 上传更新包 (.zip)</div>
            <label class="file-upload-area">
              <FolderOpen :size="14" /> 选择更新包
              <input type="file" accept=".zip" @change="onUpdateZipSelected" />
            </label>
            <div v-if="selectedUpdateZip" class="file-pending">
              <span class="file-name"><PackageIcon :size="14" /> {{ selectedUpdateZip.name }}</span>
              <button class="primary-btn btn-sm" @click="uploadUpdateZip" :disabled="updateZipUploading">
                {{ updateZipUploading ? '安装中...' : '安装更新' }}
                <template v-if="!updateZipUploading"><Download :size="12" /></template>
              </button>
            </div>
            <p v-if="updateZipMsg" class="status-msg" :class="{ success: updateZipMsg.startsWith('✅'), error: updateZipMsg.startsWith('❌') }">{{ updateZipMsg }}</p>
          </div>

          <!-- Reload modules button -->
          <div class="section-divider" style="display:flex;gap:8px;align-items:center;">
            <button class="primary-btn" @click="reloadAllModules" :disabled="modulesReloading">
              {{ modulesReloading ? '重载中...' : '热重载所有补丁' }}
              <template v-if="!modulesReloading"><RefreshCw :size="14" /></template>
            </button>
          </div>
          <p v-if="reloadMsg" class="status-msg" :class="{ success: reloadMsg.startsWith('✅'), error: reloadMsg.startsWith('❌') }">{{ reloadMsg }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Inbox, Settings as SettingsIcon, Brain as BrainIcon, Palette, Monitor as MonitorIcon, FolderOpen, Cloud, MessageSquareText, RefreshCw, Wrench, FileText as FileTextIcon, Package as PackageIcon, Image as ImageIcon, Upload, Download, Clock, Info, CheckCircle, XCircle } from 'lucide-vue-next';

import { ref, inject, watch } from 'vue';
import PathSelector from '../components/PathSelector.vue';
import { useApi, API_BASE } from '../composables/useApi.js';
import { useAppStore } from '../stores/appStore.js';
const { t, saveSettingsToServer, switchingState, switchingMessage, switchingError, switchModel, checkHealth, taijiAvailable } = useApi();
const toast = inject('toast');
const appStore = useAppStore();

// Reactive state from localStorage
const currentHardware = ref(localStorage.getItem('omnicore_hardware') || 'auto');
const currentModel = ref(localStorage.getItem('omnicore_model') || '');
const modelType = ref(localStorage.getItem('omnicore_model_type') || 'auto');
const currentSystemPrompt = ref(localStorage.getItem('omnicore_system_prompt') || '你是一个全能助手。');

// Cloud API 多配置管理
const cloudProfiles = ref([]);
const activeCloudProfileId = ref('');
const cloudApiName = ref('');
const cloudApiType = ref('openai');
const cloudApiBase = ref('');
const cloudApiKey = ref('');
const cloudApiModel = ref('');
const currentModelInfo = ref(null);
const currentModelLoading = ref(false);
const appVersion = ref('1.0.0');
const updateChecking = ref(false);
const updateAvailable = ref(false);
const updateMsg = ref('');

// Patch management state
const installedPatches = ref([]);
const patchTargetPath = ref('');
const selectedPatchFile = ref(null);
const patchUploading = ref(false);
const patchUploadMsg = ref('');
const selectedUpdateZip = ref(null);
const updateZipUploading = ref(false);
const updateZipMsg = ref('');
const modulesReloading = ref(false);
const reloadMsg = ref('');

const onBgImageSelect = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    appStore.setBgImage(ev.target.result);
    toast('✅ 背景图已设置', 'success');
  };
  reader.readAsDataURL(file);
};

const modelTypeClass = (info) => { const t = info?.model_type || info?.effective_type || ''; if (t === 'self') return 'self'; return t === 'gguf' ? 'gguf' : 'huggingface'; };
const modelTypeLabel = (info) => { const t = info?.model_type || info?.effective_type || ''; if (t === 'self') return '🧬 态极 ModelSelf（原生推理+多模态）'; return t === 'gguf' ? '⚡ GGUF 推理（仅推理，不可微调）' : '🧩 HuggingFace（可微调训练）'; };

const loadCurrentModel = async () => {
  currentModelLoading.value = true;
  try {
    const r = await fetch(`${API_BASE}/api/system/current_model`);
    if (r.ok) currentModelInfo.value = await r.json();
  } catch (e) {}
  finally { currentModelLoading.value = false; }
};

// 监听切换状态变化，切换成功/失败时刷新模型信息和连接状态
watch(switchingState, (val) => {
  if (val === 'success') {
    toast(`✅ ${switchingMessage.value || '模型切换成功'}`, 'success');
    loadCurrentModel();
    checkHealth(); // 重新检查健康状态，更新连接指示器
  } else if (val === 'error') {
    toast(`❌ ${switchingError.value || '模型切换失败'}`, 'error');
    checkHealth(); // 失败时也重新检查，确保状态同步
  }
});

const applyModelSettings = async () => {
  const path = currentModel.value.trim(); if (!path) { toast('⚠ 请输入模型路径', 'warning'); return; }
  
  // 如果已经在切换中，阻止重复点击
  if (switchingState.value === 'switching') {
    toast('⏳ 正在切换模型中，请勿重复操作', 'warning');
    return;
  }
  
  const isGGUF = modelType.value === 'gguf' || path.toLowerCase().endsWith('.gguf');
  const isModelSelf = modelType.value === 'self' || path.includes('taiji');
  const finalType = modelType.value === 'auto'
    ? (isGGUF ? 'gguf' : isModelSelf ? 'self' : 'huggingface')
    : modelType.value;

  // 态极模型 (ModelSelf) 走专门的切换路径
  if (finalType === 'self') {
    const payload = { model_type: 'self', model_name: path, gguf_path: '' };
    const ok = await switchModel(payload);
    if (ok === false && switchingState.value !== 'switching') {
      if (switchingState.value === 'error') {
        toast(`❌ ${switchingError.value || '态极模型切换失败'}`, 'error');
      }
      return;
    }
    localStorage.setItem('omnicore_model', path);
    localStorage.setItem('omnicore_model_type', 'self');
    localStorage.setItem('omnicore_hardware', currentHardware.value);
    loadCurrentModel();
    return;
  }

  // 1. 先保存硬件设置（如有变更）
  if (currentHardware.value === 'cpu') {
    await fetch(`${API_BASE}/api/settings/device`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: 'cpu' }) });
  }

  // 2. 调用异步切换
  const payload = finalType === 'gguf'
    ? { model_type: 'gguf', gguf_path: path, model_name: path }
    : { model_type: 'huggingface', model_name: path, gguf_path: '' };

  const ok = await switchModel(payload);
  if (ok === false && switchingState.value !== 'switching') {
    // 只有明确失败才显示错误
    if (switchingState.value === 'error') {
      toast(`❌ ${switchingError.value || '启动切换失败'}`, 'error');
    }
    return;
  }

  // 保存设置到 localStorage（切换成功后 watch 会自动刷新 modelInfo）
  if (modelType.value === 'auto') { modelType.value = finalType; }
  localStorage.setItem('omnicore_model', path);
  localStorage.setItem('omnicore_model_type', finalType);
  localStorage.setItem('omnicore_hardware', currentHardware.value);
  loadCurrentModel();
};

const saveSettings = async () => {
  localStorage.setItem('omnicore_system_prompt', currentSystemPrompt.value);
  await saveSettingsToServer({
    theme: appStore.currentTheme,
    accent: appStore.currentAccent,
    hardware: currentHardware.value,
    model: currentModel.value,
    engine: 'local-chat',
    system_prompt: currentSystemPrompt.value,
  });
  toast('✅ 设置已保存', 'success');
};

// 云端配置初始化
function loadCloudProfiles() {
  try {
    const saved = localStorage.getItem('omnicore_cloud_profiles');
    if (saved) {
      cloudProfiles.value = JSON.parse(saved);
    }
  } catch (e) {}
  
  // 迁移旧的单配置到多配置
  if (cloudProfiles.value.length === 0) {
    const oldType = localStorage.getItem('omnicore_cloud_type') || 'openai';
    const oldBase = localStorage.getItem('omnicore_cloud_base') || '';
    const oldKey = localStorage.getItem('omnicore_cloud_key') || '';
    const oldModel = localStorage.getItem('omnicore_cloud_model') || '';
    cloudProfiles.value = [{
      id: 'default',
      name: '默认配置',
      type: oldType,
      baseUrl: oldBase,
      apiKey: oldKey,
      model: oldModel
    }];
  }
  
  // 设置活跃配置
  const savedActive = localStorage.getItem('omnicore_cloud_active');
  if (savedActive && cloudProfiles.value.find(p => p.id === savedActive)) {
    activeCloudProfileId.value = savedActive;
  } else {
    activeCloudProfileId.value = cloudProfiles.value[0]?.id || 'default';
  }
  
  // 加载当前活跃配置到编辑区
  loadActiveProfileToFields();
}

function loadActiveProfileToFields() {
  const profile = cloudProfiles.value.find(p => p.id === activeCloudProfileId.value);
  if (profile) {
    cloudApiName.value = profile.name || '';
    cloudApiType.value = profile.type || 'openai';
    cloudApiBase.value = profile.baseUrl || '';
    cloudApiKey.value = profile.apiKey || '';
    cloudApiModel.value = profile.model || '';
  }
}

function onProfileSwitch() {
  loadActiveProfileToFields();
  // 同时保存到单值key供聊天使用
  localStorage.setItem('omnicore_cloud_active', activeCloudProfileId.value);
  const profile = cloudProfiles.value.find(p => p.id === activeCloudProfileId.value);
  if (profile) {
    localStorage.setItem('omnicore_cloud_type', profile.type || 'openai');
    localStorage.setItem('omnicore_cloud_base', profile.baseUrl || '');
    localStorage.setItem('omnicore_cloud_key', profile.apiKey || '');
    localStorage.setItem('omnicore_cloud_model', profile.model || '');
  }
}

function addCloudProfile() {
  const id = 'cloud_' + Date.now();
  cloudProfiles.value.push({
    id,
    name: `配置 ${cloudProfiles.value.length + 1}`,
    type: 'openai',
    baseUrl: '',
    apiKey: '',
    model: ''
  });
  activeCloudProfileId.value = id;
  loadActiveProfileToFields();
}

function removeCloudProfile() {
  if (cloudProfiles.value.length <= 1) return;
  const idx = cloudProfiles.value.findIndex(p => p.id === activeCloudProfileId.value);
  if (idx < 0) return;
  cloudProfiles.value.splice(idx, 1);
  activeCloudProfileId.value = cloudProfiles.value[Math.min(idx, cloudProfiles.value.length - 1)]?.id || '';
  loadActiveProfileToFields();
}

// 初始化加载云端配置
loadCloudProfiles();

const saveCloudSettings = () => {
  // 保存当前编辑的配置到 profiles
  const activeIdx = cloudProfiles.value.findIndex(p => p.id === activeCloudProfileId.value);
  if (activeIdx >= 0) {
    cloudProfiles.value[activeIdx] = {
      id: activeCloudProfileId.value,
      name: cloudApiName.value || `配置 ${activeIdx + 1}`,
      type: cloudApiType.value,
      baseUrl: cloudApiBase.value,
      apiKey: cloudApiKey.value,
      model: cloudApiModel.value
    };
  }
  
  // 保存多配置列表
  localStorage.setItem('omnicore_cloud_profiles', JSON.stringify(cloudProfiles.value));
  localStorage.setItem('omnicore_cloud_active', activeCloudProfileId.value);
  
  // 同时更新单值key（向后兼容）
  localStorage.setItem('omnicore_cloud_type', cloudApiType.value);
  localStorage.setItem('omnicore_cloud_base', cloudApiBase.value);
  localStorage.setItem('omnicore_cloud_key', cloudApiKey.value);
  localStorage.setItem('omnicore_cloud_model', cloudApiModel.value);

  // 同步云端配置到后端持久化
  saveSettingsToServer({
    cloud_profiles: cloudProfiles.value,
    cloud_active: activeCloudProfileId.value,
    cloud_type: cloudApiType.value,
    cloud_base: cloudApiBase.value,
    cloud_key: cloudApiKey.value,
    cloud_model: cloudApiModel.value,
  }).catch(() => {});

  toast('✅ 云端配置已保存', 'success');
};

const checkUpdate = async () => {
  updateChecking.value = true; updateMsg.value = '';
  try { const r = await fetch(`${API_BASE}/api/system/check_update`, { method: 'POST' }); const d = await r.json(); updateAvailable.value = d.has_update; updateMsg.value = d.has_update ? `${t('update_available')} v${d.version}` : (d.message || '已是最新版本'); } catch (e) { updateMsg.value = `❌ ${e.message}`; }
  finally { updateChecking.value = false; }
};
const applyUpdate = async () => { try { await fetch(`${API_BASE}/api/system/apply_update`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }); } catch (e) {} };

// Load on mount
loadCurrentModel();
loadPatches();
loadAppVersion();

// Patch management functions
async function loadPatches() {
  try {
    const r = await fetch(`${API_BASE}/api/system/patches`);
    if (r.ok) {
      const d = await r.json();
      installedPatches.value = d.patch_details || (d.patches || []).map(m => ({ module: m, path: m + '.py', size: 0 }));
    }
  } catch (e) {
    console.warn('Load patches failed:', e);
  }
}

async function loadAppVersion() {
  try {
    const r = await fetch(`${API_BASE}/api/system/version`);
    if (r.ok) {
      const d = await r.json();
      if (d.version) appVersion.value = d.version;
    }
  } catch (e) {}
}

function onPatchFileSelected(e) {
  const file = e.target.files[0];
  if (file) {
    selectedPatchFile.value = file;
    patchUploadMsg.value = '';
    // Auto-fill target path if empty
    if (!patchTargetPath.value) {
      patchTargetPath.value = file.name;
    }
  }
}

async function uploadPatch() {
  if (!selectedPatchFile.value) return;
  patchUploading.value = true;
  patchUploadMsg.value = '';
  try {
    const formData = new FormData();
    formData.append('file', selectedPatchFile.value);
    if (patchTargetPath.value) {
      formData.append('target_path', patchTargetPath.value);
    }
    const r = await fetch(`${API_BASE}/api/system/upload_patch`, {
      method: 'POST',
      body: formData,
    });
    const d = await r.json();
    if (d.status === 'success') {
      patchUploadMsg.value = `✅ ${d.message}`;
      toast(`✅ 补丁已部署: ${d.module}`, 'success');
      selectedPatchFile.value = null;
      patchTargetPath.value = '';
      loadPatches();
    } else {
      patchUploadMsg.value = `❌ ${d.message || '上传失败'}`;
      toast(`❌ ${d.message || '上传失败'}`, 'error');
    }
  } catch (e) {
    patchUploadMsg.value = `❌ ${e.message}`;
  } finally {
    patchUploading.value = false;
  }
}

function onUpdateZipSelected(e) {
  const file = e.target.files[0];
  if (file) {
    selectedUpdateZip.value = file;
    updateZipMsg.value = '';
  }
}

async function uploadUpdateZip() {
  if (!selectedUpdateZip.value) return;
  updateZipUploading.value = true;
  updateZipMsg.value = '';
  try {
    const formData = new FormData();
    formData.append('file', selectedUpdateZip.value);
    const r = await fetch(`${API_BASE}/api/system/upload_update`, {
      method: 'POST',
      body: formData,
    });
    const d = await r.json();
    if (d.status === 'success' || d.status === 'ok') {
      updateZipMsg.value = `✅ ${d.message}`;
      toast(`✅ ${d.message}`, 'success');
      selectedUpdateZip.value = null;
      loadPatches();
    } else {
      updateZipMsg.value = `❌ ${d.message || '安装失败'}`;
    }
  } catch (e) {
    updateZipMsg.value = `❌ ${e.message}`;
  } finally {
    updateZipUploading.value = false;
  }
}

async function reloadAllModules() {
  modulesReloading.value = true;
  reloadMsg.value = '';
  try {
    const r = await fetch(`${API_BASE}/api/system/reload_modules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const d = await r.json();
    if (d.status === 'ok') {
      reloadMsg.value = `✅ ${d.summary}`;
      toast(`✅ ${d.summary}`, 'success');
    } else {
      reloadMsg.value = `❌ ${d.message || '重载失败'}`;
    }
  } catch (e) {
    reloadMsg.value = `❌ ${e.message}`;
  } finally {
    modulesReloading.value = false;
  }
}
</script>