<template>
  <section class="dedicated-view">
    <div class="view-header">
      <h2>
        <Zap :size="24" />
        <template v-if="isTaijiModel">🧬 态极 · 生命中心</template>
        <template v-else>{{ t('fine_tuning') }}</template>
      </h2>
      <span v-if="isTaijiModel" style="font-size:0.82rem;color:var(--primary);background:var(--primary-light);padding:4px 12px;border-radius:12px;">
        {{ taijiModelInfo.size }} · {{ taijiModelInfo.config?.num_hidden_layers }}层 · {{ taijiModelInfo.config?.hidden_size }}维
      </span>
    </div>
    <div class="view-body">

      <!-- ==================== 态极模式：生命状态面板 ==================== -->
      <div v-if="isTaijiModel" class="panel-section" style="border-left:3px solid var(--primary);">
        <div class="panel-header">
          <h3><Heart :size="18" style="color:var(--danger);" /> 生命状态</h3>
          <button class="icon-btn btn-sm" @click="loadTaijiLifeStatus(); loadTaijiTimeline();" :disabled="taijiLifeLoading">
            <RefreshCw :size="14" />
          </button>
        </div>
        <div class="panel-content" style="gap:12px;">
          <div style="display:flex;flex-wrap:wrap;gap:16px;font-size:0.88rem;">
            <span>{{ taijiLifeStatus.is_sleeping ? '💤 睡眠中' : '☀️ 清醒' }}</span>
            <span>❤️ 总睡眠: {{ taijiLifeStatus.total_sleeps || 0 }} 次</span>
            <span v-if="taijiLifeStatus.last_sleep">上次睡眠: {{ new Date(taijiLifeStatus.last_sleep).toLocaleString() }}</span>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="train-btn primary" @click="feedTaiji(toast)" :disabled="taijiLifeLoading" style="font-size:0.82rem;padding:6px 14px;">
              <span>🍚</span> 喂养
            </button>
            <button class="train-btn btn-purple" @click="sleepTaiji(toast)" :disabled="taijiLifeLoading" style="font-size:0.82rem;padding:6px 14px;">
              <Moon :size="14" /> 睡眠
            </button>
            <button class="train-btn btn-info" @click="playTaiji(toast)" :disabled="taijiLifeLoading" style="font-size:0.82rem;padding:6px 14px;">
              <Gamepad :size="14" /> 玩耍
            </button>
          </div>
          <!-- Checkpoint 信息 -->
          <div v-if="taijiModelInfo.checkpoints" style="font-size:0.82rem;color:var(--text-muted);">
            <span v-if="taijiModelInfo.checkpoints.has_best">✅ 有最优模型 (best)</span>
            <span v-if="taijiModelInfo.checkpoints.latest_step"> · 最新步数: {{ taijiModelInfo.checkpoints.latest_step }}</span>
            <span v-if="taijiModelInfo.checkpoints.total_checkpoints"> · 共 {{ taijiModelInfo.checkpoints.total_checkpoints }} 个 checkpoint</span>
          </div>
        </div>
      </div>

      <!-- Dataset Upload -->
      <div class="panel-section">
        <div class="panel-header"><h3><Upload :size="18" /> {{ t('train_upload') }}</h3></div>
        <div class="panel-content">
          <FileUploadQueue
            ref="trainUploadRef"
            upload-endpoint="/api/train/upload_dataset"
            accept=".jsonl,.json,.txt,.csv,.md,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.html,.htm,.epub,.rtf,.xml,.log,.py,.js,.ts,.css,.java,.c,.cpp,.sh,.sql,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
            icon="BarChart2" title="训练数据上传" upload-icon="Download"
            :drop-text="t('train_upload')" :accept-hint="t('train_support')"
            success-text="✅ 数据集上传成功"
            @all-uploaded="loadTrainDatasets"
          />
        </div>
      </div>
      <!-- Dataset List -->
      <div class="panel-section">
        <div class="panel-header">
          <h3><PackageIcon :size="18" /> {{ t('train_datasets') }}</h3>
          <div class="train-dataset-actions">
            <button class="primary-btn" @click="loadTrainDatasets"><RefreshCw :size="14" /> {{ t('reload') }}</button>
            <button v-if="selectedDatasets.length > 0" class="danger-btn btn-sm" @click="deleteSelectedDatasets(toast)"><Trash2 :size="14" /> 删除选中 ({{ selectedDatasets.length }})</button>
          </div>
        </div>
        <div class="panel-content">
          <div v-if="trainFiles.length" class="train-files-list">
            <div class="train-select-row">
              <label style="display:flex;align-items:center;gap:4px;font-size:0.82rem;cursor:pointer;">
                <input type="checkbox" :checked="isAllSelected()" @change="toggleSelectAll" style="cursor:pointer;" />
                {{ t('select_all') }}
              </label>
              <span class="text-muted" style="font-size:0.78rem;">已选 {{ selectedDatasets.length }}/{{ trainFiles.length }}</span>
            </div>
            <div v-for="f in trainFiles" :key="f" :class="['train-file-item', { selected: selectedDatasets.includes(f) }]">
              <label style="display:flex;align-items:center;gap:8px;cursor:pointer;flex:1;min-width:0;">
                <input type="checkbox" :checked="selectedDatasets.includes(f)" @change="toggleDataset(f)" style="cursor:pointer;flex-shrink:0;" />
                <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{{ f }}</span>
              </label>
              <div style="display:flex;gap:4px;flex-shrink:0;">
                <button class="primary-btn btn-sm" @click="previewDataset(f)">{{ t('preview') }}</button>
                <button class="delete-btn" @click="deleteTrainFile(f)"><Trash2 :size="14" /></button>
              </div>
            </div>
          </div>
          <p v-else class="text-muted">{{ t('train_no_data') }}</p>
          <div v-if="trainPreview" class="train-preview-panel">
            <div class="panel-header train-preview-header">
              <h3>{{ t('dataset_preview') }} ({{ trainPreview.count || 0 }} {{ t('samples') }})</h3>
            </div>
            <div v-for="(s, i) in (trainPreview.samples || [])" :key="i" class="preview-sample">
              <div class="preview-label">{{ t('instruction') }}:</div>
              <div class="preview-text">{{ s.instruction }}</div>
              <div class="preview-label">{{ t('output') }}:</div>
              <div class="preview-text">{{ s.output }}</div>
            </div>
          </div>
        </div>
      </div>
      <!-- Training Presets (非态极模式) -->
      <div v-if="!isTaijiModel" class="panel-section">
        <div class="panel-header"><h3><Gauge :size="18" /> {{ t('agent_mode') }}</h3></div>
        <div class="panel-content">
          <div class="preset-buttons">
            <button :class="['preset-btn', { active: trainPreset === 'fast' }]" @click="applyPreset('fast')"><Zap :size="14" /> 快速 (r=8, epoch=1)</button>
            <button :class="['preset-btn', { active: trainPreset === 'standard' }]" @click="applyPreset('standard')"><Scale :size="14" /> 标准 (r=16, epoch=3)</button>
            <button :class="['preset-btn', { active: trainPreset === 'quality' }]" @click="applyPreset('quality')"><Gem :size="14" /> 高质量 (r=32, epoch=5)</button>
            <button :class="['preset-btn', { active: trainPreset === 'custom' }]" @click="trainPreset = 'custom'"><Settings2 :size="14" /> 自定义</button>
          </div>
        </div>
      </div>
      <!-- Training Params (非态极模式: LoRA) -->
      <div v-if="!isTaijiModel" class="panel-section">
        <div class="panel-header"><h3><Sliders :size="18" /> {{ t('train_params') }}</h3></div>
        <div class="panel-content params-grid">
          <label class="param-item"><span>{{ t('lora_r') }}</span><input v-model.number="trainParams.lora_r" type="number" class="form-input" min="2" max="128" /></label>
          <label class="param-item"><span>{{ t('lora_alpha') }}</span><input v-model.number="trainParams.lora_alpha" type="number" class="form-input" min="2" max="256" /></label>
          <label class="param-item"><span>{{ t('epochs') }}</span><input v-model.number="trainParams.epochs" type="number" class="form-input" min="1" max="100" /></label>
          <label class="param-item"><span>{{ t('learning_rate') }}</span><input v-model.number="trainParams.learning_rate" type="number" class="form-input" step="0.0001" min="0.00001" /></label>
        </div>
      </div>
      <!-- Training Params (态极模式: 原生训练) -->
      <div v-if="isTaijiModel" class="panel-section">
        <div class="panel-header"><h3><Brain :size="18" style="color:var(--primary);" /> 态极训练参数</h3></div>
        <div class="panel-content params-grid">
          <label class="param-item"><span>Epochs</span><input v-model.number="taijiTrainParams.num_epochs" type="number" class="form-input" min="1" max="100" /></label>
          <label class="param-item"><span>Batch Size</span><input v-model.number="taijiTrainParams.batch_size" type="number" class="form-input" min="1" max="32" /></label>
          <label class="param-item"><span>Learning Rate</span><input v-model.number="taijiTrainParams.learning_rate" type="number" class="form-input" step="0.00001" min="0.00001" /></label>
          <label class="param-item"><span>Max Length</span><input v-model.number="taijiTrainParams.max_length" type="number" class="form-input" min="64" max="2048" /></label>
          <label class="param-item"><span>Save Steps</span><input v-model.number="taijiTrainParams.save_steps" type="number" class="form-input" min="10" max="1000" /></label>
          <label class="param-item"><span>Keep Checkpoints</span><input v-model.number="taijiTrainParams.keep_checkpoints" type="number" class="form-input" min="1" max="10" /></label>
        </div>
      </div>
      <!-- Training Controls -->
      <div class="panel-section">
        <div class="panel-header"><h3><Gamepad2 :size="18" /> 控制</h3></div>
        <div class="panel-content train-controls">
          <button v-if="(trainState === 'idle' || trainState === 'completed') && !isTaijiModel" class="train-btn primary" @click="startTraining(toast)"><Play :size="14" /> {{ t('start_training') }}</button>
          <button v-if="(trainState === 'idle' || trainState === 'completed') && isTaijiModel" class="train-btn primary" @click="startTaijiTraining(toast)" style="background:var(--primary-gradient);"><Brain :size="14" /> 🧬 开始态极微调</button>
          <button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" class="train-btn btn-purple" @click="resumeFromCheckpoint(toast, $confirm)"><RefreshCw :size="14" /> 恢复训练 ({{ pendingCheckpoints.length }})</button>
          <button v-if="trainState === 'running'" class="train-btn btn-amber" @click="pauseTraining(toast)"><Pause :size="14" /> {{ t('pause_training') }}</button>
          <button v-if="trainState === 'paused'" class="train-btn primary" @click="resumeTraining(toast)"><Play :size="14" /> {{ t('resume_training') }}</button>
          <button v-if="trainState === 'running' || trainState === 'paused'" class="train-btn danger" @click="stopTraining(toast)"><StopCircle :size="14" /> {{ t('stop_training') }}</button>
          <button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" class="train-btn btn-info" @click="forcePublish(toast, $confirm)"><PackageIcon :size="14" /> 强制发布</button>
        </div>
      </div>
      <!-- Training Progress -->
      <div v-if="trainState === 'running' || trainState === 'paused'" class="train-progress-area">
        <div v-if="trainDevice.message" class="hardware-diag-banner">
          <Monitor class="hw-diag-icon" :size="16" /><span class="hw-diag-text">{{ trainDevice.message }}</span>
        </div>
        <div class="panel-section">
          <div class="panel-header" style="justify-content:space-between;">
            <h3><TrendingUp :size="18" /> 训练进度</h3>
            <span class="train-progress-pct" :class="{ paused: trainState === 'paused' }">{{ trainProgress }}%</span>
          </div>
          <div class="panel-content">
            <div class="progress-bar-track">
              <div class="progress-bar-fill" :style="{ width: trainProgress + '%' }" :class="{ paused: trainState === 'paused' }">
                <span v-if="trainProgress > 8" class="progress-bar-text">{{ trainProgress }}%</span>
              </div>
            </div>
            <p class="train-status-desc">{{ trainProgressDesc }}</p>
          </div>
        </div>
        <div class="train-metrics-row">
          <div class="train-metric-card"><Clock class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value">{{ fmtTime(trainMetrics.elapsed) }}</span><span class="metric-label">已用时间</span></div></div>
          <div class="train-metric-card"><Hourglass class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value" :class="{ 'metric-highlight': trainMetrics.eta !== null }">{{ fmtTime(trainMetrics.eta) }}</span><span class="metric-label">预计剩余</span></div></div>
          <div class="train-metric-card"><Activity class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value">{{ trainMetrics.current_loss != null ? trainMetrics.current_loss.toFixed(4) : '--' }}</span><span class="metric-label">当前 Loss</span></div></div>
          <div class="train-metric-card"><Zap class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value">{{ trainMetrics.samples_per_sec >= 0.005 ? (trainMetrics.samples_per_sec < 0.1 ? trainMetrics.samples_per_sec.toFixed(2) : trainMetrics.samples_per_sec.toFixed(1)) + '/s' : '--' }}</span><span class="metric-label">吞吐量</span></div></div>
          <div class="train-metric-card"><BookOpen class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value">{{ trainMetrics.epoch }}/{{ trainMetrics.total_epochs }}</span><span class="metric-label">Epoch</span></div></div>
          <div class="train-metric-card"><GraduationCap class="metric-icon" :size="16" /><div class="metric-info"><span class="metric-value">{{ trainMetrics.lr != null ? trainMetrics.lr.toExponential(2) : '--' }}</span><span class="metric-label">学习率</span></div></div>
        </div>
        <div class="panel-section" v-if="trainLoss.length >= 2">
          <div class="panel-header" style="justify-content:space-between;">
            <h3><TrendingUp :size="18" /> Loss 曲线</h3>
            <span class="loss-latest">最新: {{ trainLoss[trainLoss.length-1]?.loss?.toFixed(4) || '--' }}</span>
          </div>
          <div class="panel-content">
            <canvas ref="lossCanvasRef" class="loss-canvas" width="600" height="160"></canvas>
          </div>
        </div>
        <div class="train-log-panel" v-if="trainLog">
          <div class="panel-header train-log-header">
            <h3 style="color:#c9d1d9;"><FileTextIcon :size="18" /> 训练日志</h3>
            <button class="icon-btn btn-sm" @click="clearTrainLog" title="清空日志"><Trash2 :size="14" /> 清空</button>
          </div>
          <pre class="train-log-content" ref="trainLogRef">{{ trainLog }}</pre>
        </div>
      </div>
      <!-- Publish Progress Area -->
      <div v-if="publishingState === 'publishing'" class="train-progress-area" style="margin-top:12px;">
        <div class="panel-section">
          <div class="panel-header" style="justify-content:space-between;">
            <h3><PackageIcon :size="18" /> 发布进度</h3>
            <span class="train-progress-pct">{{ trainProgress }}%</span>
          </div>
          <div class="panel-content">
            <div class="progress-bar-track">
              <div class="progress-bar-fill" :style="{ width: trainProgress + '%' }">
                <span v-if="trainProgress > 8" class="progress-bar-text">{{ trainProgress }}%</span>
              </div>
            </div>
            <p class="train-status-desc">{{ trainProgressDesc }}</p>
            <div class="train-publish-actions">
              <button class="train-btn danger" @click="cancelPublish()">
                <Square :size="14" /> 取消发布
              </button>
            </div>
          </div>
        </div>
        <div class="train-log-panel" v-if="trainLog">
          <div class="panel-header train-log-header">
            <h3 style="color:#c9d1d9;"><FileTextIcon :size="18" /> 发布日志</h3>
            <button class="icon-btn btn-sm" @click="clearTrainLog" title="清空日志"><Trash2 :size="14" /> 清空</button>
          </div>
          <pre class="train-log-content" ref="trainLogRef">{{ trainLog }}</pre>
        </div>
      </div>
      <!-- Publish & Export -->
      <div v-if="trainState === 'completed' || publishingState !== 'idle'" class="panel-section">
        <div class="panel-header"><h3><PackageIcon :size="18" /> 发布与导出</h3></div>
        <div class="panel-content">
          <p class="text-muted" style="margin:0 0 12px;">{{ t('publish_desc') }}</p>
          <div class="train-controls">
            <button class="train-btn primary" @click="publishModel(toast)" :disabled="publishingState !== 'idle'">
              {{ publishingState === 'publishing' ? '发布中...' : t('publish_model_btn') }}
            </button>
            <button class="train-btn btn-purple" @click="exportModelToGGUF(toast, $confirm)" :disabled="publishingState !== 'idle'">
              {{ publishingState === 'publishing' ? '导出中...' : t('export_gguf_btn') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { BarChart2, Download, Monitor, Clock, Hourglass, Activity, Zap, BookOpen, GraduationCap, Trash2, Upload, Package as PackageIcon, RefreshCw, Sliders, Gamepad2, Play, Pause, Square, StopCircle, TrendingUp, FileText as FileTextIcon, Gauge, FlaskConical, Gem, Settings2, Scale, Heart, Moon, Gamepad, Brain } from 'lucide-vue-next';

import { inject, watch, nextTick, onMounted } from 'vue';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi } from '../composables/useApi.js';
import {
  trainState, trainLog, trainLoss, trainPreset, trainFiles,
  selectedDatasets, trainPreview, trainParams,
  publishingState, trainProgress, trainProgressDesc,
  pendingCheckpoints, trainMetrics, trainDevice,
  lossCanvasRef, trainLogRef, fmtTime,
  autoScrollTrainLog, clearTrainLog,
  applyPreset, loadTrainDatasets, previewDataset, deleteTrainFile, deleteSelectedDatasets,
  toggleSelectAll, toggleDataset, isAllSelected,
  startTraining, pauseTraining, resumeTraining, stopTraining,
  loadCheckpoints, resumeFromCheckpoint,
  publishModel, forcePublish, exportModelToGGUF,
  cancelPublish, drawLossChart,
  // 态极模式
  isTaijiModel, taijiModelInfo, taijiLifeStatus, taijiTimeline,
  taijiLifeLoading, taijiTrainParams,
  detectTaijiModel, loadTaijiLifeStatus, loadTaijiTimeline,
  feedTaiji, sleepTaiji, playTaiji, startTaijiTraining,
} from '../composables/useTraining.js';

const toast = inject('toast');
const $confirm = inject('$confirm');
const { t } = useApi();

// Watch loss changes to redraw chart
watch(() => trainLoss.value.length, () => {
  nextTick(() => drawLossChart());
});

// Load datasets on mount + detect Taiji
loadTrainDatasets();
loadCheckpoints();
onMounted(async () => {
  await detectTaijiModel();
});
</script>
