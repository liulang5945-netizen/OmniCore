<template>
  <section class="dedicated-view">
    <div class="view-header"><h2><BookOpen :size="24" /> {{ t('kb_management') }}</h2></div>
    <div class="view-body">
      <div class="panel-section">
        <div class="panel-header"><h3><Upload :size="18" /> {{ t('upload_queue') }}</h3></div>
        <div class="panel-content">
          <FileUploadQueue ref="kbUploadRef" upload-endpoint="/api/rag/upload"
            accept=".txt,.pdf,.docx,.doc,.pptx,.xlsx,.xls,.md,.csv,.json,.jsonl,.html,.htm,.epub,.rtf,.xml,.log,.ini,.cfg,.yaml,.yml,.py,.js,.ts,.css,.java,.c,.cpp,.h,.hpp,.sh,.bat,.ps1,.sql,.r,.go,.rs,.swift,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
            icon="FileText" title="知识库上传队列" upload-icon="Library" :drop-text="t('drop_upload')" :accept-hint="t('kb_support_formats')"
            success-text="✅ 上传成功，正在向量化建库" @all-uploaded="loadKBStats" />
        </div>
      </div>
      <div class="panel-section">
        <div class="panel-header"><h3><BarChart3 :size="18" /> {{ t('kb_stats') }}</h3></div>
        <div class="panel-content">
          <div v-if="kbStats" class="kb-stats-row">
            <div class="kb-stat-card"><span class="kb-stat-num">{{ kbStats.doc_count || 0 }}</span><span class="kb-stat-label">{{ t('kb_docs') }}</span></div>
            <div class="kb-stat-card"><span class="kb-stat-num">{{ kbStats.chunk_count || 0 }}</span><span class="kb-stat-label">{{ t('kb_chunks') }}</span></div>
          </div>
          <p v-else class="text-muted">{{ t('kb_empty') }}</p>
        </div>
      </div>
      <div class="panel-section">
        <div class="panel-header"><h3><Search :size="18" /> {{ t('kb_search_test') }}</h3></div>
        <div class="panel-content">
          <div class="kb-search-row">
            <input v-model="kbSearchQuery" class="form-input" style="flex:1;" :placeholder="t('kb_search_placeholder')" @keydown.enter="searchKB" />
            <button class="primary-btn" @click="searchKB">{{ t('search') }}</button>
          </div>
          <div v-if="kbSearching" class="text-muted kb-search-hint">搜索中...</div>
          <div v-else-if="kbResults.length" class="kb-results">
            <div v-for="(r, i) in kbResults" :key="i" class="kb-result-item">
              <p class="kb-result-text">{{ r.content || r.text || r }}</p>
              <span class="kb-result-score" v-if="r.score != null">Score: {{ Number(r.score).toFixed(4) }}</span>
            </div>
          </div>
          <p v-else-if="kbSearched" class="text-muted kb-search-hint">{{ t('kb_no_results') }}</p>
        </div>
      </div>
      <div class="panel-section">
        <div class="panel-header"><h3><SettingsIcon :size="18" /> 检索策略配置</h3></div>
        <div class="panel-content">
          <div class="rag-config-grid">
            <label class="rag-config-item">
              <input type="checkbox" v-model="ragConfig.enable_hybrid" @change="saveRagConfig" />
              <span><Layers :size="14" class="rag-config-icon" /> 混合检索 (Dense + BM25)</span>
              <span class="config-hint">融合语义和关键词两种检索，提升召回率</span>
            </label>
            <label class="rag-config-item">
              <input type="checkbox" v-model="ragConfig.enable_reranker" @change="saveRagConfig" />
              <span><Target :size="14" class="rag-config-icon" /> Cross-Encoder 重排序</span>
              <span class="config-hint">对候选结果精细打分，提升排序精度（首次使用需下载模型）</span>
            </label>
            <label class="rag-config-item">
              <input type="checkbox" v-model="ragConfig.enable_query_rewrite" @change="saveRagConfig" />
              <span><RefreshCw :size="14" class="rag-config-icon" /> 查询改写</span>
              <span class="config-hint">用 LLM 改写用户问题以提升检索效果</span>
            </label>
            <div class="rag-config-item">
              <span><BarChart3 :size="14" class="rag-config-icon" /> 候选数量</span>
              <input type="number" v-model.number="ragConfig.candidate_k" min="5" max="50" class="form-input" style="width:80px;" @change="saveRagConfig" />
              <span class="config-hint">每种检索器返回的候选段落数（5-50）</span>
            </div>
          </div>
          <div v-if="ragStatus" class="rag-status-bar">
            <span>文档: {{ ragStatus.doc_count }}</span>
            <span>段落: {{ ragStatus.chunk_count }}</span>
            <span>Dense: {{ ragStatus.has_embeddings ? '✓' : '✗' }}</span>
            <span>BM25: {{ ragStatus.has_bm25 ? '✓' : '✗' }}</span>
            <span>维度: {{ ragStatus.embed_dim }}</span>
          </div>
        </div>
      </div>
      <div class="panel-section">
        <div class="panel-header"><h3><FolderOpen :size="18" /> {{ t('kb_mounted') }}<button class="danger-btn btn-sm" @click="clearKB" style="margin-left:8px;">{{ t('clear_all') }}</button></h3></div>
        <div class="panel-content">
          <div v-if="kbFiles.length" class="kb-files-list">
            <div v-for="f in kbFiles" :key="f" class="kb-file-item"><span>{{ f }}</span><button class="delete-btn" @click="deleteKBFile(f)"><Trash2 :size="14" /></button></div>
          </div>
          <p v-else class="text-muted">{{ t('kb_empty') }}</p>
        </div>
      </div>
    </div>
  </section>
</template>
<script setup>
import { ref } from 'vue';
import { BookOpen, Upload, BarChart3, Search, FolderOpen, Settings as SettingsIcon, Target, RefreshCw, Trash2, Layers } from 'lucide-vue-next';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi, API_BASE } from '../composables/useApi.js';
const { t } = useApi();
const kbUploadRef = ref(null);
const kbStats = ref(null);
const kbSearchQuery = ref('');
const kbResults = ref([]);
const kbSearched = ref(false);
const kbSearching = ref(false);
const kbFiles = ref([]);
const ragConfig = ref({ enable_hybrid: true, enable_reranker: true, enable_query_rewrite: false, candidate_k: 20 });
const ragStatus = ref(null);
const loadRagConfig = async () => { try { const r = await fetch(`${API_BASE}/api/rag/config`); if (r.ok) { const d = await r.json(); if (d.config) ragConfig.value = { ...ragConfig.value, ...d.config }; } } catch (e) {} };
const loadRagStatus = async () => { try { const r = await fetch(`${API_BASE}/api/rag/status`); if (r.ok) { const d = await r.json(); if (d.status === 'ok') ragStatus.value = d; } } catch (e) {} };
const saveRagConfig = async () => { try { await fetch(`${API_BASE}/api/rag/config`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragConfig.value) }); } catch (e) {} };
loadRagConfig(); loadRagStatus();
const loadKBStats = async () => { try { const r = await fetch(`${API_BASE}/api/rag/stats`); if (r.ok) kbStats.value = await r.json(); } catch (e) {} };
const loadKBFiles = async () => { try { const r = await fetch(`${API_BASE}/api/rag/files`); if (r.ok) { const d = await r.json(); kbFiles.value = d.files || []; } } catch (e) {} };
const searchKB = async () => { if (!kbSearchQuery.value.trim()) return; kbSearched.value = true; kbSearching.value = true; try { const r = await fetch(`${API_BASE}/api/rag/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: kbSearchQuery.value, top_k: 5 }) }); const d = await r.json(); kbResults.value = d.results || []; } catch (e) { kbResults.value = []; } finally { kbSearching.value = false; } };
const clearKB = async () => { try { await fetch(`${API_BASE}/api/rag/clear`, { method: 'DELETE' }); kbStats.value = null; kbFiles.value = []; } catch (e) {} };
const deleteKBFile = async (filename) => { try { await fetch(`${API_BASE}/api/rag/file/${encodeURIComponent(filename)}`, { method: 'DELETE' }); loadKBFiles(); loadKBStats(); loadRagStatus(); } catch (e) {} };
loadKBStats(); loadKBFiles();
</script>
<style scoped>
.rag-config-grid {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.rag-config-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 8px;
  background: var(--bg-secondary, #f8f9fa);
  cursor: pointer;
  transition: background 0.2s;
}
.rag-config-item:hover {
  background: var(--bg-hover, #e9ecef);
}
.rag-config-item input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: #6366f1;
}
.config-hint {
  display: block;
  width: 100%;
  font-size: 12px;
  color: var(--text-muted, #888);
  padding-left: 24px;
  margin-top: -4px;
}
.rag-status-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding: 10px 14px;
  margin-top: 12px;
  border-radius: 8px;
  background: var(--bg-secondary, #f0f4ff);
  font-size: 13px;
  color: var(--text-secondary, #555);
}
.rag-status-bar span {
  white-space: nowrap;
}
.rag-config-icon {
  color: var(--primary);
  vertical-align: middle;
  flex-shrink: 0;
}
</style>
