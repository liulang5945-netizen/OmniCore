<template>
  <section class="dedicated-view">
    <div class="view-header"><h2><Bot :size="24" /> {{ t('agent_config') }}</h2></div>
    <div class="view-body">
      <!-- Tab 切换 -->
      <div class="mcp-tabs">
        <button :class="['tab-btn', { active: activeTab === 'tools' }]" @click="activeTab = 'tools'"><Wrench :size="18" /> {{ t('agent_tools') }}</button>
        <button :class="['tab-btn', { active: activeTab === 'marketplace' }]" @click="activeTab = 'marketplace'; loadMarketplace()"><Puzzle :size="18" /> {{ t('mcp_marketplace') }}</button>
        <button :class="['tab-btn', { active: activeTab === 'installed' }]" @click="activeTab = 'installed'; loadInstalled()"><PackageIcon :size="18" /> {{ t('mcp_installed') }}</button>
      </div>

      <!-- Tab 1: 原有工具列表 -->
      <div v-if="activeTab === 'tools'">
        <div class="panel-section">
          <div class="panel-header"><h3><Wrench :size="18" /> {{ t('agent_tools') }}</h3></div>
          <div class="panel-content">
            <div v-if="agentTools.length" class="tools-grid">
              <div v-for="tool in agentTools" :key="tool.name" class="tool-card">
                <Wrench :size="14" class="tool-icon" />
                <div class="tool-info"><span class="tool-name">{{ tool.name }}</span><span class="tool-desc">{{ tool.description }}</span></div>
              </div>
            </div>
            <p v-else class="text-muted">{{ loading ? t('checking_model') : t('no_tools') }}</p>
          </div>
        </div>
        <div class="panel-section">
          <div class="panel-header"><h3><Sliders :size="18" /> {{ t('agent_mode') }}</h3></div>
          <div class="panel-content params-grid">
            <label class="param-item"><span>{{ t('agent_max_iterations') }}</span><input v-model.number="maxIter" type="number" class="form-input" min="1" max="50" @change="save" /></label>
            <label class="param-item"><span>{{ t('agent_temperature') }}</span><input v-model.number="temp" type="number" class="form-input" step="0.1" min="0" max="2" @change="save" /></label>
          </div>
        </div>
        <div class="panel-section">
          <div class="panel-header"><h3><Search :size="18" /> {{ t('search_engine') }}</h3></div>
          <div class="panel-content params-grid">
            <label class="param-item">
              <span>{{ t('search_engine') }}</span>
              <select v-model="engine" class="theme-select" @change="save">
                <option value="DuckDuckGo">DuckDuckGo（免费，无需 Key）</option>
                <option value="Baidu">百度（免费，无需 Key）</option>
                <option value="Bing">必应（免费，无需 Key）</option>
                <option value="Serper">Serper（需要 API Key）</option>
                <option value="Tavily">Tavily（需要 API Key）</option>
                <option value="smart-multi">{{ t('multi_search') }}（自动选择最佳引擎）</option>
              </select>
            </label>
            <label v-if="engine === 'Serper' || engine === 'Tavily'" class="param-item">
              <span>{{ engine }} API Key</span>
              <input v-model="apiKey" type="password" class="form-input" :placeholder="engine === 'Serper' ? 'Serper API Key (serper.dev)' : 'Tavily API Key (tavily.com)'" @change="save" />
            </label>
            <p v-else style="font-size:0.75rem;color:var(--text-muted);margin:4px 0 0;">✅ 当前引擎免费使用，无需配置 API Key</p>
          </div>
        </div>
      </div>

      <!-- Tab 2: MCP 市场 -->
      <div v-if="activeTab === 'marketplace'" class="mcp-marketplace">
        <!-- 搜索栏 -->
        <div class="mcp-search-bar">
          <input v-model="mcpSearch" class="form-input mcp-search-input" :placeholder="t('mcp_search_placeholder')" @input="debounceSearch" />
          <select v-model="mcpCategory" class="theme-select mcp-cat-select" @change="loadMarketplace">
            <option value="">{{ t('mcp_all') }}</option>
            <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
          </select>
          <button class="mcp-btn-refresh" @click="refreshMarketplace" :disabled="refreshing" :title="'从在线源刷新 MCP 服务器列表'">
            <RefreshCw :size="14" :class="{ 'spin': refreshing }" /> {{ refreshing ? '刷新中...' : '同步在线源' }}
          </button>
        </div>

        <!-- 服务器卡片网格 -->
        <div v-if="marketplaceServers.length" class="mcp-grid">
          <div v-for="server in marketplaceServers" :key="server.id" class="mcp-card">
            <div class="mcp-card-header">
              <Puzzle class="mcp-icon" :size="24" />
              <div class="mcp-title-area">
                <span class="mcp-name">{{ server.name }}</span>
                <span class="mcp-category-tag">{{ server.category }}</span>
              </div>
              <span v-if="server.running" class="mcp-status-badge running"><Circle :size="10" fill="currentColor" /> {{ t('mcp_running') }}</span>
              <span v-else-if="server.installed" class="mcp-status-badge installed"><PackageIcon :size="18" /> {{ t('mcp_installed') }}</span>
            </div>
            <p class="mcp-desc">{{ server.description }}</p>
            <div class="mcp-meta">
              <span class="mcp-meta-item"><PackageIcon :size="12" class="mcp-meta-icon" /> {{ server.npm_package }}</span>
              <span class="mcp-meta-item"><Wrench :size="12" class="mcp-meta-icon" /> {{ server.tools_count || '?' }} {{ t('mcp_tools_count') }}</span>
              <span v-if="server.rating" class="mcp-meta-item"><Star :size="12" class="mcp-meta-icon mcp-star" /> {{ server.rating }}</span>
            </div>
            <div class="mcp-card-actions">
              <button v-if="!server.installed" class="btn-primary btn-sm" @click="installServer(server.id)" :disabled="server._installing">
                {{ server._installing ? '...' : t('mcp_install') }}
                <template v-if="!server._installing"><Download :size="12" /></template>
              </button>
              <template v-else>
                <button v-if="!server.running" class="btn-success btn-sm" @click="startServer(server.id)" :disabled="server._starting">
                  {{ server._starting ? '...' : t('mcp_start') }}
                  <template v-if="!server._starting"><Play :size="12" /></template>
                </button>
                <button v-else class="btn-warning btn-sm" @click="stopServer(server.id)"><Square :size="12" /> {{ t('mcp_stop') }}</button>
                <button class="btn-danger btn-sm" @click="uninstallServer(server.id)"><Trash2 :size="12" /> {{ t('mcp_uninstall') }}</button>
              </template>
            </div>
          </div>
        </div>
        <div v-else class="mcp-empty">
          <p class="text-muted">{{ mcpLoading ? t('checking_model') : '暂无匹配结果' }}</p>
        </div>
      </div>

      <!-- Tab 3: 已安装 -->
      <div v-if="activeTab === 'installed'" class="mcp-installed">
        <div v-if="installedServers.length" class="mcp-grid">
          <div v-for="server in installedServers" :key="server.id" class="mcp-card installed-card">
            <div class="mcp-card-header">
              <Puzzle class="mcp-icon" :size="24" />
              <div class="mcp-title-area">
                <span class="mcp-name">{{ server.name || server.id }}</span>
                <span class="mcp-npm">{{ server.npm_package }}</span>
              </div>
              <span v-if="server.running" class="mcp-status-badge running"><Circle :size="10" fill="currentColor" /> {{ t('mcp_running') }}</span>
              <span v-else class="mcp-status-badge stopped"><Circle :size="10" fill="currentColor" /> {{ t('mcp_stopped') }}</span>
            </div>
            <p class="mcp-desc">{{ server.description }}</p>
            <!-- 运行时信息 -->
            <div v-if="server.runtime_info" class="mcp-runtime">
              <div class="mcp-runtime-tools">
                <span class="mcp-runtime-label">{{ t('mcp_tools') }}:</span>
                <span v-for="tool in (server.runtime_info.tools || []).slice(0, 5)" :key="tool.name" class="mcp-tool-tag">{{ tool.name }}</span>
                <span v-if="(server.runtime_info.tools || []).length > 5" class="mcp-tool-more">+{{ server.runtime_info.tools.length - 5 }}</span>
              </div>
            </div>
            <div class="mcp-card-actions">
              <button v-if="!server.running" class="btn-success btn-sm" @click="startServer(server.id)" :disabled="server._starting">
                {{ server._starting ? '...' : t('mcp_start') }}
                <template v-if="!server._starting"><Play :size="12" /></template>
              </button>
              <button v-else class="btn-warning btn-sm" @click="stopServer(server.id)"><Square :size="12" /> {{ t('mcp_stop') }}</button>
              <button v-if="server.running" class="mcp-btn-secondary btn-sm" @click="restartServer(server.id)"><RefreshCw :size="12" /> {{ t('mcp_restart') }}</button>
              <button class="btn-danger btn-sm" @click="uninstallServer(server.id)"><Trash2 :size="12" /> {{ t('mcp_uninstall') }}</button>
            </div>
          </div>
        </div>
        <div v-else class="mcp-empty">
          <p class="text-muted">{{ t('mcp_no_installed') }}</p>
          <button class="btn-primary" @click="activeTab = 'marketplace'; loadMarketplace()"><Puzzle :size="18" /> {{ t('mcp_marketplace') }}</button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Puzzle, Bot, Wrench, Package as PackageIcon, Sliders, Search, Globe, Key, Brain as BrainIcon, Circle, Play, Square, StopCircle, Trash2, RefreshCw, Download, AlertCircle, Star } from 'lucide-vue-next';

import { ref, reactive, inject, watch } from 'vue';
import { useApi, API_BASE } from '../composables/useApi.js';
const { t, saveSettingsToServer } = useApi();
const toast = inject('toast');
const $confirm = inject('$confirm');

// ===== 原有工具配置 =====
const agentTools = ref([]);
const loading = ref(false);
const maxIter = ref(10);
const temp = ref(0.7);
const engine = ref(localStorage.getItem('omnicore_search_engine') || 'DuckDuckGo');
const apiKey = ref(localStorage.getItem(`omnicore_search_key_${engine.value}`) || '');

// 切换搜索引擎时加载对应的 API Key
watch(engine, (newEngine) => {
  apiKey.value = localStorage.getItem(`omnicore_search_key_${newEngine}`) || '';
});

const save = () => {
  localStorage.setItem('omnicore_search_engine', engine.value);
  // 按引擎分别存储 API Key
  localStorage.setItem(`omnicore_search_key_${engine.value}`, apiKey.value);
  // 兼容旧版：同步写入通用 key（取当前选中引擎的 key）
  if (engine.value === localStorage.getItem('omnicore_search_engine')) {
    localStorage.setItem('omnicore_search_key', apiKey.value);
  }
  // 同步搜索引擎配置到后端持久化
  saveSettingsToServer({
    search_engine: engine.value,
    search_key: apiKey.value,
  }).catch(() => {});
};
const load = async () => { loading.value = true; try { const r = await fetch(`${API_BASE}/api/agent/tools`); if (r.ok) agentTools.value = (await r.json()).tools || []; } catch (e) {} finally { loading.value = false; } };
load();

// ===== MCP 市场 =====
const activeTab = ref('tools');
const marketplaceServers = ref([]);
const installedServers = ref([]);
const categories = ref([]);
const mcpSearch = ref('');
const mcpCategory = ref('');
const mcpLoading = ref(false);
const importing = ref(false);
const refreshing = ref(false);

let searchTimer = null;
const debounceSearch = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadMarketplace(), 300);
};

const loadMarketplace = async () => {
  mcpLoading.value = true;
  try {
    const params = new URLSearchParams();
    if (mcpSearch.value) params.set('keyword', mcpSearch.value);
    if (mcpCategory.value) params.set('category', mcpCategory.value);
    const r = await fetch(`${API_BASE}/api/mcp/marketplace?${params}`);
    if (r.ok) {
      const data = await r.json();
      marketplaceServers.value = (data.servers || []).map(s => ({ ...s, _installing: false, _starting: false }));
      categories.value = data.categories || [];
    }
  } catch (e) { console.error('加载MCP市场失败:', e); }
  finally { mcpLoading.value = false; }
};

const refreshMarketplace = async () => {
  refreshing.value = true;
  try {
    const r = await fetch(`${API_BASE}/api/mcp/marketplace/refresh`, { method: 'POST' });
    if (r.ok) {
      toast('正在从在线源刷新 MCP 服务器列表...', 'info');
      // 等几秒让后台完成，然后重新加载
      setTimeout(() => { loadMarketplace(); refreshing.value = false; }, 3000);
    } else {
      refreshing.value = false;
      toast('刷新失败', 'error');
    }
  } catch (e) {
    refreshing.value = false;
    console.error('刷新MCP市场失败:', e);
    toast('刷新失败: ' + e.message, 'error');
  }
};

const loadInstalled = async () => {
  try {
    const r = await fetch(`${API_BASE}/api/mcp/installed`);
    if (r.ok) {
      const data = await r.json();
      installedServers.value = (data.servers || []).map(s => ({ ...s, _starting: false }));
    }
  } catch (e) { console.error('加载已安装MCP失败:', e); }
};

const installServer = async (id) => {
  const server = marketplaceServers.value.find(s => s.id === id);
  if (server) server._installing = true;
  try {
    const r = await fetch(`${API_BASE}/api/mcp/install`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    });
    if (r.ok) {
      const data = await r.json();
      if (data.status === 'ok' || data.status === 'already_installed') {
        loadMarketplace();
      }
    }
  } catch (e) { console.error('安装MCP失败:', e); }
  finally { if (server) server._installing = false; }
};

const startServer = async (id) => {
  const server = marketplaceServers.value.find(s => s.id === id) || installedServers.value.find(s => s.id === id);
  if (server) server._starting = true;
  try {
    const r = await fetch(`${API_BASE}/api/mcp/start`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    });
    if (r.ok) {
      loadMarketplace();
      loadInstalled();
    }
  } catch (e) { console.error('启动MCP失败:', e); }
  finally { if (server) server._starting = false; }
};

const stopServer = async (id) => {
  try {
    const r = await fetch(`${API_BASE}/api/mcp/stop`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    });
    if (r.ok) { loadMarketplace(); loadInstalled(); }
  } catch (e) { console.error('停止MCP失败:', e); }
};

const restartServer = async (id) => {
  try {
    const r = await fetch(`${API_BASE}/api/mcp/restart`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    });
    if (r.ok) { loadInstalled(); }
  } catch (e) { console.error('重启MCP失败:', e); }
};

const uninstallServer = async (id) => {
  const ok = await $confirm({ title: '卸载确认', message: `确定卸载 MCP 服务器 "${id}"？`, type: 'danger' });
  if (!ok) return;
  try {
    const r = await fetch(`${API_BASE}/api/mcp/uninstall`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_id: id }),
    });
    if (r.ok) { loadMarketplace(); loadInstalled(); }
  } catch (e) { console.error('卸载MCP失败:', e); }
};

const importCline = async () => {
  importing.value = true;
  try {
    const r = await fetch(`${API_BASE}/api/mcp/import_cline`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (r.ok) {
      const data = await r.json();
      toast(data.message || '导入完成', 'success');
      loadMarketplace();
      loadInstalled();
    }
  } catch (e) { console.error('导入Cline配置失败:', e); }
  finally { importing.value = false; }
};
</script>

<style scoped>
/* Tab 切换 */
.mcp-tabs { display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 0; }
.tab-btn { padding: 10px 20px; background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--text-secondary); cursor: pointer; font-size: 14px; transition: all 0.2s; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }

/* 搜索栏 */
.mcp-search-bar { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
.mcp-search-input { flex: 1; min-width: 200px; }
.mcp-cat-select { min-width: 120px; }

/* 卡片网格 */
.mcp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; }
.mcp-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; transition: border-color 0.2s; }
.mcp-card:hover { border-color: var(--primary); }
.installed-card { border-left: 3px solid var(--primary); }

.mcp-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.mcp-icon { font-size: 28px; }
.mcp-title-area { flex: 1; }
.mcp-name { font-weight: 600; font-size: 15px; display: block; }
.mcp-category-tag { font-size: 11px; background: var(--bg-muted); padding: 2px 8px; border-radius: 10px; color: var(--text-secondary); }
.mcp-npm { font-size: 11px; color: var(--text-secondary); display: block; }

.mcp-status-badge { font-size: 12px; padding: 3px 10px; border-radius: 12px; white-space: nowrap; }
.mcp-status-badge.running { background: var(--success-light); color: var(--success); }
.mcp-status-badge.installed { background: var(--primary-light); color: var(--primary); }
.mcp-status-badge.stopped { background: var(--danger-light); color: var(--danger); }

.mcp-desc { font-size: 13px; color: var(--text-secondary); margin: 6px 0 10px; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.mcp-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.mcp-meta-item { font-size: 11px; color: var(--text-muted); display: inline-flex; align-items: center; gap: 3px; }
.mcp-meta-icon { color: var(--text-muted); vertical-align: middle; }
.mcp-star { color: var(--warning); }

.mcp-card-actions { display: flex; gap: 6px; flex-wrap: wrap; }

/* 运行时信息 */
.mcp-runtime { margin: 8px 0; padding: 8px; background: var(--bg-muted); border-radius: 6px; }
.mcp-runtime-tools { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.mcp-runtime-label { font-size: 12px; color: var(--text-secondary); margin-right: 4px; }
.mcp-tool-tag { font-size: 11px; background: var(--bg-muted); padding: 2px 6px; border-radius: 4px; color: var(--primary); }
.mcp-tool-more { font-size: 11px; color: var(--text-muted); }

/* 按钮 */
.btn-primary { background: var(--primary-gradient); color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
.btn-primary:hover { opacity: 0.85; }
.btn-success { background: var(--success); color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.btn-warning { background: var(--warning); color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.btn-danger { background: var(--danger); color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.mcp-btn-secondary { background: var(--bg-muted); color: var(--text); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.mcp-btn-refresh { background: var(--primary-light); color: var(--primary); border: 1px solid rgba(99,102,241,0.2); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; display: inline-flex; align-items: center; gap: 4px; transition: var(--transition); }
.mcp-btn-refresh:hover { background: var(--primary-subtle); border-color: var(--primary); }
.mcp-btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.btn-sm { padding: 5px 10px; font-size: 12px; }
button:disabled { opacity: 0.5; cursor: not-allowed; }

.mcp-empty { text-align: center; padding: 40px 20px; }
.text-muted { color: var(--text-muted); }
</style>