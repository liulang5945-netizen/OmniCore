<template>
  <section class="workspace-view">
    <!-- 工具栏 -->
    <div class="ws-toolbar">
      <div class="ws-toolbar-left">
        <h2 class="ws-title"><Laptop :size="16" class="ws-icon" /> {{ t('workspace') || '工作台' }}</h2>
        <span class="ws-path" v-if="workspacePath">{{ workspacePath }}</span>
        <button class="btn-ws btn-path-change" @click="openFolderPicker" title="切换项目路径"><FolderOpen :size="14" /></button>
      </div>
      <div class="ws-toolbar-right">
        <button class="btn-ws" @click="refreshTree" title="刷新目录"><RefreshCw :size="14" /></button>
        <button class="btn-ws" @click="createNewFile" title="新建文件"><FilePlus :size="14" /></button>
        <button class="btn-ws" @click="createNewFolder" title="新建文件夹"><FolderPlus :size="14" /></button>
        <button class="btn-ws" :class="{ active: showTerminal }" @click="showTerminal = !showTerminal" title="终端"><Terminal :size="14" /></button>
      </div>
    </div>

    <!-- 隐藏的文件夹选择器 -->
    <input ref="folderPicker" type="file" webkitdirectory directory style="display:none" @change="onFolderSelected" />

    <!-- 路径切换对话框（备选：手动输入） -->
    <div v-if="showPathDialog" class="path-dialog-overlay" @click.self="showPathDialog = false">
      <div class="path-dialog">
        <h3>切换项目路径</h3>
        <p class="path-dialog-hint">输入新的项目文件夹路径，或从常用路径中快速选择。</p>
        <!-- 常用路径快捷按钮 -->
        <div class="quick-paths" v-if="quickPaths.length">
          <span class="quick-paths-label">常用路径：</span>
          <button v-for="qp in quickPaths" :key="qp.path" class="quick-path-btn" @click="quickSelectPath(qp.path)" :title="qp.path">
            <FolderOpen :size="12" /> {{ qp.label }}
          </button>
        </div>
        <input v-model="newPathInput" class="form-input" placeholder="如 C:\Projects\my-app 或 /home/user/project" @keydown.enter="applyNewPath" />
        <div class="path-dialog-actions">
          <button class="primary-btn" @click="applyNewPath">切换</button>
          <button class="btn-secondary" @click="showPathDialog = false">取消</button>
        </div>
        <p v-if="pathDialogError" class="path-error">{{ pathDialogError }}</p>
      </div>
    </div>

    <div class="ws-main">
      <!-- 左侧目录树 -->
      <div class="ws-sidebar" :style="{ width: sidebarWidth + 'px' }">
        <div class="sidebar-header">
          <span style="display:flex;align-items:center;gap:4px;"><FolderOpen :size="14" /> 文件</span>
        </div>
        <div class="tree-container">
          <div v-if="!fileTree.length" class="tree-empty">
            <FolderOpen class="tree-empty-icon" :size="32" />
            <p>空工作台</p>
          </div>
          <template v-for="node in fileTree" :key="node.path">
            <div
              class="tree-item"
              :class="{ 'is-dir': node.type === 'directory', 'is-open': expandedDirs.has(node.path) }"
              :style="{ paddingLeft: (node.depth * 16 + 8) + 'px' }"
              @click="handleTreeClick(node)"
              @contextmenu.prevent="showContextMenu($event, node)"
            >
              <component :is="node.type === 'directory' ? (expandedDirs.has(node.path) ? FolderOpen : Folder) : getFileIcon(node.name)" :size="14" class="tree-icon" />
              <span class="tree-name">{{ node.name }}</span>
            </div>
          </template>
        </div>
        <!-- 拖拽调整大小 -->
        <div class="resize-handle" @mousedown="startResize"></div>
      </div>

      <!-- 右侧编辑器+终端 -->
      <div class="ws-content">
        <!-- 编辑器区域 -->
        <div class="ws-editor" :style="{ flex: showTerminal ? '1 1 60%' : '1 1 100%' }">
          <MonacoEditor ref="monacoEditor" />
        </div>
        <!-- 终端区域 -->
        <div v-if="showTerminal" class="ws-terminal" :style="{ height: terminalHeight + 'px' }">
          <div class="terminal-resize-handle" @mousedown="startTerminalResize"></div>
          <WebTerminal ref="webTerminal" />
        </div>
      </div>
    </div>

    <!-- 右键菜单 -->
    <div v-if="contextMenu.visible" class="context-menu" :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }" @click="contextMenu.visible = false">
      <div class="ctx-item" @click="openInEditor"><Edit3 :size="14" class="ctx-icon" /> 打开</div>
      <div class="ctx-divider"></div>
      <div class="ctx-item" @click="renameItem"><Edit2 :size="14" class="ctx-icon" /> 重命名</div>
      <div class="ctx-divider"></div>
      <div class="ctx-item ctx-danger" @click="deleteItem"><Trash2 :size="14" class="ctx-icon" /> 删除</div>
    </div>
  </section>
</template>

<script setup>
import { ref, onMounted, watch, inject } from 'vue';
import { useApi, API_BASE } from '../composables/useApi.js';
import { Laptop, RefreshCw, FilePlus, FolderPlus, Terminal, FolderOpen, Folder, FileCode, FileText, Image as ImageIcon, Database, Edit3, Edit2, Trash2 } from 'lucide-vue-next';
import MonacoEditor from '../components/MonacoEditor.vue';
import WebTerminal from '../components/WebTerminal.vue';

const { t } = useApi();
const toast = inject('toast');
const $confirm = inject('$confirm');

const workspacePath = ref('');
const fileTree = ref([]);
const expandedDirs = ref(new Set());
const showTerminal = ref(true);
const sidebarWidth = ref(240);
const terminalHeight = ref(250);
const monacoEditor = ref(null);
const webTerminal = ref(null);
const folderPicker = ref(null);

const contextMenu = ref({ visible: false, x: 0, y: 0, node: null });

// 路径切换对话框
const showPathDialog = ref(false);
const newPathInput = ref('');
const pathDialogError = ref('');

// 常用路径快捷按钮
const quickPaths = ref([
  { label: '桌面', path: '' },
  { label: '文档', path: '' },
  { label: '用户主目录', path: '' },
]);

// 文件夹选择器
function openFolderPicker() {
  if (folderPicker.value) {
    folderPicker.value.value = ''; // 重置以便重复选择同一文件夹
    folderPicker.value.click();
  }
}

function onFolderSelected(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  // 从第一个文件的 relativePath (webkitRelativePath) 提取根文件夹名
  // webkitRelativePath 格式: "folderName/path/to/file.ext"
  // 但我们无法直接获取完整绝对路径，只能获取相对路径
  // 所以我们用文件路径去掉第一段来推导
  const firstFile = files[0];
  const relativePath = firstFile.webkitRelativePath || '';
  const folderName = relativePath.split('/')[0] || '';

  // 浏览器安全限制：无法获取完整绝对路径
  // 因此我们打开手动输入对话框，但预填一个提示
  newPathInput.value = workspacePath.value || '';
  showPathDialog.value = true;
  pathDialogError.value = '';

  // 提示用户
  if (folderName) {
    pathDialogError.value = `浏览器安全限制无法获取完整路径。您选择的文件夹名为 "${folderName}"，请在下方输入该文件夹的完整路径。`;
  }
}

async function applyNewPath() {
  const path = newPathInput.value.trim();
  if (!path) { pathDialogError.value = '请输入路径'; return; }
  pathDialogError.value = '';
  try {
    const r = await fetch(`${API_BASE}/api/workspace/path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const data = await r.json();
    if (r.ok && data.status === 'ok') {
      workspacePath.value = data.path;
      showPathDialog.value = false;
      newPathInput.value = '';
      // 刷新目录树
      expandedDirs.value = new Set();
      loadTree();
      toast('✅ 项目路径已切换', 'success');
    } else {
      pathDialogError.value = data.detail || data.message || '路径设置失败';
    }
  } catch (e) {
    pathDialogError.value = `请求失败: ${e.message}`;
  }
}

// 加载工作台路径
async function loadWorkspacePath() {
  try {
    const r = await fetch(`${API_BASE}/api/workspace/path`);
    if (r.ok) {
      const data = await r.json();
      workspacePath.value = data.path || '';
    }
  } catch (e) {}
}

// 加载目录树
async function loadTree(dir = '') {
  try {
    const r = await fetch(`${API_BASE}/api/workspace/tree`);
    if (r.ok) {
      const data = await r.json();
      fileTree.value = flattenTree(data.tree || [], 0);
    }
  } catch (e) {}
}

// 将树结构扁平化为带深度的列表
function flattenTree(nodes, depth) {
  let result = [];
  for (const node of nodes) {
    node.depth = depth;
    result.push(node);
    if (node.type === 'directory' && expandedDirs.value.has(node.path) && node.children) {
      result = result.concat(flattenTree(node.children, depth + 1));
    }
  }
  return result;
}

function refreshTree() {
  loadTree();
}

// 点击目录树节点
function handleTreeClick(node) {
  if (node.type === 'directory') {
    if (expandedDirs.value.has(node.path)) {
      expandedDirs.value.delete(node.path);
    } else {
      expandedDirs.value.add(node.path);
    }
    // 重新展开树
    loadTree();
  } else {
    openFileInEditor(node.path);
  }
}

// 在编辑器中打开文件
function openFileInEditor(filePath) {
  if (monacoEditor.value) {
    monacoEditor.value.openFile(filePath);
  }
}

function getFileIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase();
  const icons = {
    py: FileCode, js: FileCode, ts: FileCode, json: FileCode, html: FileCode, css: FileCode,
    java: FileCode, go: FileCode, rs: FileCode, cpp: FileCode, c: FileCode, h: FileCode,
    md: FileText, vue: FileCode, txt: FileText, xml: FileCode, yml: FileCode, yaml: FileCode,
    sh: Terminal, bat: Terminal, sql: Database, png: ImageIcon, jpg: ImageIcon, svg: ImageIcon,
  };
  return icons[ext] || FileText;
}

// 右键菜单
function showContextMenu(event, node) {
  contextMenu.value = { visible: true, x: event.clientX, y: event.clientY, node };
}

function openInEditor() {
  if (contextMenu.value.node && contextMenu.value.node.type === 'file') {
    openFileInEditor(contextMenu.value.node.path);
  }
}

async function renameItem() {
  const node = contextMenu.value.node;
  if (!node) return;
  const newName = prompt('新名称:', node.name);
  if (!newName || newName === node.name) return;
  // 简单实现：API 暂无 rename，提示用户
  toast('重命名功能开发中', 'info');
}

async function deleteItem() {
  const node = contextMenu.value.node;
  if (!node) return;
  const ok = await $confirm({ title: '删除确认', message: `确定删除 ${node.name}？`, type: 'danger' });
  if (!ok) return;
  try {
    const r = await fetch(`${API_BASE}/api/workspace/delete/${node.path}`, { method: 'DELETE' });
    if (r.ok) loadTree();
  } catch (e) {}
}

async function createNewFile() {
  const name = prompt('文件名:');
  if (!name) return;
  try {
    await fetch(`${API_BASE}/api/workspace/file`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content: '' }),
    });
    loadTree();
    openFileInEditor(name);
  } catch (e) {}
}

async function createNewFolder() {
  const name = prompt('文件夹名:');
  if (!name) return;
  try {
    await fetch(`${API_BASE}/api/workspace/file`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name + '/.gitkeep', content: '' }),
    });
    loadTree();
  } catch (e) {}
}

// 侧边栏拖拽调整大小
let resizing = false;
function startResize(e) {
  resizing = true;
  const startX = e.clientX;
  const startW = sidebarWidth.value;
  const onMove = (ev) => { sidebarWidth.value = Math.max(150, Math.min(500, startW + ev.clientX - startX)); };
  const onUp = () => { resizing = false; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

// 终端拖拽调整高度
function startTerminalResize(e) {
  const startY = e.clientY;
  const startH = terminalHeight.value;
  const onMove = (ev) => { terminalHeight.value = Math.max(100, Math.min(600, startH - (ev.clientY - startY))); };
  const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

// 关闭右键菜单
document.addEventListener('click', () => { contextMenu.value.visible = false; });

function quickSelectPath(path) {
  newPathInput.value = path;
}

// 加载常用路径
async function loadQuickPaths() {
  try {
    const r = await fetch(`${API_BASE}/api/workspace/quick_paths`);
    if (r.ok) {
      const data = await r.json();
      if (data.paths && data.paths.length) {
        quickPaths.value = data.paths;
      }
    }
  } catch (e) {
    // 静默失败，使用默认路径
  }
}

onMounted(() => {
  loadWorkspacePath();
  loadTree();
  loadQuickPaths();
});
</script>

<style scoped>
.workspace-view { display: flex; flex-direction: column; flex: 1; height: 100%; min-height: 0; overflow: hidden; background: var(--bg-card); }

/* 工具栏 */
.ws-toolbar { display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; background: var(--bg); border-bottom: 1px solid var(--border); min-height: 40px; gap: 12px; }
.ws-toolbar-left { display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1; }
.ws-title { font-size: 14px; font-weight: 600; color: var(--text); margin: 0; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ws-path { font-size: 11px; color: var(--text-muted); font-family: 'Consolas', 'Courier New', monospace; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 2px 8px; background: var(--bg-muted); border-radius: 4px; }
.ws-toolbar-right { display: flex; gap: 2px; }
.btn-ws { background: transparent; border: 1px solid transparent; color: var(--text-secondary); padding: 5px 8px; border-radius: 5px; cursor: pointer; font-size: 13px; transition: all 0.15s ease; display: flex; align-items: center; gap: 4px; }
.btn-ws:hover { background: var(--bg-hover); color: var(--text); border-color: var(--border); }
.btn-ws.active { background: var(--primary-light); color: var(--primary); border-color: rgba(91,122,138,0.2); }

/* 主区域 */
.ws-main { display: flex; flex: 1; overflow: hidden; }

/* 侧边栏文件树 */
.ws-sidebar { display: flex; flex-direction: column; background: var(--bg); border-right: 1px solid var(--border); position: relative; min-width: 150px; }
.sidebar-header { padding: 8px 12px; font-size: 11px; font-weight: 600; color: var(--text-muted); border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.05em; }
.tree-container { flex: 1; overflow-y: auto; padding: 4px 0; }
.tree-empty { text-align: center; padding: 32px 16px; color: var(--text-muted); font-size: 13px; }
.tree-empty-icon { font-size: 2rem; opacity: 0.4; margin-bottom: 8px; display: block; }
.tree-item { display: flex; align-items: center; gap: 6px; padding: 4px 8px; cursor: pointer; font-size: 13px; color: var(--text-secondary); white-space: nowrap; transition: all 0.1s ease; border-left: 2px solid transparent; }
.tree-item:hover { background: var(--bg-hover); color: var(--text); }
.tree-item.active { background: var(--primary-subtle); color: var(--primary); border-left-color: var(--primary); }
.tree-item.is-dir { font-weight: 500; color: var(--text); }
.tree-icon { font-size: 13px; flex-shrink: 0; width: 16px; text-align: center; }
.tree-name { overflow: hidden; text-overflow: ellipsis; }
.resize-handle { position: absolute; top: 0; right: -2px; width: 4px; height: 100%; cursor: col-resize; z-index: 10; transition: background 0.15s ease; }
.resize-handle:hover { background: var(--primary); }

/* 编辑器+终端 */
.ws-content { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: hidden; }
.ws-editor { flex: 1 1 60%; min-height: 0; overflow: hidden; display: flex; flex-direction: column; }
.ws-terminal { flex: 0 0 auto; border-top: 1px solid var(--border); position: relative; }
.terminal-resize-handle { position: absolute; top: -2px; left: 0; right: 0; height: 4px; cursor: row-resize; z-index: 10; transition: background 0.15s ease; }
.terminal-resize-handle:hover { background: var(--primary); }

/* 空状态 */
.ws-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted); gap: 12px; }
.ws-empty-icon { font-size: 3rem; opacity: 0.3; }
.ws-empty-text { font-size: 14px; }
.ws-empty-hint { font-size: 12px; color: var(--text-muted); opacity: 0.7; }

/* 右键菜单 */
.context-menu { position: fixed; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 4px 0; z-index: 9999; min-width: 160px; box-shadow: var(--shadow-lg); animation: ctxIn 0.12s ease; }
.ctx-item { padding: 7px 14px; font-size: 13px; color: var(--text); cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background 0.1s ease; }
.ctx-item:hover { background: var(--bg-hover); }
.ctx-item.ctx-danger:hover { background: var(--danger-light); color: var(--danger); }
.ctx-divider { height: 1px; background: var(--border); margin: 4px 0; }
@keyframes ctxIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }

/* 路径切换对话框 */
.path-dialog-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 10000; animation: fadeIn 0.15s ease; }
.path-dialog { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; min-width: 420px; max-width: 90vw; box-shadow: var(--shadow-xl); }
.path-dialog h3 { margin: 0 0 8px; font-size: 16px; color: var(--text); }
.path-dialog-hint { font-size: 13px; color: var(--text-muted); margin: 0 0 12px; }
.path-dialog .form-input { width: 100%; margin-bottom: 12px; }
.path-dialog-actions { display: flex; gap: 8px; justify-content: flex-end; }
.path-error { color: var(--danger); font-size: 12px; margin: 8px 0 0; }
.btn-path-change { margin-left: 4px; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* 常用路径快捷按钮 */
.quick-paths { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 12px; padding: 8px 10px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
.quick-paths-label { font-size: 12px; color: var(--text-muted); margin-right: 4px; white-space: nowrap; }
.quick-path-btn { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; font-size: 12px; color: var(--text-secondary); background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; cursor: pointer; transition: all 0.15s ease; white-space: nowrap; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }
.quick-path-btn:hover { background: var(--primary-light); color: var(--primary); border-color: var(--primary); }
</style>
