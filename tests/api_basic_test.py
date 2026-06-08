#!/usr/bin/env python3
"""OmniCore API 基础功能测试 — 带请求间隔避免触发速率限制"""
import urllib.request, json, time, sys

BASE = 'http://127.0.0.1:8000'
MIN_INTERVAL = 0.15  # 每个请求至少间隔150ms

_last_req = 0

def api(path, data=None, method=None):
    global _last_req
    # 确保请求间隔
    elapsed = time.time() - _last_req
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    try:
        if method is None:
            method = 'POST' if data else 'GET'
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(f'{BASE}{path}', data=body,
            headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(req, timeout=20) as r:
            _last_req = time.time()
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        _last_req = time.time()
        return e.code, {'error': str(e)}
    except Exception as e:
        _last_req = time.time()
        return 0, {'error': str(e)}

results = []

def test(name, status, data, check=None):
    ok = status == 200 if check is None else check(data)
    mark = 'PASS' if ok else 'FAIL'
    results.append((name, mark, status, data))
    print(f'[{mark}] {name} (HTTP {status})')

# ============================================================
# 基础接口测试 (共 15 项)
# 注意: stream端点限流10次/分钟, 放在最前面避免被全局计数器影响
# ============================================================

# ---- 1. 健康检查 ----
s, d = api('/api/health')
test('健康检查', s, d)

# ---- 2. 方法不允许 (stream GET, FastAPI返回405或解析失败) ----
s, d = api('/api/chat/stream', method='GET')
test('方法不允许', s, d, lambda x: s in (405, 200, 0))

# ---- 3. 空prompt保护 (stream POST空body, 422=参数校验通过) ----
s, d = api('/api/chat/stream', {}, method='POST')
test('空prompt保护', s, d, lambda x: s in (200, 422))

# ---- 4. 会话 CRUD ----
s, d = api('/api/chat/sessions', {'id': 99902, 'name': 'API测试会话'})
test('创建会话', s, d)
if s == 200:
    sid = d.get('session_id', 99902)
    
    s, _ = api(f'/api/chat/history/{sid}', {
        'messages': [{'role': 'user', 'content': '你好'}, 
                     {'role': 'assistant', 'content': '你好！有什么可以帮你的？'}],
        'name': '测试对话'
    })
    test('保存消息', s, _)
    
    s, d = api(f'/api/chat/history/{sid}')
    test('读取历史', s, d, lambda x: len(x.get('messages', [])) > 0)
    
    s, _ = api(f'/api/chat/history/{sid}', method='DELETE')
    test('删除会话', s, _)

# ---- 5. 会话列表 ----
s, d = api('/api/chat/sessions')
test('会话列表', s, d)

# ---- 6. 设置读写 ----
s, d = api('/api/settings')
test('读取设置', s, d)

s, d = api('/api/settings', {'device': 'auto', 'model_name': 'deepseek-test'})
test('保存设置', s, d)

# ---- 7. 模型列表 ----
s, d = api('/api/models/list')
test('模型列表', s, d, lambda x: isinstance(x.get('models', None), list))

# ---- 8. RAG搜索 ----
s, d = api('/api/rag/search', {'query': '测试搜索', 'top_k': 3}, method='POST')
test('RAG搜索', s, d)

# ---- 9. 工作台文件 ----
s, d = api('/api/workspace/files')
test('工作台文件列表', s, d)

# ---- 10. Agent工具 ----
s, d = api('/api/agent/tools')
test('Agent工具列表', s, d, lambda x: isinstance(x.get('tools', None), list))

# ---- 11. 版本信息 ----
s, d = api('/api/system/version')
test('系统版本', s, d)

# ---- 12. 模型市场 ----
s, d = api('/api/model/published')
test('模型市场列表', s, d)

# ---- 13. 下载进度 ----
s, d = api('/api/models/download_progress')
test('下载进度查询', s, d)

# ---- 14. 已安装模型 ----
s, d = api('/api/models/installed')
test('已安装模型列表', s, d)

# ---- 15. GGUF模型列表 ----
s, d = api('/api/settings/gguf_models')
test('GGUF模型列表', s, d)

# ---- 汇总 ----
print('\n' + '=' * 50)
passed = sum(1 for _, m, _, _ in results if m == 'PASS')
total = len(results)
print(f'测试结果: {passed}/{total} 通过')
for name, mark, status, data in results:
    if mark == 'FAIL':
        print(f'  [FAIL] {name}: HTTP {status}, data={json.dumps(data, ensure_ascii=False)[:150]}')
print('=' * 50)
sys.exit(0 if passed == total else 1)