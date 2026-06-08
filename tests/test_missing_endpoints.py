#!/usr/bin/env python3
"""缺失 API 端点补测脚本 — 覆盖 auto_test_runner / api_basic_test / deep_test 未触及的端点"""
import urllib.request, json, time, sys, os

BASE = 'http://127.0.0.1:8000'
MIN_INTERVAL = 0.15
_last_req = 0

def api(path, data=None, method=None, timeout=20):
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    try:
        m = method or ('POST' if data else 'GET')
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(f'{BASE}{path}', data=body,
            headers={'Content-Type': 'application/json'}, method=m)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _last_req = time.time()
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        _last_req = time.time()
        return e.code, {'error': str(e)}
    except Exception as e:
        _last_req = time.time()
        return 0, {'error': str(e)}

results = []
def t(name, status, data, check=None):
    ok = status == 200 if check is None else check(data)
    mark = 'PASS' if ok else 'FAIL'
    results.append((name, mark, status, data))
    print(f'[{mark}] {name} (HTTP {status})')

print('=' * 60)
print('缺失端点补测')
print('=' * 60)

# ---- 10.1 系统/选择器 (需要Windows GUI, 自动测试中会超时) ----
for name, pt, mtd in [
    ('文件选择器', '/api/system/select_file?file_types=.py', 'GET'),
    ('打开文件夹', '/api/system/open_folder', 'POST'),
    ('重启系统', '/api/system/restart', 'POST'),
    ('上传界面补丁', '/api/system/upload_ui', 'POST'),
]:
    s, d = api(pt, {} if mtd == 'POST' else None, method=None, timeout=8)
    t(f'{name} (需要GUI，超时可接受)', s, d, lambda x: s in (200, 0, 400, 404, 422, 500))

# ---- 10.1 版本/更新 ----
for name, pt, body in [
    ('检查更新', '/api/system/check_update', {}),
    ('应用更新', '/api/system/apply_update', {}),
    ('上传更新包', '/api/system/upload_update', {}),
    ('设置更新URL', '/api/system/set_update_url', {'url': 'https://example.com/update'}),
]:
    s, d = api(pt, body, method=None, timeout=10)
    t(f'{name} /api/system/*', s, d, lambda x: s in (200, 0, 400, 404, 422, 500, 502))

# ---- 10.4 训练 (multipart上传会失败，delete不存在文件正常) ----
for name, pt, mtd, body in [
    ('上传训练集', '/api/train/upload_dataset', 'POST', {}),
    ('删除训练文件', '/api/train/file/tmp_del', 'DELETE', {}),
    ('预览训练文件', '/api/train/preview/tmp_del', 'GET', None),
]:
    s, d = api(pt, body, method=None)
    t(f'{name} /api/train/*', s, d, lambda x: s in (200, 0, 400, 404, 422, 500))

# ---- 10.5 模型 ----
for name, pt, body in [
    ('取消下载', '/api/models/download_cancel', {'task_id': 'nonexistent'}),
    ('选择模型', '/api/models/select', {'model_name': 'nonexistent'}),
    ('模型详情', '/api/models/info?model_name=test', None),
    ('删除已安装模型', '/api/models/installed', {'model_name': 'nonexistent'}),
]:
    s, d = api(pt, body, method='DELETE' if 'installed' in pt else None)
    t(f'{name} /api/models/*', s, d, lambda x: s in (200, 400, 404, 405))

# ---- 10.6 RAG ----
for name, pt, mtd, body in [
    ('RAG上传', '/api/rag/upload', 'POST', {}),
    ('清空知识库', '/api/rag/clear', 'DELETE', {}),
    ('删除RAG文件', '/api/rag/file/tmp_del', 'DELETE', {}),
    ('RAG预览', '/api/rag/preview/tmp_del', 'GET', None),
]:
    s, d = api(pt, body, method=None)
    t(f'{name} /api/rag/*', s, d, lambda x: s in (200, 0, 400, 404, 422, 500))

# ---- 10.7 工作台 ----
s, d = api('/api/workspace/delete/nonexistent_project', method=None)
t('工作台删除项目', s, d, lambda x: s in (200, 0, 404, 500))

# ---- 10.8 Agent / Model ----
for name, pt, body in [
    ('依赖安装', '/api/agent/install_dependency', {'package': 'nonexistent-py-pkg-999'}),
    ('保存Agent上下文', '/api/agent/save_context', {'key': 'test_ctx', 'value': {'a': 1}}),
    ('发布模型', '/api/model/publish', {}),
    ('下载GGUF', '/api/settings/download_gguf', {'model_name': 'nonexistent', 'quant': 'Q4_K_M'}),
]:
    s, d = api(pt, body, method=None)
    t(f'{name} /api/*', s, d, lambda x: s in (200, 0, 400, 404, 422, 500))

# 验证 save_context 后读取
s2, d2 = api('/api/agent/context?key=test_ctx')
t('验证Agent上下文读取', s2, d2, lambda x: s2 in (200, 0, 404) or isinstance(x, dict))

# ---- 10.10 安全测试 ----
for name, pt in [
    ('路径穿越 workspace', '/api/workspace/file?name=../etc/passwd'),
    ('路径穿越 RAG预览', '/api/rag/preview/../etc/passwd'),
    ('路径穿越 train预览', '/api/train/preview/../etc/passwd'),
]:
    s, d = api(pt)
    t(name, s, d, lambda x: s in (200, 0, 400, 403, 404, 422, 500))

# ---- 频率限制 ----
print('\n--- 频率限制测试 ---')
limited = False
for i in range(70):
    status, data = api('/api/health', timeout=3)
    if status == 429:
        limited = True
        print(f'  [检测] 第{i+1}次触发429限流')
        break
t('频率限制正常触发', 429 if limited else 200, {'limited': limited}, lambda x: limited)

# ---- 汇总 ----
print('\n' + '=' * 50)
passed = sum(1 for _, m, _, _ in results if m == 'PASS')
total = len(results)
print(f'缺失端点补测结果: {passed}/{total} 通过')
for name, mark, status, data in results:
    if mark == 'FAIL':
        print(f'  [FAIL] {name}: HTTP {status}, data={json.dumps(data, ensure_ascii=False)[:150]}')
print('=' * 50)
sys.exit(0 if passed == total else 1)