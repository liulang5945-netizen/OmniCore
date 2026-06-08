"""
OmniCore 深度测试 - 边缘情况、并发、压力、异常处理
"""
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://127.0.0.1:8000"
RESULTS = []

def log(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))

def get(path, timeout=10):
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode("utf-8"))
        except: return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}

def post(path, data=None, timeout=30):
    try:
        body = json.dumps(data).encode("utf-8") if data else b"{}"
        req = urllib.request.Request(f"{BASE_URL}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode("utf-8"))
        except: return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}

def delete(path, timeout=10):
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}", method="DELETE")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode("utf-8"))
        except: return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}

def sec(title):
    print(f"\n{'─'*60}\n📋 {title}\n{'─'*60}")

def main():
    print("=" * 60)
    print("🔬 OmniCore 深度测试套件")
    print("=" * 60)

    # ─── 1. 边界值测试 ───
    sec("1. 边界值测试")

    # 超长输入
    s, d = post("/api/rag/search", {"query": "测" * 10000, "top_k": 5})
    log("超长查询(10000字)", s == 200 or s == 413, f"HTTP {s}")

    s, d = post("/api/rag/search", {"query": "", "top_k": 5})
    log("空查询 (query='')", s == 200, f"HTTP {s}")

    s, d = post("/api/rag/search", {"query": "test", "top_k": -1})
    log("负top_k=-1", s == 200 or s == 422, f"HTTP {s}")

    s, d = post("/api/rag/search", {"query": "test", "top_k": 99999})
    log("超大top_k=99999", s == 200, f"HTTP {s}")

    # 空设置保存
    s, d = post("/api/settings", {})
    log("空设置保存 {}", s == 200)

    s, d = post("/api/settings", None)
    log("null设置保存", s == 200 or s == 422)

    # 极端值保存
    extreme = {"theme": "X" * 5000, "model": "test", "engine": "local-chat"}
    s, d = post("/api/settings", extreme)
    log("超长主题名(5000字)", s == 200, f"HTTP {s}")

    # ─── 2. 并发测试 ───
    sec("2. 并发测试")

    errors = []
    lock = threading.Lock()
    def concurrent_get():
        try:
            s, d = get("/api/health", timeout=10)
            with lock:
                if s != 200:
                    errors.append(f"Health: HTTP {s}")
        except Exception as e:
            with lock:
                errors.append(str(e))

    threads = [threading.Thread(target=concurrent_get) for _ in range(20)]
    start = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.time() - start
    log(f"20并发健康检查 ({len(errors)}错误, {elapsed:.1f}s)", len(errors) == 0, str(errors)[:100] if errors else f"耗时{elapsed:.1f}s")

    # 并发创建/删除会话
    errors2 = []
    def concurrent_sessions(i):
        try:
            sid = 700000 + i
            s1, d1 = post("/api/chat/sessions", {"id": sid, "name": f"并发会话{i}"})
            if s1 != 200: 
                with lock: errors2.append(f"创建失败 sid={sid}: HTTP {s1}")
            time.sleep(0.05)
            s2, d2 = delete(f"/api/chat/history/{sid}")
            if s2 != 200:
                with lock: errors2.append(f"删除失败 sid={sid}: HTTP {s2}")
        except Exception as e:
            with lock: errors2.append(str(e))

    threads2 = [threading.Thread(target=concurrent_sessions, args=(i,)) for i in range(10)]
    for t in threads2: t.start()
    for t in threads2: t.join()
    log(f"10并发创建+删除会话 ({len(errors2)}错误)", len(errors2) == 0, str(errors2)[:200] if errors2 else "OK")

    # ─── 3. 文件操作边缘测试 ───
    sec("3. 文件操作边缘测试")

    # 空文件名
    s, d = post("/api/workspace/file", {"name": "", "content": "test"})
    log("空文件名创建", s == 422 or (d.get("error", "") != ""), f"HTTP {s}, error={d.get('error','')[:80]}")

    # 特殊字符文件名
    s, d = post("/api/workspace/file", {"name": "test file with space.py", "content": "x"})
    log("含空格文件名", s == 200, f"HTTP {s}")
    if s == 200: delete("/api/workspace/delete/test file with space.py")

    s, d = post("/api/workspace/file", {"name": "中文文件名.py", "content": "# 注释"})
    log("中文文件名", s == 200, f"HTTP {s}")
    if s == 200: delete("/api/workspace/delete/中文文件名.py")

    s, d = post("/api/workspace/file", {"name": "test-dash_ok.py", "content": "# ok"})
    log("含-和_的文件名", s == 200, f"HTTP {s}")
    if s == 200: delete("/api/workspace/delete/test-dash_ok.py")

    # 读取不存在的文件
    s, d = get("/api/workspace/file?name=__nonexistent__.py")
    log("读取不存在文件", d.get("error", "") != "", f"HTTP {s}, error={d.get('error','')[:60]}")

    # 删除不存在的文件
    s, d = delete("/api/workspace/delete/__nonexistent__.py")
    log("删除不存在文件", s == 200 or (d.get("error") not in [None, ""]), f"HTTP {s}")

    # 路径遍历攻击
    payloads = [
        ("../../../windows/system32/config/sam", "深层路径遍历"),
        ("..\\..\\..\\windows\\system.ini", "Windows反斜杠路径"),
        ("....//....//....//etc/passwd", "双点斜杠变体"),
        ("%2e%2e%2fetc%2fpasswd", "URL编码路径"),
    ]
    for path, desc in payloads:
        s, d = get(f"/api/workspace/file?name={path}")
        safe = s != 200 or d.get("error", "") != ""
        log(f"路径攻击拦截: {desc}", safe, f"HTTP {s}, {'拦截✅' if safe else '未拦截❌'}")

    # ─── 4. 会话管理深度测试 ───
    sec("4. 会话管理深度测试")

    # 创建重名会话
    s, d = post("/api/chat/sessions", {"id": 800001, "name": "重名测试"})
    s2, d2 = post("/api/chat/sessions", {"id": 800001, "name": "重名测试2"})
    log("覆盖创建同ID会话", s == 200 and s2 == 200, f"s1={s}, s2={s2}")

    # 创建大量消息
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"消息{i}"} for i in range(50)]
    s, d = post("/api/chat/history/800001", {"messages": msgs, "name": "大量消息"})
    log("保存50条消息", s == 200)

    s, d = get("/api/chat/history/800001")
    log(f"读取50条消息 (实际{len(d.get('messages', []))}条)", len(d.get('messages', [])) == 50)

    # 空消息列表
    s, d = post("/api/chat/history/800001", {"messages": [], "name": "空历史"})
    log("保存空消息列表", s == 200)
    s, d = get("/api/chat/history/800001")
    log(f"空历史读取 ({len(d.get('messages', []))}条)", len(d.get('messages', [])) == 0)

    # 清理
    delete("/api/chat/history/800001")

    # ─── 5. 模型API深度测试 ───
    sec("5. 模型API深度测试")

    s, d = get("/api/models/list")
    if "models" in d:
        models = d["models"]
        log(f"模型列表总数: {len(models)}", len(models) > 0)
        has_required_fields = all("name" in m and "size" in m for m in models) if models else True
        log("模型必填字段齐全", has_required_fields)
        families = set(m.get("family", "") for m in models if m.get("family"))
        log(f"模型系列: {', '.join(sorted(families)[:10])}{'…' if len(families)>10 else ''}", len(families) > 0)

    s, d = get("/api/models/tags")
    tags = d.get("tags", [])
    log(f"标签数: {len(tags)}", len(tags) > 0)
    if tags:
        log(f"标签示例: {', '.join(tags[:5])}", True)

    s, d = get("/api/models/download_progress")
    log("下载进度JSON结构正确", "active_downloads" in d, str(d)[:80])

    # 不存在的模型查询
    s, d = get("/api/models/download_progress?model_id=__invalid__")
    log("查询不存在模型下载进度", s == 200)

    # ─── 6. RAG深度测试 ───
    sec("6. RAG深度测试")

    s, d = get("/api/rag/files")
    log("/api/rag/files 返回列表", "files" in d, f"文件数: {len(d.get('files', []))}")

    # 多语言查询
    for lang, query in [("中文", "人工智能"), ("英文", "artificial intelligence"), ("混合", "AI 学习")]:
        s, d = post("/api/rag/search", {"query": query, "top_k": 3})
        log(f"RAG搜索({lang}): '{query}'", s == 200 and "results" in d)

    # 空文件列表查询
    s, d = post("/api/rag/search", {"query": "what is this", "top_k": 0})
    log("RAG top_k=0", s == 200)

    # ─── 7. Agent深度测试 ───
    sec("7. Agent深度测试")

    s, d = get("/api/agent/tools")
    tools = d.get("tools", [])
    log(f"Agent工具总数: {len(tools)}", len(tools) > 0)
    # 检查每个工具的必要字段
    all_valid = all(
        isinstance(t, dict) and "name" in t and "description" in t
        for t in tools
    )
    log("所有工具包含name+description", all_valid)
    # 列出所有工具名
    tool_names = [t["name"] for t in tools]
    log(f"工具名称: {', '.join(tool_names)}", len(tool_names) >= 4)

    s, d = get("/api/agent/plans")
    log("Agent计划列表", "plans" in d or "sessions" in d)

    s, d = get("/api/agent/context?key=test_key_123")
    log("Agent上下文查询(不存在的key)", s == 200)

    # ─── 8. 系统状态深度测试 ───
    sec("8. 系统状态深度测试")

    s, d = get("/api/system/version")
    log(f"系统版本: v{d.get('version', 'N/A')}", "status" in d)

    s, d = get("/api/system/patches")
    log(f"已应用补丁: {d}", "available_patches" in d or "applied" in d or s == 200)

    # 模块重载
    s, d = post("/api/system/reload_modules", {"modules": ["core.config"]})
    log("模块重载", s == 200, str(d)[:100])

    # ─── 9. 异常输入测试 ───
    sec("9. 异常输入测试")

    # 畸形JSON
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/chat/sessions",
            data=b"not json data", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            log("畸形JSON创建会话", r.status == 422 or r.status == 400, f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        log(f"畸形JSON创建会话", e.code == 422 or e.code == 400, f"HTTP {e.code}")
    except Exception as e:
        log(f"畸形JSON创建会话(异常)", False, str(e)[:80])

    # 缺失Content-Type的POST
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/settings",
            data=b'{"theme":"dark"}', method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            log("缺失Content-Type保存设置", r.status == 422 or r.status == 200, f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        log(f"缺失Content-Type保存设置", e.code == 422, f"HTTP {e.code}")
    except Exception as e:
        log(f"缺失Content-Type保存设置(异常)", False, str(e)[:80])

    # 不存在的路由
    s, d = get("/api/__nonexistent__")
    log("不存在的API路由", s == 404, f"HTTP {s}")

    # 错误的HTTP方法
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/health", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            log("GET端点用POST请求", r.status == 405, f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        log(f"GET端点用POST请求", e.code == 405, f"HTTP {e.code}")

    # ─── 10. 性能基准 ───
    sec("10. 性能基准")
    elapsed_health = []
    for i in range(20):
        start = time.perf_counter()
        get("/api/health", timeout=5)
        elapsed_health.append(time.perf_counter() - start)
    avg = sum(elapsed_health) / len(elapsed_health)
    log(f"健康检查平均响应: {avg*1000:.1f}ms", avg < 1.0, f"最快{min(elapsed_health)*1000:.1f}ms, 最慢{max(elapsed_health)*1000:.1f}ms")

    elapsed_list = []
    for i in range(10):
        start = time.perf_counter()
        get("/api/models/list", timeout=10)
        elapsed_list.append(time.perf_counter() - start)
    avg_list = sum(elapsed_list) / len(elapsed_list)
    log(f"模型列表平均响应: {avg_list*1000:.1f}ms", avg_list < 2.0, f"10次采样")

    # ─── 汇总 ───
    passed = sum(1 for _, p, _ in RESULTS if p)
    failed = sum(1 for _, p, _ in RESULTS if not p)
    print(f"\n{'='*60}")
    print(f"📊 深度测试报告")
    print(f"{'='*60}")
    print(f"总测试项: {len(RESULTS)}")
    print(f"✅ 通过: {passed}  ({passed*100//len(RESULTS) if RESULTS else 0}%)")
    print(f"❌ 失败: {failed}")
    print(f"{'='*60}")

    return passed, failed

if __name__ == "__main__":
    main()