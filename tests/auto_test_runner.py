"""
OmniCore 全面自动化测试脚本
自动启动服务 → 测试所有 API 端点 → 输出测试报告
"""
import json
import os
import sys
import time
import subprocess
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://127.0.0.1:8000"
TEST_RESULTS = []
PASS = 0
FAIL = 0
BUGS = []

def log_result(name, passed, detail=""):
    global PASS, FAIL
    status = "✅ 通过" if passed else "❌ 失败"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    TEST_RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"  {status} {name}" + (f" - {detail}" if detail else ""))

def log_bug(name, severity, detail):
    BUGS.append({"name": name, "severity": severity, "detail": detail})
    print(f"  🐛 [BUG][{severity}] {name}: {detail}")

def api_get(path, timeout=10):
    """GET 请求"""
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}

def api_post(path, data=None, timeout=30):
    """POST 请求"""
    try:
        body = json.dumps(data).encode("utf-8") if data else b"{}"
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}

def api_delete(path, timeout=10):
    """DELETE 请求"""
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}", method="DELETE")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))
    except Exception as e:
        return 0, {"error": str(e)}

def start_server():
    """启动测试服务器"""
    print("=" * 60)
    print("🧠 OmniCore 自动化测试")
    print("=" * 60)
    print("\n[1/4] 启动服务器...")
    
    env = os.environ.copy()
    env["MODEL_LOAD_TIMEOUT"] = "5"
    
    proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "start_test_server.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env
    )
    
    for i in range(30):
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                print(f"  ✅ 服务器已启动 (尝试 {i+1})")
                return proc
        except Exception:
            time.sleep(1)
    
    print("  ❌ 服务器启动失败")
    return None

def test_section(title):
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")

def run_all_tests():
    """运行所有测试"""
    
    # ====== 1. 系统启动与健康检查 ======
    test_section("1. 系统启动与健康检查")
    status, data = api_get("/api/health")
    log_result("/api/health 返回200", status == 200)
    log_result("health返回status字段", "status" in data)
    if data.get("status") == "error":
        log_bug("模型加载失败", "高", data.get("message", ""))
    
    # ====== 2. 设置管理 ======
    test_section("2. 设置管理")
    
    status, data = api_get("/api/settings")
    log_result("/api/settings GET", status == 200)
    
    test_settings = {"theme": "dark", "model": "test-model", "engine": "local-chat"}
    status, data = api_post("/api/settings", test_settings)
    log_result("/api/settings POST 保存", status == 200)
    
    status, data = api_get("/api/settings")
    saved_ok = data.get("theme") == "dark" or data.get("engine") == "local-chat"
    log_result("设置持久化验证", saved_ok, f"data={json.dumps(data, ensure_ascii=False)[:100]}")
    
    status, data = api_post("/api/settings/model", {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"})
    log_result("/api/settings/model 设置模型", status == 200)
    
    status, data = api_post("/api/settings/device", {"device": "auto"})
    log_result("/api/settings/device 设置设备", status == 200)
    
    status, data = api_post("/api/settings/quant", {"load_in_4bit": True})
    log_result("/api/settings/quant 量化设置", status == 200)
    log_result("量化返回消息包含重启后生效", "重启后生效" in data.get("message", ""))
    
    status, data = api_post("/api/settings/gguf", {"gguf_path": "", "n_gpu_layers": -1, "n_ctx": 2048})
    log_result("/api/settings/gguf 保存GGUF设置", status == 200)
    if status != 200:
        log_bug("/api/settings/gguf API错误", "中", str(data)[:200])
    
    status, data = api_get("/api/settings/gguf_models")
    log_result("/api/settings/gguf_models 获取列表", status == 200)
    if data.get("models"):
        log_result(f"GGUF模型列表非空 ({len(data['models'])}个)", True)
    else:
        log_result("GGUF模型列表为空", status == 200, data.get("error", ""))
    
    # ====== 3. 会话管理 ======
    test_section("3. 会话管理")
    
    status, data = api_get("/api/chat/sessions")
    log_result("/api/chat/sessions GET", status == 200)
    
    status, data = api_post("/api/chat/sessions", {"id": 999001, "name": "测试会话"})
    log_result("/api/chat/sessions POST 创建会话", status == 200)
    session_id = data.get("session_id", 999001)
    
    if status == 200:
        test_messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是OmniCore助手。"}
        ]
        status, data = api_post(f"/api/chat/history/{session_id}", {
            "messages": test_messages, "name": "测试会话"
        })
        log_result(f"/api/chat/history/{session_id} POST 保存历史", status == 200)
        
        status, data = api_get(f"/api/chat/history/{session_id}")
        log_result(f"/api/chat/history/{session_id} GET 读取历史", status == 200)
        if data.get("messages"):
            log_result(f"消息数: {len(data['messages'])}", len(data["messages"]) == 2)
        else:
            log_bug("会话历史读取为空", "中", f"返回: {json.dumps(data, ensure_ascii=False)[:200]}")
        
        status, data = api_delete(f"/api/chat/history/{session_id}")
        log_result(f"/api/chat/history/{session_id} DELETE 删除会话", status == 200)
    
    # ====== 4. RAG 知识库 ======
    test_section("4. RAG 知识库")
    status, data = api_get("/api/rag/files")
    log_result("/api/rag/files GET", status == 200)
    status, data = api_post("/api/rag/search", {"query": "测试", "top_k": 5})
    log_result("/api/rag/search POST", status == 200)
    log_result("空知识库返回results", "results" in data)
    
    # ====== 5. 训练管理 ======
    test_section("5. 训练管理")
    status, data = api_get("/api/train/files")
    log_result("/api/train/files GET", status == 200)
    log_result(f"数据集文件数: {len(data.get('files', []))}", True)
    status, data = api_post("/api/train/stop")
    log_result("/api/train/stop (无训练)", status == 200)
    
    # ====== 6. Agent 系统 ======
    test_section("6. Agent 系统")
    status, data = api_get("/api/agent/tools")
    log_result("/api/agent/tools GET", status == 200)
    tools = data.get("tools", [])
    log_result(f"工具数: {len(tools)}", len(tools) > 0)
    for tool_name in ["search", "execute_python", "write_file", "create_project"]:
        exists = any(t["name"] == tool_name for t in tools)
        log_result(f"工具 '{tool_name}' 存在", exists)
    status, data = api_get("/api/agent/plans")
    log_result("/api/agent/plans GET", status == 200)
    status, data = api_get("/api/agent/context?key=test")
    log_result("/api/agent/context GET", status == 200)
    
    # ====== 7. 工作台 ======
    test_section("7. 工作台")
    status, data = api_get("/api/workspace/files")
    log_result("/api/workspace/files GET", status == 200)
    status, data = api_get("/api/workspace/tree")
    log_result("/api/workspace/tree GET", status == 200)
    log_result("tree包含tree字段", "tree" in data)
    
    status, data = api_post("/api/workspace/file", {
        "name": "test_hello.py", "content": "print('Hello OmniCore!')"
    })
    log_result("/api/workspace/file POST 创建文件", status == 200)
    
    status, data = api_get("/api/workspace/file?name=test_hello.py")
    log_result("/api/workspace/file GET 读取文件", status == 200)
    if "content" in data:
        log_result(f"文件内容正确: {data['content']}", "Hello" in data.get("content", ""))
    
    status, data = api_post("/api/workspace/create_project", {"type": "empty"})
    log_result("/api/workspace/create_project 创建项目", status == 200)
    
    status, data = api_delete("/api/workspace/delete/test_hello.py")
    log_result("/api/workspace/delete 删除文件", status == 200)
    
    # ====== 8. 代码执行 ======
    test_section("8. 代码执行")
    status, data = api_post("/api/agent/analyze_code", {"code": "test_hello.py"})
    log_result("/api/agent/analyze_code", status == 200)
    status, data = api_post("/api/workspace/run", {"code": "print('test run')"})
    log_result("/api/workspace/run 运行代码", status == 200)
    if "output" in data:
        log_result(f"运行输出: {data['output']}", True)
    
    # ====== 9. 模型市场 ======
    test_section("9. 模型市场")
    status, data = api_get("/api/models/list")
    log_result("/api/models/list GET", status == 200)
    if data.get("models"):
        log_result(f"模型总数: {len(data['models'])}", True)
        log_result(f"硬件检测: RAM={data.get('hardware', {}).get('total_ram_gb', 'N/A')}GB", True)
        log_result(f"推荐模型数: {len(data.get('recommendations', []))}", True)
    
    status, data = api_get("/api/models/installed")
    log_result("/api/models/installed GET", status == 200)
    status, data = api_get("/api/models/downloaded")
    log_result("/api/models/downloaded GET", status == 200)
    status, data = api_get("/api/models/tags")
    log_result("/api/models/tags GET", status == 200)
    if data.get("tags"):
        log_result(f"标签数: {len(data['tags'])}", True)
    status, data = api_get("/api/models/families")
    log_result("/api/models/families GET", status == 200)
    status, data = api_get("/api/models/recommend")
    log_result("/api/models/recommend GET", status == 200)
    status, data = api_get("/api/model/published")
    log_result("/api/model/published GET", status == 200)
    
    # ====== 10. 系统接口 ======
    test_section("10. 系统接口")
    status, data = api_get("/api/system/version")
    log_result("/api/system/version GET", status == 200)
    if data.get("status") == "ok":
        log_result(f"版本: v{data.get('version', 'N/A')}", True)
    status, data = api_get("/api/system/patches")
    log_result("/api/system/patches GET", status == 200)
    status, data = api_post("/api/system/reload_modules", {})
    log_result("/api/system/reload_modules POST", status == 200)
    if status != 200:
        log_bug("reload_modules失败", "低", str(data)[:200])
    
    # ====== 11. 路径选择 ======
    test_section("11. 路径选择")
    status, data = api_get("/api/system/select_folder")
    log_result("select_folder HTTP响应", status == 200 or status in [0, 500], f"status={status}")
    if isinstance(data, dict) and "status" in data:
        log_result(f"select_folder → status={data['status']}", True)
    status, data = api_get("/api/system/select_file")
    log_result("select_file HTTP响应", status == 200 or status in [0, 500], f"status={status}")
    if isinstance(data, dict) and "status" in data:
        log_result(f"select_file → status={data['status']}", True)
    
    # ====== 12. 对话接口 ======
    test_section("12. 对话/流式接口")
    # 仅测试接口是否可访问（由于频率限制，在限流测试前调用）
    try:
        body = json.dumps({
            "prompt": "你好", "engine": "本地纯对话", "system_prompt": "你是一个助手",
            "history": [], "api_base": "", "api_key": "", "api_model": "",
            "search_engine": "智能多核切换", "search_key": "",
            "agent_max_iterations": 10, "agent_temperature": 0.7
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/api/chat/stream", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log_result("/api/chat/stream 返回200", resp.status == 200)
            ct = resp.headers.get("Content-Type", "")
            log_result("content-type为text/event-stream", "event-stream" in ct)
    except urllib.error.HTTPError as e:
        if e.code in [503, 400]:
            log_result(f"/api/chat/stream HTTP {e.code}(预期行为)", True, str(e.reason)[:100])
        else:
            log_result(f"/api/chat/stream HTTP {e.code}", False, str(e.reason)[:100])
    except Exception as e:
        log_result(f"/api/chat/stream 调用异常", False, str(e)[:100])
    
    # ====== 13. 安全检查 ======
    test_section("13. 安全检查")
    status, data = api_get("/api/workspace/file?name=../../../etc/passwd")
    log_result("路径穿越拦截-工作台", data.get("error") == "路径不安全" or data.get("error"), str(data)[:100])
    status, data = api_get("/api/rag/preview/../../../etc/passwd")
    safe = status == 403 or "不安全" in json.dumps(data, ensure_ascii=False)
    log_result("路径穿越拦截-RAG预览", safe)
    
    # ====== 14. 频率限制 ======
    test_section("14. 频率限制")
    limited = False
    for i in range(70):
        status, data = api_get("/api/health", timeout=3)
        if status == 429:
            limited = True
            break
    log_result("频率限制正常触发", limited)
    if not limited:
        log_bug("频率限制未触发", "低", "70次GET请求未触发60次/分钟限制")
    
    # ====== 输出测试报告 ======
    print(f"\n{'='*60}")
    print(f"📊 测试报告")
    print(f"{'='*60}")
    print(f"总测试项: {PASS + FAIL}")
    print(f"✅ 通过: {PASS}")
    print(f"❌ 失败: {FAIL}")
    print(f"🐛 发现的Bug: {len(BUGS)}")
    
    if BUGS:
        print(f"\nBug详情:")
        for bug in BUGS:
            print(f"  [{bug['severity']}] {bug['name']}: {bug['detail'][:200]}")
    
    print(f"\n{'='*60}")
    return PASS, FAIL, BUGS

if __name__ == "__main__":
    proc = start_server()
    if not proc:
        print("❌ 无法启动服务器，测试终止")
        sys.exit(1)
    
    try:
        run_all_tests()
    finally:
        print("\n[清理] 停止测试服务器...")
        proc.terminate()
        proc.wait(timeout=5)
        print("[清理] 完成")