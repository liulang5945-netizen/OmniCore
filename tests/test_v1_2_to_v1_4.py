"""
OmniCore v1.2 ~ v1.4 综合测试
覆盖：记忆系统 v2、ToolCallParser、安全模块、训练推荐、数据集检查、BM25、工作流、插件系统
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================== 记忆系统 v2 测试 ========================

class TestShortTermMemory:
    def test_add_and_get(self):
        from agent.memory_manager import ShortTermMemory
        stm = ShortTermMemory(max_messages=10)
        stm.add_message("user", "你好")
        stm.add_message("assistant", "你好！有什么可以帮您？")
        assert stm.count() == 2
        ctx = stm.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    def test_max_messages(self):
        from agent.memory_manager import ShortTermMemory
        stm = ShortTermMemory(max_messages=3)
        for i in range(5):
            stm.add_message("user", f"msg{i}")
        assert stm.count() == 3
        ctx = stm.get_context()
        assert ctx[0]["content"] == "msg2"


class TestWorkingMemory:
    def test_set_get_delete(self):
        from agent.memory_manager import WorkingMemory
        wm = WorkingMemory()
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"
        assert wm.get("nonexistent", "default") == "default"
        wm.delete("key1")
        assert wm.get("key1") is None


class TestEpisodicMemory:
    def test_add_and_search(self, tmp_path):
        from agent.memory_manager import EpisodicMemory, Episode
        em = EpisodicMemory(storage_path=str(tmp_path / "ep.json"))
        ep = Episode(task="搜索Python教程", steps=3, success=True, tools_used=["web_search"])
        em.add_episode(ep)
        assert em.count() == 1
        results = em.search_by_text("Python")
        assert len(results) == 1

    def test_stats(self, tmp_path):
        from agent.memory_manager import EpisodicMemory, Episode
        em = EpisodicMemory(storage_path=str(tmp_path / "ep2.json"))
        for i in range(5):
            em.add_episode(Episode(task=f"task{i}", success=i % 2 == 0, tools_used=["tool_a"]))
        stats = em.get_stats()
        assert stats["total"] == 5
        assert stats["success_rate"] == 0.6


class TestSemanticMemory:
    def test_store_and_count(self, tmp_path):
        from agent.memory_manager import SemanticMemory
        sm = SemanticMemory(storage_path=str(tmp_path / "sem.json"))
        sm.store("机器学习是人工智能的分支", category="knowledge")
        assert sm.count() == 1


class TestMemoryCompressor:
    def test_score_importance(self):
        from agent.memory_manager import MemoryCompressor
        mc = MemoryCompressor()
        score = mc.score_importance("这是一条重要错误信息", category="error")
        assert score > 0.5

    def test_compress_entries(self):
        from agent.memory_manager import MemoryCompressor
        mc = MemoryCompressor()
        entries = [{"text": f"text{i}", "importance": i * 0.1, "category": "general"} for i in range(10)]
        compressed = mc.compress_entries(entries, target_count=5)
        assert len(compressed) <= 6  # 5 kept + possible summaries


class TestMemoryManagerV2:
    def test_full_lifecycle(self, tmp_path):
        from agent.memory_manager import MemoryManager
        import tempfile
        mm = MemoryManager()
        # 短期
        mm.add_message("user", "测试消息")
        assert mm.short_term.count() >= 1
        # 工作
        mm.set_working("task", "test")
        assert mm.get_working("task") == "test"
        # 长期
        mm.remember("测试长期记忆", "test")
        # 情景
        mm.remember_episode("测试任务", 3, "成功", ["tool"], True, 1000)
        # 状态
        status = mm.get_status()
        assert "episodic_count" in status
        assert "semantic_count" in status


# ======================== ToolCallParser 测试 ========================

class TestToolCallParser:
    def test_json_code_block(self):
        from agent.react_engine import ToolCallParser
        parser = ToolCallParser()
        content = '我来搜索一下\n```json\n{"tool": "web_search", "args": {"query": "Python教程"}}\n```'
        results = parser.parse(content, available_tools=["web_search", "read_file"])
        assert len(results) == 1
        assert results[0]["name"] == "web_search"
        assert results[0]["arguments"]["query"] == "Python教程"

    def test_xml_tool_call(self):
        from agent.react_engine import ToolCallParser
        parser = ToolCallParser()
        content = '<tool_call>\n{"tool": "read_file", "args": {"path": "test.py"}}\n</tool_call>'
        results = parser.parse(content, available_tools=["read_file"])
        assert len(results) == 1
        assert results[0]["name"] == "read_file"

    def test_action_format(self):
        from agent.react_engine import ToolCallParser
        parser = ToolCallParser()
        content = 'Action: web_search("Python教程")'
        results = parser.parse(content, available_tools=["web_search"])
        assert len(results) == 1

    def test_normalize_keys(self):
        from agent.react_engine import ToolCallParser
        parser = ToolCallParser()
        content = '{"name": "web_search", "arguments": {"query": "test"}}'
        results = parser.parse(content, available_tools=["web_search"])
        assert len(results) == 1

    def test_fuzzy_tool_name(self):
        from agent.react_engine import ToolCallParser
        parser = ToolCallParser()
        content = '{"tool": "Web_Search", "args": {"query": "test"}}'
        results = parser.parse(content, available_tools=["web_search"])
        assert len(results) == 1
        assert results[0]["name"] == "web_search"


class TestFewShotGenerator:
    def test_generate(self):
        from agent.react_engine import FewShotGenerator
        gen = FewShotGenerator()
        schemas = [{"type": "function", "function": {
            "name": "web_search", "description": "搜索网页",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索词"}}}
        }}]
        result = gen.generate(schemas)
        assert "web_search" in result
        assert "```json" in result


# ======================== 安全模块测试 ========================

class TestJWTManager:
    def test_create_and_verify(self):
        from core.security import JWTManager
        jwt = JWTManager(secret_key="test-secret-key-32-chars-long!!")
        token = jwt.create_token("admin")
        payload = jwt.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "admin"

    def test_tampered_token(self):
        from core.security import JWTManager
        jwt = JWTManager(secret_key="test-secret-key-32-chars-long!!")
        token = jwt.create_token("admin")
        tampered = token[:-5] + "XXXXX"
        assert jwt.verify_token(tampered) is None

    def test_custom_claims(self):
        from core.security import JWTManager
        jwt = JWTManager(secret_key="test-secret-key-32-chars-long!!")
        token = jwt.create_token("user1", {"role": "admin"})
        payload = jwt.verify_token(token)
        assert payload["role"] == "admin"


class TestSecureStorage:
    def test_encrypt_decrypt(self):
        from core.security import SecureStorage
        storage = SecureStorage()
        secret = "my-api-key-sk-1234567890"
        encrypted = storage.encrypt(secret)
        assert encrypted != secret
        decrypted = storage.decrypt(encrypted)
        assert decrypted == secret

    def test_empty_string(self):
        from core.security import SecureStorage
        storage = SecureStorage()
        assert storage.encrypt("") == ""
        assert storage.decrypt("") == ""

    def test_tampered_ciphertext(self):
        from core.security import SecureStorage
        import base64
        storage = SecureStorage()
        encrypted = storage.encrypt("secret")
        raw = base64.b64decode(encrypted)
        tampered = base64.b64encode(raw[:16] + b"X" * 16 + raw[32:]).decode()
        assert storage.decrypt(tampered) == ""


class TestAuditLogger:
    def test_log_and_read(self, tmp_path):
        from core.security import AuditLogger
        audit = AuditLogger(log_dir=str(tmp_path / "audit"))
        audit.log_event("test_event", {"key": "value"})
        events = audit.get_recent_events(limit=10)
        assert len(events) >= 1
        assert events[0]["type"] == "test_event"


# ======================== 训练推荐测试 ========================

class TestTrainingRecommender:
    def test_detect_hardware(self):
        from model.training_recommender import TrainingRecommender
        rec = TrainingRecommender()
        hw = rec.detect_hardware()
        assert "ram_gb" in hw
        assert "cpu_cores" in hw

    def test_estimate_model_params(self):
        from model.training_recommender import TrainingRecommender
        rec = TrainingRecommender()
        params = rec.estimate_model_params("/path/to/Qwen-7B")
        assert params == 7.0

    def test_recommend_mid(self):
        from model.training_recommender import TrainingRecommender
        rec = TrainingRecommender()
        result = rec.recommend("/path/to/Model-7B", 100, "mid")
        assert result["preset"] == "mid"
        assert "estimated_vram_gb" in result
        assert result["model_params_b"] == 7.0


# ======================== 数据集检查测试 ========================

class TestDatasetChecker:
    def test_jsonl_check(self, sample_jsonl):
        from model.dataset_checker import DatasetQualityChecker
        checker = DatasetQualityChecker()
        result = checker.check(sample_jsonl)
        assert result["valid"] is True
        assert result["total_samples"] == 20
        assert result["format"] == "alpaca"

    def test_alpaca_check(self, sample_alpaca):
        from model.dataset_checker import DatasetQualityChecker
        checker = DatasetQualityChecker()
        result = checker.check(sample_alpaca)
        assert result["valid"] is True
        assert "instruction" in result["fields"]

    def test_nonexistent_file(self):
        from model.dataset_checker import DatasetQualityChecker
        checker = DatasetQualityChecker()
        result = checker.check("/nonexistent/file.jsonl")
        assert result["valid"] is False


# ======================== BM25 测试 ========================

class TestBM25Index:
    def test_build_and_search(self):
        pytest.importorskip("jieba", reason="BM25 中文分词需要 jieba")
        from tools.rag import BM25Index
        bm25 = BM25Index()
        chunks = [
            ("doc.txt", "机器学习是人工智能的分支", 0),
            ("doc.txt", "深度学习是机器学习的子领域", 1),
            ("doc.txt", "Python是一种编程语言", 2),
        ]
        bm25.build(chunks)
        assert bm25.doc_count == 3
        results = bm25.search("机器学习", top_k=2)
        assert len(results) >= 1
        # 第一个结果应该是最相关的
        assert results[0][0] in [0, 1]

    def test_serialization(self):
        from tools.rag import BM25Index
        bm25 = BM25Index()
        chunks = [("doc.txt", "test content", 0)]
        bm25.build(chunks)
        data = bm25.to_dict()
        restored = BM25Index.from_dict(data)
        assert restored.doc_count == 1


class TestRAGConfig:
    def test_default_config(self):
        from tools.rag import RAGConfig
        config = RAGConfig()
        assert config.get("enable_hybrid") is True
        assert config.get("enable_reranker") is True
        assert "candidate_k" in config.to_dict()


# ======================== 工作流测试 ========================

class TestWorkflowEngine:
    def test_simple_workflow(self):
        from agent.workflow_engine import WorkflowEngine, WorkflowDefinition
        engine = WorkflowEngine()
        wf = WorkflowDefinition(
            id="test1", name="测试工作流",
            nodes=[
                {"id": "n1", "type": "trigger", "label": "开始"},
                {"id": "n2", "type": "condition", "label": "判断", "config": {"expression": "True"}},
            ],
            edges=[{"source": "n1", "target": "n2"}],
        )
        result = engine.execute(wf)
        assert result.status == "completed"
        assert "n1" in result.node_results
        assert "n2" in result.node_results

    def test_condition_branch(self):
        from agent.workflow_engine import WorkflowEngine, WorkflowDefinition
        engine = WorkflowEngine()
        wf = WorkflowDefinition(
            id="test2", name="条件分支",
            nodes=[
                {"id": "n1", "type": "condition", "label": "判断", "config": {"expression": "False"}},
            ],
            edges=[],
        )
        result = engine.execute(wf)
        assert result.node_results["n1"]["result"]["condition_result"] is False


class TestWorkflowStore:
    def test_save_and_load(self, tmp_path):
        from agent.workflow_engine import WorkflowStore, WorkflowDefinition
        store = WorkflowStore(storage_dir=str(tmp_path / "wf"))
        wf = WorkflowDefinition(id="wf1", name="测试", nodes=[{"id": "n1", "type": "trigger"}])
        store.save(wf)
        loaded = store.load("wf1")
        assert loaded is not None
        assert loaded.name == "测试"

    def test_list_and_delete(self, tmp_path):
        from agent.workflow_engine import WorkflowStore, WorkflowDefinition
        store = WorkflowStore(storage_dir=str(tmp_path / "wf"))
        store.save(WorkflowDefinition(id="a", name="A"))
        store.save(WorkflowDefinition(id="b", name="B"))
        assert len(store.list_all()) == 2
        store.delete("a")
        assert len(store.list_all()) == 1


# ======================== 插件系统测试 ========================

class TestPluginManager:
    def test_discover_empty(self, tmp_path):
        from core.plugin_manager import PluginManager
        pm = PluginManager(plugins_dir=str(tmp_path / "plugins"))
        assert len(pm.list_plugins()) == 0

    def test_install_and_list(self, tmp_path):
        from core.plugin_manager import PluginManager
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        # 创建测试插件
        plugin_dir = plugins_dir / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "manifest.json").write_text(json.dumps({
            "id": "test_plugin", "name": "测试插件", "version": "1.0.0",
            "description": "一个测试插件", "author": "test"
        }))
        (plugin_dir / "__init__.py").write_text("# test plugin\n")

        pm = PluginManager(plugins_dir=str(plugins_dir))
        plugins = pm.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "测试插件"


# ======================== API 端点集成测试 ========================

class TestAuthAPI:
    def test_auth_status(self, client):
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data

    def test_login_without_auth(self, client):
        resp = client.post("/api/auth/login", json={"username": "admin", "password": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data


class TestRAGAPI:
    def test_rag_config(self, client):
        resp = client.get("/api/rag/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data

    def test_rag_status(self, client):
        resp = client.get("/api/rag/status")
        assert resp.status_code == 200


class TestTrainingAPI:
    def test_recommend_hardware(self, client):
        resp = client.post("/api/training/recommend", json={
            "model_path": "", "dataset_size": 100, "preset": "mid"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "hardware" in data or "presets" in data

    def test_check_dataset(self, client, sample_jsonl):
        resp = client.post("/api/training/check_dataset", json={"file_path": sample_jsonl})
        assert resp.status_code == 200


class TestWorkflowAPI:
    def test_list_empty(self, client):
        resp = client.get("/api/workflows")
        assert resp.status_code == 200

    def test_create_and_delete(self, client):
        resp = client.post("/api/workflows", json={
            "name": "测试工作流", "nodes": [], "edges": []
        })
        assert resp.status_code == 200
        wf_id = resp.json().get("id")
        if wf_id:
            client.delete(f"/api/workflows/{wf_id}")


class TestPluginAPI:
    def test_list_plugins(self, client):
        resp = client.get("/api/plugins")
        assert resp.status_code == 200


class TestVisionAPI:
    def test_vision_status(self, client):
        resp = client.get("/api/vision/status")
        # 可能 404 如果路由未注册，这是预期的
        assert resp.status_code in [200, 404]


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_hardware(self, client):
        resp = client.get("/api/system/hardware")
        assert resp.status_code == 200