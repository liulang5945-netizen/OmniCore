"""
API 集成测试
使用 FastAPI TestClient 测试各路由模块的端点是否正确注册和响应
"""
import pytest
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _create_test_client():
    """创建测试客户端（延迟导入避免启动时加载模型）"""
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


class TestHealthEndpoints:
    """健康检查端点"""

    def test_health_returns_json(self):
        client = _create_test_client()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_system_hardware(self):
        client = _create_test_client()
        resp = client.get("/api/system/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_system_memory(self):
        client = _create_test_client()
        resp = client.get("/api/system/memory")
        assert resp.status_code == 200


class TestSettingsEndpoints:
    """设置管理端点（从 routes_settings.py）"""

    def test_get_settings(self):
        client = _create_test_client()
        resp = client.get("/api/settings")
        assert resp.status_code == 200

    def test_current_model(self):
        client = _create_test_client()
        resp = client.get("/api/system/current_model")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_memory_refresh(self):
        client = _create_test_client()
        resp = client.post("/api/system/memory/refresh")
        assert resp.status_code == 200


class TestUpdateEndpoints:
    """更新系统端点（从 routes_update.py）"""

    def test_get_version(self):
        client = _create_test_client()
        resp = client.get("/api/system/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_list_patches(self):
        client = _create_test_client()
        resp = client.get("/api/system/patches")
        assert resp.status_code == 200
        data = resp.json()
        assert "patches" in data


class TestModelSwitchEndpoints:
    """模型切换端点（从 routes_model_switch.py）"""

    def test_switch_status(self):
        client = _create_test_client()
        resp = client.get("/api/system/switch_status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_pub_reset(self):
        client = _create_test_client()
        resp = client.post("/api/system/pub_reset")
        assert resp.status_code == 200


class TestAgentWorkspaceEndpoints:
    """工作台端点（从 routes_agent_workspace.py）"""

    def test_workspace_path(self):
        client = _create_test_client()
        resp = client.get("/api/workspace/path")
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data

    def test_workspace_tree(self):
        client = _create_test_client()
        resp = client.get("/api/workspace/tree")
        assert resp.status_code == 200
        data = resp.json()
        assert "tree" in data

    def test_validate_path(self):
        client = _create_test_client()
        resp = client.post("/api/system/validate_path", json={"path": __file__, "type": "file"})
        assert resp.status_code == 200
        assert resp.json().get("status") == "ok"

    def test_network_diagnose(self):
        client = _create_test_client()
        resp = client.get("/api/network/diagnose")
        assert resp.status_code == 200


class TestAgentEndpoints:
    """Agent 核心端点（从 routes_agent.py）"""

    def test_list_tools(self):
        client = _create_test_client()
        resp = client.get("/api/agent/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert len(data["tools"]) > 0

    def test_tool_registry(self):
        client = _create_test_client()
        resp = client.get("/api/agent/tools/registry")
        assert resp.status_code == 200


class TestAgentMemoryEndpoints:
    """记忆系统端点（从 routes_agent_memory.py）"""

    def test_memory_status(self):
        client = _create_test_client()
        resp = client.get("/api/agent/memory/status")
        assert resp.status_code == 200

    def test_memory_working(self):
        client = _create_test_client()
        resp = client.get("/api/agent/memory/working")
        assert resp.status_code == 200


class TestMCPEndpoints:
    """MCP 服务器端点（从 routes_agent_mcp.py）"""

    def test_mcp_installed(self):
        client = _create_test_client()
        resp = client.get("/api/mcp/installed")
        assert resp.status_code == 200

    def test_mcp_status(self):
        client = _create_test_client()
        resp = client.get("/api/mcp/status")
        assert resp.status_code == 200

    def test_mcp_marketplace(self):
        client = _create_test_client()
        resp = client.get("/api/mcp/marketplace")
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])