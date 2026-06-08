"""
OmniCore 统一异常定义
提供项目级异常层次结构，替代零散的字符串错误返回
"""


class OmniCoreError(Exception):
    """OmniCore 基础异常"""
    def __init__(self, message: str, code: str = "UNKNOWN", status: int = 500):
        self.message = message
        self.code = code
        self.status = status
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "status": "error",
            "code": self.code,
            "message": self.message,
        }


class ModelNotLoadedError(OmniCoreError):
    """模型未加载"""
    def __init__(self, message: str = "模型尚未加载，请等待加载完成或检查配置"):
        super().__init__(message, code="MODEL_NOT_LOADED", status=503)


class ModelLoadError(OmniCoreError):
    """模型加载失败"""
    def __init__(self, message: str = "模型加载失败"):
        super().__init__(message, code="MODEL_LOAD_FAILED", status=500)


class TrainingInProgressError(OmniCoreError):
    """训练正在进行中"""
    def __init__(self, message: str = "训练正在进行中，请等待完成或中止后再试"):
        super().__init__(message, code="TRAINING_IN_PROGRESS", status=409)


class FileOperationError(OmniCoreError):
    """文件操作失败"""
    def __init__(self, message: str = "文件操作失败"):
        super().__init__(message, code="FILE_OPERATION_ERROR", status=500)


class UnsafePathError(OmniCoreError):
    """不安全的路径访问"""
    def __init__(self, message: str = "路径不安全，拒绝访问"):
        super().__init__(message, code="UNSAFE_PATH", status=403)


class ModelSwitchError(OmniCoreError):
    """模型切换失败"""
    def __init__(self, message: str = "模型切换失败"):
        super().__init__(message, code="MODEL_SWITCH_FAILED", status=500)


class AgentTaskError(OmniCoreError):
    """Agent 任务执行失败"""
    def __init__(self, message: str = "Agent 任务执行失败"):
        super().__init__(message, code="AGENT_TASK_FAILED", status=500)


class MCPError(OmniCoreError):
    """MCP 服务器操作失败"""
    def __init__(self, message: str = "MCP 服务器操作失败"):
        super().__init__(message, code="MCP_ERROR", status=500)