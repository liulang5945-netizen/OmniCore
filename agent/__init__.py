"""
OmniCore Agent Module
=====================
Provides autonomous reasoning, tool calling, memory system, and multi-agent collaboration.

Core Components:
- tool_registry: Tool registry for standardized tool registration and calling
- react_engine: ReAct reasoning engine (Thought -> Action -> Observation loop)
- memory_manager: Three-layer memory system (short-term/working/long-term)
- multi_agent: Multi-agent collaboration system (role-based task delegation)
- agent_planner: Task plan management
- agent_tools: Built-in tool set (file operations, code analysis, etc.)
- sandbox_executor: Sandboxed code execution
"""

def get_tool_registry():
    from agent.tool_registry import registry
    return registry

def get_react_engine(**kwargs):
    from agent.react_engine import ReActEngine
    return ReActEngine(**kwargs)

def get_agent_controller(**kwargs):
    from agent.react_engine import AgentController
    return AgentController(**kwargs)

def get_memory():
    from agent.memory_manager import memory
    return memory

def get_orchestrator():
    from agent.multi_agent import orchestrator
    return orchestrator

def get_message_bus():
    from agent.multi_agent import message_bus
    return message_bus