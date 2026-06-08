"""
态极 (Taiji) 统一多模态引擎
神经系统 — 所有模态协同工作，形成完整的多模态智能体

集成视觉、音频、图像生成、语音合成、视频引擎为一个统一的接口。
"""
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from threading import Event

from taiji.core.native_agent import NativeAgentEngine, AgentResult
from taiji.multimodal.vision_encoder import TaijiVisionEncoder
from taiji.multimodal.audio_encoder import TaijiAudioEncoder
from taiji.multimodal.image_generator import TaijiImageGenerator
from taiji.multimodal.voice_generator import TaijiVoiceGenerator
from taiji.multimodal.video_engine import TaijiVideoEngine

logger = logging.getLogger("Taiji.Multimodal")


class TaijiMultimodalEngine:
    """
    态极统一多模态引擎 — 所有器官协同工作

    集成了:
    - 🧠 语言大脑: NativeAgentEngine (感知/记忆/规划/反思)
    - 👁️ 视觉: TaijiVisionEncoder
    - 👂 听觉: TaijiAudioEncoder
    - 🎨 图像生成: TaijiImageGenerator
    - 🗣️ 语音合成: TaijiVoiceGenerator
    - 🎬 视频: TaijiVideoEngine
    """

    # 多模态工具定义
    MULTIMODAL_TOOLS = {
        "describe_image": {
            "description": "描述图像内容",
            "input_format": "image_path",
            "modality": "image",
            "output": "text",
        },
        "generate_image": {
            "description": "根据文字描述生成图像",
            "input_format": "prompt | size",
            "modality": "image",
            "output": "image_path",
        },
        "transcribe_audio": {
            "description": "将音频转为文字（语音识别）",
            "input_format": "audio_path",
            "modality": "audio",
            "output": "text",
        },
        "text_to_speech": {
            "description": "将文字转为语音（语音合成）",
            "input_format": "text | voice",
            "modality": "audio",
            "output": "audio_path",
        },
        "understand_video": {
            "description": "理解视频内容并描述",
            "input_format": "video_path",
            "modality": "video",
            "output": "text",
        },
        "generate_video": {
            "description": "根据文字描述生成视频",
            "input_format": "prompt | duration",
            "modality": "video",
            "output": "video_path",
        },
        "video_to_gif": {
            "description": "将视频转换为GIF动图",
            "input_format": "video_path | fps",
            "modality": "video",
            "output": "image_path",
        },
    }

    def __init__(
        self,
        model, tokenizer, device="cpu",
        workspace_path=None, memory_save_path=None,
        stream_callback=None, max_steps=15,
    ):
        # 语言大脑
        self.text_engine = NativeAgentEngine(
            model, tokenizer, device,
            max_steps=max_steps, stream_callback=stream_callback,
            workspace_path=workspace_path, memory_save_path=memory_save_path,
        )

        # 多模态模块（延迟加载）
        self.vision = TaijiVisionEncoder(hidden_size=model.config.hidden_size, device=device)
        self.audio = TaijiAudioEncoder(hidden_size=model.config.hidden_size, device=device)
        self.image_gen = TaijiImageGenerator()
        self.voice_gen = TaijiVoiceGenerator()
        self.video = TaijiVideoEngine()

        self.max_steps = max_steps
        self.stream_callback = stream_callback
        self.workspace_path = workspace_path
        self.memory_save_path = memory_save_path
        self._cancelled = False

    def generate(self, prompt, tokenizer=None, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """兼容 BaseInferenceEngine"""
        return self.text_engine.inference.generate(prompt, tokenizer, max_new_tokens, temperature, top_p)

    def generate_stream(self, prompt, tokenizer=None, max_new_tokens=512, temperature=0.7, top_p=0.9, stop_event=None):
        return self.text_engine.inference.generate_stream(prompt, tokenizer, max_new_tokens, temperature, top_p, stop_event)

    def run(self, task: str, tool_registry=None) -> AgentResult:
        """
        执行多模态 Agent 任务

        工具会路由到对应的多模态模块或外部工具注册表
        """
        self._cancelled = False
        self.text_engine.reflector.reset()
        result = AgentResult(task=task)
        start_time = time.time()

        self._emit("start", {"task": task, "max_steps": self.max_steps})

        for step_num in range(1, self.max_steps + 1):
            if self._cancelled:
                result.status = "stopped"
                break

            step = AgentStep(step_number=step_num)
            step_start = time.time()

            try:
                self._emit("step_start", {"step": step_num})

                # 构建 prompt
                prompt_parts = []
                mem_context = self.text_engine.memory.get_context_tokens(self.text_engine.inference.tokenizer)
                if mem_context:
                    prompt_parts.append(f"<mem_read>{self.text_engine.inference.tokenizer.decode(mem_context, skip_special_tokens=False)}</mem_read>")
                if self.text_engine.planner.current_plan:
                    prompt_parts.append(self.text_engine.planner.current_plan.to_token_text())

                # 天生联网：自动注入搜索上下文
                if hasattr(self.text_engine, 'web_context_provider') and self.text_engine.web_context_provider:
                    try:
                        web_ctx = self.text_engine.web_context_provider(task, step_num)
                        if web_ctx:
                            prompt_parts.append(f"<web_search>\n{web_ctx}\n</web_search>")
                    except Exception:
                        pass

                for prev_step in result.steps:
                    if prev_step.thought:
                        prompt_parts.append(f"<think>{prev_step.thought}</think>")
                    if prev_step.action:
                        prompt_parts.append(f"<tool_call>{prev_step.action} {json.dumps(prev_step.action_args or {})}")
                    if prev_step.observation:
                        prompt_parts.append(f"<tool_result>{prev_step.observation[:300]}</tool_result>")

                # 构建工具描述
                all_tools = dict(self.MULTIMODAL_TOOLS)
                if tool_registry:
                    try:
                        for t in tool_registry.list_tools(enabled_only=True):
                            all_tools[t.name] = {"description": t.description, "modality": "tool"}
                    except Exception:
                        pass

                tool_desc = "可用工具:\n" + "\n".join(
                    f"  - {k}: {v.get('description', '')}"
                    for k, v in all_tools.items()
                )

                prompt_parts.append(f"[系统] 你是态极 AI 助手。{tool_desc}")
                prompt_parts.append(f"[用户] {task}")
                prompt_parts.append("[助手] ")
                full_prompt = "\n".join(prompt_parts)

                # 原生推理
                react_result = self.text_engine.inference.generate_react_step(
                    full_prompt, max_new_tokens=256, temperature=0.3,
                )

                step.thought = react_result.get("thought", "")

                if not self.text_engine.planner.current_plan and step_num == 1:
                    self.text_engine.planner.create_plan(task, [f"执行: {task[:50]}"])

                if "final_answer" in react_result:
                    step.is_final = True
                    step.observation = react_result["final_answer"]
                    result.steps.append(step)
                    result.final_answer = react_result["final_answer"]
                    result.status = "completed"
                    self.text_engine.memory.auto_write(f"完成任务: {task[:100]}", importance=0.8)
                    self.text_engine.planner.complete_current_step("任务完成")
                    self.text_engine.reflector.evaluate_result("任务", "任务完成")
                    self._emit("final_answer", {"answer": step.observation, "step": step_num})
                    break

                action = react_result.get("action", "")
                action_args = react_result.get("action_args", {})

                if action:
                    step.action = action
                    step.action_args = action_args
                    self._emit("tool_call", {"step": step_num, "tool": action, "args": action_args})

                    # 多模态工具路由
                    if action in self.MULTIMODAL_TOOLS:
                        observation = self._execute_multimodal_tool(action, action_args)
                    elif tool_registry:
                        observation = tool_registry.execute(action, action_args)
                    else:
                        observation = f"工具 {action} 不可用"

                    step.observation = observation
                    reflection = self.text_engine.reflector.evaluate_result(action, observation)
                    if reflection.type.value == "detect" and reflection.should_retry:
                        self.text_engine.memory.auto_write(f"错误: {action} -> {reflection.message}", importance=0.6)
                    elif reflection.type.value == "confirm":
                        self.text_engine.memory.auto_write(f"{action}: {observation[:200]}", importance=0.5)
                    self.text_engine.planner.complete_current_step(f"{action} -> {observation[:100]}")
                    self._emit("observation", {"step": step_num, "tool": action, "result": observation[:500]})
                else:
                    step.is_final = True
                    result.steps.append(step)
                    result.final_answer = step.thought or "(无输出)"
                    result.status = "completed"
                    break

                self.text_engine.memory.consolidate()
            except Exception as e:
                step.error = str(e)
                self.text_engine.reflector.evaluate_result("系统", str(e))
                if self.text_engine.reflector.should_abort():
                    result.status = "error"
                    result.final_answer = f"连续错误过多，任务终止: {e}"
                    break

            step.duration_ms = (time.time() - step_start) * 1000
            result.steps.append(step)
            self._emit("step_end", {"step": step_num, "thought": step.thought[:200], "action": step.action, "duration_ms": step.duration_ms})
        else:
            result.status = "max_steps"
            result.final_answer = f"达到最大推理步数 ({self.max_steps})"

        result.total_duration_ms = (time.time() - start_time) * 1000
        if self.memory_save_path:
            self.text_engine.memory.save(self.memory_save_path)
        self._emit("complete", {"status": result.status, "steps": len(result.steps), "duration_ms": result.total_duration_ms})
        return result

    def _execute_multimodal_tool(self, tool_name: str, args: dict) -> str:
        try:
            input_str = args.get("input", "")
            def extract(key, default_index=0, default_val=""):
                if key in args: return args[key]
                parts = str(input_str).split("|")
                return parts[default_index].strip() if len(parts) >= default_index + 1 else default_val
            
            if tool_name == "describe_image":
                return self.vision.describe_image_simple(extract("image_path", 0, input_str))
            elif tool_name == "generate_image":
                res = self.image_gen.generate(extract("prompt", 0, input_str), extract("size", 1, "512x512"))
                if res.get("success"): self.text_engine.memory.auto_write(f"<img_result>{res.get('path')}</img_result>", importance=0.8)
                return f"[Success] Image: {res.get('path')}" if res.get("success") else f"[Error]: {res.get('error')}"
            elif tool_name == "transcribe_audio":
                return self.audio.transcribe(extract("audio_path", 0, input_str))
            elif tool_name == "text_to_speech":
                res = self.voice_gen.synthesize(extract("text", 0, input_str), extract("voice", 1, "zh-CN-XiaoxiaoNeural"))
                if res.get("success"): self.text_engine.memory.auto_write(f"<tts_result>{res.get('path')}</tts_result>", importance=0.8)
                return f"[Success] Audio: {res.get('path')}" if res.get("success") else f"[Error]: {res.get('error')}"
            elif tool_name == "understand_video":
                return self.video.understand_video(extract("video_path", 0, input_str))
            elif tool_name == "generate_video":
                res = self.video.generate_video(extract("prompt", 0, input_str), int(extract("duration", 1, 4)))
                if res.get("success"): self.text_engine.memory.auto_write(f"<video>{res.get('path')}</video>", importance=0.8)
                return f"[Success] Video: {res.get('path')}" if res.get("success") else f"[Error]: {res.get('error')}"
            elif tool_name == "video_to_gif":
                res = self.video.video_to_gif(extract("video_path", 0, input_str), fps=float(extract("fps", 1, 4.0)))
                if res.get("success"): self.text_engine.memory.auto_write(f"<img_result>{res.get('path')}</img_result>", importance=0.8)
                return f"[Success] GIF: {res.get('path')}" if res.get("success") else f"[Error]: {res.get('error')}"
            return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {str(e)}"
    def register_tools(self, tool_names: list):
        self.text_engine.register_tools(tool_names)

    def get_status(self) -> dict:
        return {
            "type": "multimodal_agent",
            "name": "态极 (Taiji)",
            "version": "4.0.0",
            "memory": self.text_engine.memory.get_stats(),
            "plan": self.text_engine.planner.get_status(),
            "reflection": self.text_engine.reflector.get_stats(),
            "registered_tools": len(self.text_engine.inference.tokenizer._tool_name_to_id),
            "multimodal_tools": list(self.MULTIMODAL_TOOLS.keys()),
        }

    def cancel(self):
        self._cancelled = True

    def _emit(self, event_type: str, data: dict):
        if self.stream_callback:
            try:
                self.stream_callback(event_type, data)
            except Exception:
                pass


from taiji.core.native_agent import AgentStep