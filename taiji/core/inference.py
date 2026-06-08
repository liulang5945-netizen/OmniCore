"""
Taiji Native Inference Engine

The native inference engine for Taiji (态极), a natively trained AI life form.
Taiji does NOT use HuggingFace generate() — it has its own inference loop.

Features:
- Regular conversation: generate() / generate_stream()
- ReAct steps: generate_react_step() — structured tool call output
- Compatible with BaseInferenceEngine interface

Optimizations applied:
- torch.inference_mode() (faster than no_grad, disables view tracking)
- Optional torch.compile() acceleration (PyTorch 2.0+)
- Batch token decoding (reduces tokenizer call overhead)
- Repetition detection (early stopping on repeated tokens)
- Special token-based stopping (tool_call / final_answer)
"""
import json
import logging
import threading
from typing import Optional, Generator, Dict, Any

import torch
import torch.nn.functional as F

from taiji.config import SPECIAL_TOKENS
from taiji.architecture import ModelSelf
from taiji.tokenizer import ModelSelfTokenizer

logger = logging.getLogger("ModelSelf")


class NativeInferenceEngine:
    """
    ModelSelf 原生推理引擎

    与 BaseInferenceEngine 对齐，但增加原生 ReAct 支持。
    可直接作为 app_state.trainer 使用。
    """

    def __init__(
        self,
        model: ModelSelf,
        tokenizer: ModelSelfTokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._compiled = False

        # FP16 推理（GPU 上 2x 吞吐量，50% 内存减少）
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.half()
            logger.info("FP16 推理已启用")
        elif device == "cpu":
            # CPU 上自动启用 INT8 动态量化（2-4x 加速）
            try:
                from taiji.core.quantization import TaijiQuantizer
                self.model = TaijiQuantizer.quantize_dynamic(self.model)
                logger.info("INT8 动态量化已启用（CPU 加速）")
            except Exception as e:
                logger.debug(f"量化跳过: {e}")

        self.model.eval()

        # 自动启用 torch.compile（首次推理会编译，后续提速 30-50%）
        self.try_compile()

    def try_compile(self):
        """
        尝试对模型应用 torch.compile() 加速（PyTorch 2.0+）。

        首次推理会触发编译（耗时 10-60 秒），后续推理提速 20-40%。
        仅在支持的环境下生效，失败则静默跳过。
        """
        if self._compiled:
            return

        # Windows 上 Inductor 后端需要 cl.exe + OpenMP，大多数环境不具备
        # 直接跳过，对小模型（<1B）影响很小
        import sys
        if sys.platform == "win32":
            logger.info("Windows 环境，跳过 torch.compile（Inductor 需要完整 VS Build Tools）")
            return

        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                logger.info("✅ torch.compile() 已应用，首次推理将触发编译...")
            else:
                logger.debug("torch.compile 不可用（需要 PyTorch 2.0+）")
        except Exception as e:
            logger.debug(f"torch.compile 失败（非关键，回退到 eager 模式）: {e}")

    def generate(
        self,
        prompt: str,
        tokenizer=None,  # 兼容 BaseInferenceEngine 接口，忽略
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vision_features=None,
        audio_features=None,
    ) -> str:
        """
        普通文本生成 (对话模式)。

        与 BaseInferenceEngine.generate() 接口兼容。
        支持可选的视觉/音频特征注入（端到端多模态）。
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self._generate_tokens(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # 只解码新生成的 token
        new_ids = output_ids[0][prompt_len:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # 轻量级自我批评（仅检查安全性和截断）
        result = self._lightweight_critique(result, prompt)

        return result

    def _lightweight_critique(self, response: str, prompt: str) -> str:
        """轻量级自我批评（仅安全和完整性检查）"""
        if not response or len(response) < 10:
            return response
        try:
            from taiji.constitutional_ai import get_constitutional_ai
            critic = get_constitutional_ai()
            context = {"task": prompt[:200]}
            critique = critic.critique(response, context)
            if not critique.passed and critique.revised_response:
                return critique.revised_response
        except Exception:
            pass
        return response

    def generate_stream(
        self,
        prompt: str,
        tokenizer=None,  # 兼容接口
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """
        流式文本生成 (SSE 模式)。

        与 BaseInferenceEngine.generate_stream() 接口兼容。
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        kv_cache = None
        current_ids = input_ids

        # 批量解码缓冲：每 N 个 token 解码一次，减少 tokenizer 调用开销
        _DECODE_BATCH = 8
        token_buf = []

        # 重复检测：连续相同 token 超过阈值时提前停止，避免乱码输出
        _last_token_id = None
        _repeat_count = 0
        _MAX_REPEAT = 20

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if stop_event and stop_event.is_set():
                    break

                output = self.model(
                    current_ids,
                    kv_cache=kv_cache,
                    use_cache=True,
                )
                kv_cache = output.kv_cache

                # 采样下一个 token
                next_logits = output.logits[:, -1, :] / temperature
                next_token = self._sample_token(next_logits, top_p)

                # 重复检测
                _cur_id = next_token[0].item()
                if _cur_id == _last_token_id:
                    _repeat_count += 1
                    if _repeat_count >= _MAX_REPEAT:
                        logger.debug(f"检测到连续 {_repeat_count} 个重复 token，提前停止生成")
                        break
                else:
                    _repeat_count = 0
                _last_token_id = _cur_id

                token_buf.append(next_token[0])

                # 达到批量阈值或遇到特殊 token 时解码
                if len(token_buf) >= _DECODE_BATCH:
                    text = self.tokenizer.decode(
                        torch.stack(token_buf), skip_special_tokens=True
                    )
                    if text:
                        yield text
                    token_buf = []

                # 检查 EOS
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break

                current_ids = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

        # 刷出缓冲区剩余 token
        if token_buf:
            text = self.tokenizer.decode(
                torch.stack(token_buf), skip_special_tokens=True
            )
            if text:
                yield text

    def generate_react_step(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        生成一个 ReAct 步骤。

        输出格式:
        - 有工具调用: {"thought": "...", "action": "tool_name", "action_args": {...}}
        - 最终回答: {"thought": "...", "final_answer": "..."}

        Args:
            prompt: 已注入工具描述和记忆的完整 prompt
            max_new_tokens: 最大生成长度
            temperature: 温度 (ReAct 用较低温度)

        Returns:
            解析后的 ReAct 步骤字典
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            # 生成思考部分
            output_ids = self._generate_tokens(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                stop_on=[SPECIAL_TOKENS["tool_call"], SPECIAL_TOKENS["answer"]],
            )

        # 解析输出
        new_ids = output_ids[0][input_ids.shape[1]:]
        result = self._parse_react_output(new_ids)

        # 自我批评：如果有 final_answer，检查并修正
        if "final_answer" in result:
            result = self._self_critique(result, prompt)

        return result

    def _self_critique(self, result: dict, prompt: str) -> dict:
        """对 final_answer 进行自我批评和修正"""
        try:
            from taiji.constitutional_ai import get_constitutional_ai
            critic = get_constitutional_ai()

            context = {
                "task": prompt[:200],  # 截取任务描述
                "observation": result.get("observation", ""),
            }

            critique = critic.critique(result["final_answer"], context)

            if not critique.passed and critique.revised_response:
                logger.info(
                    f"Self-critique: {len(critique.violations)} violations found, "
                    f"revising response"
                )
                result["final_answer"] = critique.revised_response
                result["_critique_applied"] = True
                result["_violations"] = [
                    v["principle"] for v in critique.violations
                ]

        except Exception as e:
            logger.debug(f"Self-critique skipped: {e}")

        return result

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_on: Optional[list] = None,
        vision_features=None,
        audio_features=None,
    ) -> torch.Tensor:
        """
        自回归生成。

        修复 stop_on 逻辑：遇到终止 token 时先不加入 generated 列表。
        支持可选的视觉/音频特征注入（端到端多模态）。
        """
        kv_cache = None
        current_ids = input_ids
        generated = []
        multimodal_injected = False  # 多模态特征只在第一步注入

        for _ in range(max_new_tokens):
            # 第一步传入多模态特征，后续步不再传入（已在序列中）
            if multimodal_injected:
                output = self.model(current_ids, kv_cache=kv_cache, use_cache=True)
            else:
                output = self.model(
                    current_ids,
                    kv_cache=kv_cache,
                    use_cache=True,
                    vision_features=vision_features,
                    audio_features=audio_features,
                )
                multimodal_injected = True
            kv_cache = output.kv_cache

            next_logits = output.logits[:, -1, :] / max(temperature, 1e-6)
            # 对已生成的 token 施加重复惩罚（向量化，无 Python 循环）
            if generated:
                _prev_ids = torch.stack([g[0] for g in generated if g.dim() == 1 or g.shape[0] == 1])
                _prev_ids = _prev_ids.unique()
                next_logits[0, _prev_ids] /= 1.2  # repetition_penalty
            next_token = self._sample_token(next_logits, top_p)
            token_id = next_token[0].item()

            # 检查停止条件：【先检查，再添加】
            if token_id == self.tokenizer.eos_token_id:
                break
            if stop_on and token_id in stop_on:
                break

            generated.append(next_token)

            current_ids = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token

        if not generated:
            return input_ids

        gen_tensors = [g.unsqueeze(0) if g.dim() == 1 else g for g in generated]
        return torch.cat([input_ids] + gen_tensors, dim=1)

    def _sample_token(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Top-P (Nucleus) 采样 — 优化版本。

        优化点：
        1. 使用 in-place 操作减少内存分配
        2. 避免不必要的 clone
        3. 使用 torch.no_grad 包装
        """
        with torch.no_grad():
            # 温度已在调用方处理，这里直接做 top-p
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 找到超过 top_p 的位置
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # 应用 mask
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits.masked_fill_(indices_to_remove, float("-inf"))

            # 采样
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _parse_react_output(self, token_ids: torch.Tensor) -> Dict[str, Any]:
        """
        解析 ReAct 输出为结构化结果。

        支持:
        - 单工具调用
        - 多工具调用 (返回第一个，后续通过再次调用处理)
        - 最终回答
        - 错误恢复 (不完整输出回退为最终回答)
        """
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids

        thought_parts = []
        actions = []  # 支持收集多个工具调用
        final_answer = None

        i = 0
        while i < len(ids):
            tid = ids[i]

            # 工具调用
            if tid == SPECIAL_TOKENS["tool_call"]:
                # 下一个 token 应该是工具名
                if i + 1 < len(ids):
                    tool_id = ids[i + 1]
                    tool_name = self.tokenizer.get_tool_name(tool_id)
                    if tool_name:
                        # 收集参数直到遇到 tool_result / answer / 另一个 tool_call / 序列结束
                        arg_parts = []
                        i += 2
                        while i < len(ids) and ids[i] not in (
                            SPECIAL_TOKENS["tool_result"],
                            SPECIAL_TOKENS["answer"],
                            SPECIAL_TOKENS["tool_call"],
                        ):
                            arg_parts.append(self.tokenizer.decode([ids[i]], skip_special_tokens=True))
                            i += 1
                        arg_str = "".join(arg_parts).strip()
                        try:
                            action_args = json.loads(arg_str) if arg_str else {}
                        except json.JSONDecodeError:
                            action_args = {"input": arg_str}

                        actions.append({
                            "action": tool_name,
                            "action_args": action_args,
                        })
                        continue
                    else:
                        logger.warning(f"未知工具 ID: {tool_id}，跳过该工具调用")
                i += 1
                continue

            # 跳过 tool_result 标记（由外部注入）
            if tid == SPECIAL_TOKENS["tool_result"]:
                i += 1
                continue

            # 最终回答
            if tid == SPECIAL_TOKENS["answer"]:
                answer_parts = []
                i += 1
                while i < len(ids):
                    answer_parts.append(self.tokenizer.decode([ids[i]], skip_special_tokens=True))
                    i += 1
                final_answer = "".join(answer_parts).strip()
                break

            # 普通 token → 思考内容
            thought_parts.append(self.tokenizer.decode([tid], skip_special_tokens=True))
            i += 1

        thought_text = "".join(thought_parts).strip()

        # 有工具调用：返回第一个（ReAct 引擎会循环调用）
        if actions:
            return {
                "thought": thought_text,
                "action": actions[0]["action"],
                "action_args": actions[0]["action_args"],
                "extra_actions": actions[1:] if len(actions) > 1 else [],
            }

        # 有最终回答
        if final_answer:
            return {
                "thought": thought_text,
                "final_answer": final_answer,
            }

        # 没有结构标记：视为最终回答（错误恢复）
        if thought_text:
            logger.info("未检测到结构标记，将思考内容作为最终回答")
            return {
                "thought": thought_text,
                "final_answer": thought_text,
            }

        # 完全为空
        return {
            "thought": "",
            "final_answer": "(模型未生成有效输出)",
        }

    def register_tools(self, tool_names: list):
        """注册工具名到分词器和模型"""
        for name in tool_names:
            self.tokenizer.register_tool(name)
        self.model.set_num_tools(len(self.tokenizer._tool_name_to_id))
