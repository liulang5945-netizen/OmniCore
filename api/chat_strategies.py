import asyncio
import json
import logging
import os
import threading

from core.utils import get_external_path
from agent.agent import run_agent

logger = logging.getLogger("ApiServer.Chat.Strategies")

_NEWLINE = "\n"


def _apply_rag(prompt, app_state):
    if app_state.rag_kb and app_state.rag_kb.chunks:
        context = app_state.rag_kb.search_with_fallback(prompt)
        if context:
            context_str = "\n---\n".join(context)
            return (f"基于以下参考资料回答问题。\n\n"
                    f"【参考资料】\n{context_str}\n\n【问题】\n{prompt}")
    return prompt

async def _stream_agent(request, prompt, app_state, collector):
    full_text = ""
    _ui_settings = {}
    try:
        _settings_path = get_external_path("app_settings.json")
        if os.path.exists(_settings_path):
            with open(_settings_path, "r", encoding="utf-8") as _f:
                _ui_settings = json.load(_f)
    except Exception:
        pass
    
    for chunk in run_agent(
        prompt, request.engine, request.api_base, request.api_key,
        request.api_model, request.search_engine, request.search_key,
        "", _ui_settings, app_state, app_state.get_tokenizer()
    ):
        full_text += chunk
        yield f"data: {chunk.replace(chr(10), _NEWLINE)}\n\n"
        await asyncio.sleep(0.01)
        
    try:
        if collector:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"

async def _stream_cloud(request, prompt, collector):
    if request.api_type == "anthropic":
        from agent.agent import run_anthropic_chat_stream as _cloud_stream
    else:
        from agent.agent import run_api_chat_stream as _cloud_stream
    
    full_text = ""
    for chunk in _cloud_stream(
        prompt, request.history, request.system_prompt,
        request.api_base, request.api_key, request.api_model
    ):
        full_text += chunk
        yield f"data: {chunk.replace(chr(10), _NEWLINE)}\n\n"
        await asyncio.sleep(0.01)
        
    try:
        if collector:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"

async def _stream_local_base(request, prompt, app_state, stop_event, collector):
    from agent.token_optimizer import compress_history
    compressed = compress_history(request.history, max_rounds=3, max_chars_per_round=300)
    context_str = ""
    if compressed:
        context_str = ("【上下文】\n" + "\n".join(f"用户: {u}\n助手: {a}" for u, a in compressed) + "\n\n")

    formatted = (f"{request.system_prompt}\n\n{context_str}"
                 f"### Instruction:\n{prompt}\n### Response:\n")

    trainer = app_state.get_trainer()
    tokenizer = app_state.get_tokenizer()
    full_text = ""
    try:
        for chunk in trainer.generate_stream(formatted, tokenizer, max_new_tokens=512, stop_event=stop_event):
            full_text += chunk
            yield f"data: {chunk.replace(chr(10), _NEWLINE)}\n\n"
            await asyncio.sleep(0.01)
    except Exception as stream_err:
        logger.warning(f"流式生成失败，尝试全量生成: {stream_err}")
        full_text = trainer.generate(formatted, tokenizer, max_new_tokens=512)
        if "### Response:\n" in full_text:
            full_text = full_text.split("### Response:\n", 1)[-1].strip()
        yield f"data: [START]\n\n"
        yield f"data: {full_text.replace(chr(10), _NEWLINE)}\n\n"

    try:
        if full_text and collector:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"

def create_event_generator(request, app_state, collector_factory):
    async def event_generator():
        stop_event = threading.Event()
        try:
            collector = None
            try:
                collector = collector_factory()
            except Exception:
                pass

            prompt = _apply_rag(request.prompt, app_state)

            if "agent" in request.engine:
                async for chunk in _stream_agent(request, prompt, app_state, collector):
                    yield chunk
                return

            if "cloud" in request.engine:
                async for chunk in _stream_cloud(request, prompt, collector):
                    yield chunk
                return

            async for chunk in _stream_local_base(request, prompt, app_state, stop_event, collector):
                yield chunk

        except (GeneratorExit, RuntimeError, asyncio.CancelledError):
            logger.info("推理客户端已断开连接，正在停止生成...")
            stop_event.set()
        except Exception as e:
            logger.error(f"推理生成出错: {e}")
            yield f"data: 生成出错: {e}\n\n"
        finally:
            stop_event.set()

    return event_generator