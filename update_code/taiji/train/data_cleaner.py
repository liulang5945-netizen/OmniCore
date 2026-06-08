"""
态极训练数据清洗管道
====================

对训练数据进行多层清洗：
1. 去重（按 task / 首条 user 消息）
2. 过滤模板填充（"..." 占位符）
3. 工具名对齐（映射到 DEFAULT_TOOL_DESC）
4. 质量评分（长度、结构、信息量）
5. 合并输出

用法：
    from taiji.data_cleaner import clean_training_data
    stats = clean_training_data("taiji/training_data", "taiji/training_data/cleaned")
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("Taiji.DataCleaner")


# 工具名映射：数据中的工具名 → DEFAULT_TOOL_DESC 中的工具名
TOOL_NAME_MAPPING = {
    "browse_web": "read_webpage",
    "create_directory": "write_file",
    "delete_file": "write_file",  # 安全起见，映射为 write_file
    "my_capabilities": "query_knowledge",
    "smart_fetch": "read_webpage",
    "run_python": "execute_python",
    "code_execute": "execute_python",
    "execute_code": "execute_python",
    "file_read": "read_local_file",
    "file_write": "write_file",
    "file_edit": "edit_file",
    "web_search": "search",
    "web_read": "read_webpage",
    "pkg_install": "install_dependency",
    "knowledge_learn": "learn_knowledge",
    "knowledge_query": "query_knowledge",
    "cmd_run": "run_command",
    "list_files": "list_directory",
    "dir_list": "list_directory",
}

# 模板填充模式
TEMPLATE_PATTERNS = [
    "...",
    "。。。",
    "（待补充）",
    "（略）",
    "[TODO]",
    "[待完成]",
    "TBD",
    "TODO",
]


def clean_training_data(
    input_dir: str,
    output_dir: str = None,
    quality_threshold: float = 0.3,
) -> dict:
    """
    清洗训练数据的主入口。

    Args:
        input_dir: 输入目录（包含 JSONL 文件）
        output_dir: 输出目录（默认为 input_dir/cleaned）
        quality_threshold: 质量评分阈值（0-1），低于此值的样本被过滤

    Returns:
        清洗统计信息
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "cleaned")

    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "files_processed": 0,
        "total_input": 0,
        "duplicates_removed": 0,
        "templates_removed": 0,
        "low_quality_removed": 0,
        "tools_remapped": 0,
        "total_output": 0,
        "per_file": {},
    }

    # 处理所有 JSONL 文件
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue

        fpath = os.path.join(input_dir, fname)
        file_stats = _clean_file(fpath, output_dir, quality_threshold)
        stats["files_processed"] += 1
        stats["total_input"] += file_stats["input"]
        stats["duplicates_removed"] += file_stats["duplicates"]
        stats["templates_removed"] += file_stats["templates"]
        stats["low_quality_removed"] += file_stats["low_quality"]
        stats["tools_remapped"] += file_stats["tools_remapped"]
        stats["total_output"] += file_stats["output"]
        stats["per_file"][fname] = file_stats

    # 生成清洗报告
    _save_report(stats, output_dir)

    logger.info(
        f"数据清洗完成: {stats['total_input']} → {stats['total_output']} "
        f"(去重 {stats['duplicates_removed']}, 过滤模板 {stats['templates_removed']}, "
        f"低质量 {stats['low_quality_removed']})"
    )

    return stats


def _clean_file(
    input_path: str,
    output_dir: str,
    quality_threshold: float,
) -> dict:
    """清洗单个 JSONL 文件"""
    fname = os.path.basename(input_path)
    stats = {"input": 0, "duplicates": 0, "templates": 0, "low_quality": 0, "tools_remapped": 0, "output": 0}

    # 读取数据
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
                stats["input"] += 1
            except json.JSONDecodeError:
                logger.warning(f"跳过无效 JSON 行: {fname}")

    if not data:
        return stats

    # 判断数据类型
    is_react = "task" in data[0] and "steps" in data[0]
    is_conv = "messages" in data[0]

    # Step 1: 去重
    seen = set()
    deduped = []
    for item in data:
        key = _get_dedup_key(item, is_react)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
        else:
            stats["duplicates"] += 1

    # Step 2: 过滤模板填充
    filtered = []
    for item in deduped:
        if _is_template(item, is_react):
            stats["templates"] += 1
            continue
        filtered.append(item)

    # Step 3: 工具名对齐
    aligned = []
    for item in filtered:
        new_item, remapped = _align_tool_names(item, is_react)
        aligned.append(new_item)
        if remapped:
            stats["tools_remapped"] += 1

    # Step 4: 质量评分过滤
    final = []
    for item in aligned:
        score = _score_quality(item, is_react)
        if score >= quality_threshold:
            final.append(item)
        else:
            stats["low_quality"] += 1

    # 保存清洗后的数据
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats["output"] = len(final)
    logger.info(f"  {fname}: {stats['input']} → {stats['output']} (去重 {stats['duplicates']})")

    return stats


def _get_dedup_key(item: dict, is_react: bool) -> str:
    """获取去重键"""
    if is_react:
        # ReAct 数据按 task 去重
        return hashlib.md5(item.get("task", "").encode()).hexdigest()
    else:
        # 对话数据按首条 user 消息去重
        messages = item.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                return hashlib.md5(msg.get("content", "").encode()).hexdigest()
        return hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()


def _is_template(item: dict, is_react: bool) -> bool:
    """检查是否是模板填充"""
    if is_react:
        # ReAct 数据检查 final_answer
        for step in item.get("steps", []):
            answer = step.get("final_answer", "")
            if _is_template_text(answer):
                return True
        return False
    else:
        # 对话数据检查 assistant 回答
        for msg in item.get("messages", []):
            if msg.get("role") == "assistant":
                if _is_template_text(msg.get("content", "")):
                    return True
        return False


def _is_template_text(text: str) -> bool:
    """检查文本是否是模板填充"""
    if not text:
        return True

    text_stripped = text.strip()

    # 检查模板模式
    for pattern in TEMPLATE_PATTERNS:
        if pattern in text_stripped:
            # 如果文本很短且包含模板模式，认为是模板
            if len(text_stripped) < 200:
                return True

    # 检查 "..." 占位符密度
    dot_count = text_stripped.count("...")
    if dot_count >= 3 and len(text_stripped) < 300:
        return True

    return False


def _align_tool_names(item: dict, is_react: bool) -> Tuple[dict, bool]:
    """对齐工具名"""
    remapped = False

    if not is_react:
        return item, remapped

    new_item = dict(item)
    new_steps = []

    for step in item.get("steps", []):
        new_step = dict(step)
        action = step.get("action", "")

        if action in TOOL_NAME_MAPPING:
            new_step["action"] = TOOL_NAME_MAPPING[action]
            remapped = True

            # 如果有 action_args，也需要调整
            if action == "create_directory" and "action_args" in new_step:
                # create_directory 映射为 write_file，需要调整参数格式
                args = new_step["action_args"]
                if "input" in args:
                    dir_path = args["input"]
                    new_step["action_args"] = {"input": dir_path + "/.gitkeep | # Directory placeholder"}

        new_steps.append(new_step)

    new_item["steps"] = new_steps
    return new_item, remapped


def _score_quality(item: dict, is_react: bool) -> float:
    """质量评分（0-1）"""
    score = 0.5  # 基线

    if is_react:
        task = item.get("task", "")
        steps = item.get("steps", [])

        # 任务描述长度
        if len(task) > 10:
            score += 0.1
        if len(task) > 30:
            score += 0.1

        # 步骤数合理性
        if 1 <= len(steps) <= 10:
            score += 0.1

        # 有思考过程
        has_thought = any(s.get("thought") for s in steps)
        if has_thought:
            score += 0.1

        # 有工具调用或最终回答
        has_action = any(s.get("action") for s in steps)
        has_answer = any(s.get("final_answer") for s in steps)
        if has_action or has_answer:
            score += 0.1

    else:
        messages = item.get("messages", [])
        if len(messages) >= 2:
            score += 0.1

        # assistant 回答长度
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if len(content) > 50:
                    score += 0.1
                if len(content) > 200:
                    score += 0.1
                # 有结构化内容（列表、代码等）
                if any(marker in content for marker in ["```", "1.", "- ", "**"]):
                    score += 0.1

    return min(score, 1.0)


def _save_report(stats: dict, output_dir: str):
    """保存清洗报告"""
    report_path = os.path.join(output_dir, "cleaning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 人类可读报告
    text_path = os.path.join(output_dir, "cleaning_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("态极训练数据清洗报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"处理文件数: {stats['files_processed']}\n")
        f.write(f"输入样本数: {stats['total_input']}\n")
        f.write(f"输出样本数: {stats['total_output']}\n")
        f.write(f"去重数量: {stats['duplicates_removed']}\n")
        f.write(f"模板过滤: {stats['templates_removed']}\n")
        f.write(f"低质量过滤: {stats['low_quality_removed']}\n")
        f.write(f"工具名重映射: {stats['tools_remapped']}\n")
        f.write(f"\n保留率: {stats['total_output']/max(stats['total_input'],1)*100:.1f}%\n")
        f.write("\n各文件统计:\n")
        for fname, fs in stats.get("per_file", {}).items():
            f.write(f"  {fname}: {fs['input']} -> {fs['output']}\n")


def merge_cleaned_data(cleaned_dir: str, output_path: str):
    """合并清洗后的数据为单个文件"""
    react_data = []
    conv_data = []

    for fname in sorted(os.listdir(cleaned_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(cleaned_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "task" in item and "steps" in item:
                        react_data.append(item)
                    elif "messages" in item:
                        conv_data.append(item)
                except json.JSONDecodeError:
                    pass

    # 分别保存
    react_path = os.path.join(output_path, "all_react.jsonl")
    conv_path = os.path.join(output_path, "all_conversation.jsonl")

    with open(react_path, "w", encoding="utf-8") as f:
        for item in react_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(conv_path, "w", encoding="utf-8") as f:
        for item in conv_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"合并完成: {len(react_data)} ReAct + {len(conv_data)} Conversation")
    return len(react_data), len(conv_data)
