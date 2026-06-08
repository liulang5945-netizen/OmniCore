"""
OmniCore 严苛评估 — 自动化全量分析脚本
执行：python tests/run_evaluation.py
"""
import ast
import importlib
import os
import sys
import subprocess
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULE_PATHS = [
    "agent/agent", "agent/agent_planner", "agent/agent_tools", "agent/run_exe", "agent/sandbox_executor",
    "api/api_server", "api/cli", "api/main", "api/run_app",
    "build_scripts/build_client", "build_scripts/build_installer", "build_scripts/hot_update", "build_scripts/runtime_hook_numpy", "build_scripts/splash", "build_scripts/updater",
    "core/app_state", "core/config", "core/utils",
    "model/data_loader", "model/gguf_engine", "model/model_registry", "model/model_setup", "model/trainer",
    "tools/bilibili_subtitle", "tools/file_parser", "tools/rag",
    "db_query", "long_term_memory", "url_to_rag"
]


def run_tests():
    """运行 pytest 并返回结果"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    return result.stdout + result.stderr


def count_lines(filepath):
    """统计代码行数（不含空行和注释）"""
    try:
        with open(filepath, "rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
            code_lines = [t for t in tokens if t.type == tokenize.ENCODING is False]
    except Exception:
        with open(filepath, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip() and not l.strip().startswith("#")]
            return len(lines)
    count = 0
    for tok in tokens:
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT):
            continue
        if tok.type == tokenize.STRING and tok.start[1] == 0:
            continue  # docstring
        count += 1
    return len(set(t.start[0] for t in tokens if t.type not in (
        tokenize.NEWLINE, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT
    ) and not (t.type == tokenize.STRING and t.start[1] == 0)))


def analyze_complexity(filepath):
    """简单复杂度检测（函数行数 > 50 告警）"""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or start
                length = end - start + 1
                if length > 50:
                    issues.append(f"    ⚠ {node.name}(): {length} 行 (超过 50 行阈值)")
        return issues
    except SyntaxError as e:
        return [f"    ❌ 语法错误: {e}"]


def analyze_exception_handling(filepath):
    """检测裸 except / except Exception 使用"""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(f"    ⚠ 行 {node.lineno}: 裸 except (应指定具体异常类型)")
        return issues
    except SyntaxError:
        return []


def detect_todos(filepath):
    """检测 TODO/FIXME/HACK 注释"""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        todos = []
        for i, line in enumerate(lines, 1):
            if any(marker in line for marker in ("TODO", "FIXME", "HACK", "XXX", "BUG")):
                todos.append(f"    📝 行 {i}: {line.strip()[:100]}")
        return todos
    except Exception:
        return []


def main():
    print("=" * 70)
    print("🔬 OmniCore 严苛评估 — 静态 + 动态分析")
    print("=" * 70)

    # 1. 代码行数统计
    print("\n📊 一、代码规模统计")
    total_lines = 0
    for mod_path in MODULE_PATHS:
        fpath = os.path.join(ROOT, f"{mod_path}.py")
        if os.path.exists(fpath):
            lines = count_lines(fpath)
            total_lines += lines
            print(f"  {mod_path}.py: ~{lines} 行")
    print(f"  总计: ~{total_lines} 行有效代码")

    # 2. 复杂度检测
    print("\n🐘 二、函数复杂度检测 (>50行)")
    complex_count = 0
    for mod_path in MODULE_PATHS:
        fpath = os.path.join(ROOT, f"{mod_path}.py")
        if os.path.exists(fpath):
            issues = analyze_complexity(fpath)
            if issues:
                print(f"  {mod_path}.py:")
                for issue in issues:
                    print(issue)
                    complex_count += 1
    if complex_count == 0:
        print("  ✅ 所有函数均 <= 50 行")

    # 3. 异常处理检测
    print("\n🛡️  三、异常处理安全检测")
    bare_count = 0
    for mod_path in MODULE_PATHS:
        fpath = os.path.join(ROOT, f"{mod_path}.py")
        if os.path.exists(fpath):
            issues = analyze_exception_handling(fpath)
            if issues:
                print(f"  {mod_path}.py:")
                for issue in issues:
                    print(issue)
                    bare_count += 1
    if bare_count == 0:
        print("  ✅ 无裸 except 语句")

    # 4. TODO 标记检测
    print("\n📝 四、待完成标记检测")
    todo_count = 0
    for mod_path in MODULE_PATHS:
        fpath = os.path.join(ROOT, f"{mod_path}.py")
        if os.path.exists(fpath):
            todos = detect_todos(fpath)
            if todos:
                print(f"  {mod_path}.py:")
                for t in todos:
                    print(t)
                    todo_count += 1
    if todo_count == 0:
        print("  ✅ 无待完成标记")

    # 5. 沙箱安全性验证
    print("\n🔒 五、安全沙箱验证")
    from agent.sandbox_executor import SAFE_MODULES, MAX_EXECUTION_TIME
    assert "os" not in SAFE_MODULES, "❌ os 不应在白名单中"
    assert "sys" not in SAFE_MODULES, "❌ sys 不应在白名单中"
    assert "subprocess" not in SAFE_MODULES, "❌ subprocess 不应在白名单中"
    assert "eval" not in str(SAFE_MODULES), "❌ eval 不应在安全代码中"
    print(f"  ✅ 白名单模块: {len(SAFE_MODULES)} 个")
    print(f"  ✅ 最大执行时间: {MAX_EXECUTION_TIME}s")
    print(f"  ✅ 危险模块成功隔离 (os, sys, subprocess 不在白名单)")

    # 6. 运行单元测试
    print("\n🧪 六、单元测试执行")
    try:
        output = run_tests()
        print(output[-500:])  # 最后 500 字符
    except Exception as e:
        print(f"  ❌ 测试执行失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 严苛评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()