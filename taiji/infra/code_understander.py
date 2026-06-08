"""
态极代码理解能力 (Code Understander)
======================================

态极的新能力 #2：不只是读代码，而是理解代码。

用 Python AST 解析器分析代码结构：
- 识别函数、类、变量、导入
- 追踪跨文件引用
- 分析依赖关系
- 提取代码摘要

这样态极在修改代码前，能先理解代码的全貌。
"""
import ast
import os
import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("CodeUnderstander")


@dataclass
class FunctionInfo:
    """函数信息"""
    name: str
    args: List[str]
    line: int
    end_line: int
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_async: bool = False
    calls: List[str] = field(default_factory=list)  # 调用的其他函数


@dataclass
class ClassInfo:
    """类信息"""
    name: str
    bases: List[str] = field(default_factory=list)
    line: int = 0
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ImportInfo:
    """导入信息"""
    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    line: int = 0
    is_from: bool = False


@dataclass
class CodeAnalysis:
    """代码分析结果"""
    file_path: str
    language: str = "python"
    total_lines: int = 0
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    global_vars: List[Tuple[str, int]] = field(default_factory=list)
    complexity_score: float = 0.0  # 0-10, 简单到复杂
    summary: str = ""


class CodeUnderstander:
    """
    态极的代码理解引擎
    
    用 AST 解析代码结构，让态极"看懂"代码而不只是"读到"代码。
    
    支持：
    - Python AST 解析
    - JavaScript/TypeScript 正则解析（简化版）
    - 通用文本分析（其他语言的回退方案）
    """
    
    def analyze(self, code: str, file_path: str = "") -> CodeAnalysis:
        """
        分析代码内容。
        
        Args:
            code: 代码文本
            file_path: 文件路径（用于推断语言）
            
        Returns:
            CodeAnalysis 分析结果
        """
        lang = self._detect_language(file_path)
        
        if lang == "python":
            return self._analyze_python(code, file_path)
        elif lang in ("javascript", "typescript"):
            return self._analyze_js_regex(code, file_path)
        else:
            return self._analyze_generic(code, file_path)
    
    def analyze_file(self, file_path: str) -> Optional[CodeAnalysis]:
        """分析文件"""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.analyze(code, file_path)
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path: str) -> str:
        """根据文件扩展名推断语言"""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".vue": "vue",
            ".json": "json",
        }
        ext = os.path.splitext(file_path)[1].lower()
        return ext_map.get(ext, "unknown")
    
    # ─── Python AST 分析 ───────────────────────────
    
    def _analyze_python(self, code: str, file_path: str) -> CodeAnalysis:
        """用 AST 深度分析 Python 代码"""
        analysis = CodeAnalysis(file_path=file_path, language="python")
        analysis.total_lines = code.count("\n") + 1
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            analysis.summary = f"语法错误: {e}"
            return analysis
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                analysis.functions.append(self._extract_function(node))
            elif isinstance(node, ast.ClassDef):
                analysis.classes.append(self._extract_class(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                analysis.imports.extend(self._extract_import(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        analysis.global_vars.append((target.id, node.lineno))
        
        # 计算复杂度
        analysis.complexity_score = self._calc_complexity(tree)
        
        # 生成摘要
        analysis.summary = self._generate_summary(analysis)
        
        return analysis
    
    def _extract_function(self, node) -> FunctionInfo:
        """提取函数信息"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        docstring = ast.get_docstring(node)
        
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
        
        return FunctionInfo(
            name=node.name,
            args=args,
            line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            decorators=decorators,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            calls=list(set(calls)),
        )
    
    def _extract_class(self, node) -> ClassInfo:
        """提取类信息"""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        
        methods = []
        attributes = []
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(child))
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        return ClassInfo(
            name=node.name,
            bases=bases,
            line=node.lineno,
            methods=methods,
            attributes=attributes,
            docstring=ast.get_docstring(node),
        )
    
    def _extract_import(self, node) -> List[ImportInfo]:
        """提取导入信息"""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(
                    module=alias.name,
                    alias=alias.asname,
                    line=node.lineno,
                ))
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            imports.append(ImportInfo(
                module=node.module or "",
                names=names,
                line=node.lineno,
                is_from=True,
            ))
        
        return imports
    
    def _calc_complexity(self, tree) -> float:
        """计算代码复杂度（简化版圈复杂度）"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        # 归一化到 0-10
        return min(complexity / 5.0, 10.0)
    
    def _generate_summary(self, analysis: CodeAnalysis) -> str:
        """生成代码摘要"""
        parts = []
        
        if analysis.functions:
            parts.append(f"{len(analysis.functions)} 个函数")
        if analysis.classes:
            parts.append(f"{len(analysis.classes)} 个类")
        if analysis.imports:
            parts.append(f"{len(analysis.imports)} 个导入")
        
        summary = f"Python 文件 ({analysis.total_lines} 行): {', '.join(parts) if parts else '无结构'}"
        
        if analysis.complexity_score > 5:
            summary += f" [复杂度高: {analysis.complexity_score:.1f}/10]"
        
        return summary
    
    # ─── JavaScript 正则分析（简化版）──────────────
    
    def _analyze_js_regex(self, code: str, file_path: str) -> CodeAnalysis:
        """用正则表达式分析 JavaScript/TypeScript 代码（简化版）"""
        analysis = CodeAnalysis(
            file_path=file_path,
            language="javascript",
            total_lines=code.count("\n") + 1,
        )
        
        # 提取函数
        for match in re.finditer(r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))', code):
            line_num = code[:match.start()].count("\n") + 1
            analysis.functions.append(FunctionInfo(
                name=match.group(1),
                args=[],
                line=line_num,
                end_line=line_num,
            ))
        
        # 提取类
        for match in re.finditer(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', code):
            line_num = code[:match.start()].count("\n") + 1
            analysis.classes.append(ClassInfo(
                name=match.group(1),
                bases=[match.group(2)] if match.group(2) else [],
                line=line_num,
            ))
        
        # 提取导入
        for match in re.finditer(r'import\s+(?:{([^}]+)}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]', code):
            line_num = code[:match.start()].count("\n") + 1
            names = []
            if match.group(1):
                names = [n.strip() for n in match.group(1).split(",")]
            analysis.imports.append(ImportInfo(
                module=match.group(3),
                names=names,
                line=line_num,
                is_from=True,
            ))
        
        analysis.summary = f"JS/TS 文件 ({analysis.total_lines} 行): {len(analysis.functions)} 函数, {len(analysis.classes)} 类"
        return analysis
    
    # ─── 通用文本分析（回退方案）──────────────────
    
    def _analyze_generic(self, code: str, file_path: str) -> CodeAnalysis:
        """通用文本分析（无法用 AST 时的回退方案）"""
        analysis = CodeAnalysis(
            file_path=file_path,
            language="unknown",
            total_lines=code.count("\n") + 1,
        )
        
        # 统计基本结构
        analysis.summary = f"文本文件 ({analysis.total_lines} 行)"
        return analysis
    
    # ─── 依赖分析 ──────────────────────────────────
    
    def find_dependencies(self, code: str, file_path: str = "") -> List[str]:
        """提取代码的外部依赖"""
        analysis = self.analyze(code, file_path)
        deps = []
        for imp in analysis.imports:
            if imp.is_from:
                deps.append(imp.module)
            else:
                deps.append(imp.module)
        return list(set(deps))
    
    def find_references(self, code: str, symbol: str) -> List[Dict[str, Any]]:
        """查找代码中引用某个符号的所有位置"""
        results = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if symbol in line:
                results.append({
                    "line": i,
                    "content": line.strip(),
                    "context": "assignment" if "=" in line and symbol in line.split("=")[0] else "usage",
                })
        return results


# ─── 全局实例 ─────────────────────────────────────

_global_understander: Optional[CodeUnderstander] = None


def get_code_understander() -> CodeUnderstander:
    """获取全局代码理解器实例"""
    global _global_understander
    if _global_understander is None:
        _global_understander = CodeUnderstander()
    return _global_understander