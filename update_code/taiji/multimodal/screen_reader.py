"""
态极屏幕理解能力 (Screen Reader)
==================================

态极的新能力 #7：看懂屏幕。

支持：
- 截图文字提取（OCR）
- 错误信息识别
- UI 元素分析
- 代码截图解析
"""
import os
import re
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ScreenReader")


@dataclass
class ScreenAnalysis:
    """屏幕分析结果"""
    source: str                  # 图片路径或描述
    extracted_text: str = ""     # 提取的文字
    error_messages: List[str] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)
    ui_elements: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0      # 分析置信度


class ScreenReader:
    """
    态极的屏幕理解引擎
    
    从截屏中提取结构化信息：
    1. 文字内容（OCR）
    2. 错误信息
    3. 代码片段
    4. UI 元素
    5. URL 链接
    
    依赖：pytesseract（可选，回退到基础分析）
    """
    
    def __init__(self):
        self._ocr_available = False
        try:
            import pytesseract
            self._ocr_available = True
            logger.info("ScreenReader: pytesseract available")
        except ImportError:
            logger.info("ScreenReader: pytesseract not available, using text-only mode")
    
    def analyze_image(self, image_path: str) -> Optional[ScreenAnalysis]:
        """
        分析一张截屏图片。
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            ScreenAnalysis 分析结果
        """
        if not os.path.exists(image_path):
            return None
        
        analysis = ScreenAnalysis(source=image_path)
        
        # 尝试 OCR
        if self._ocr_available:
            try:
                import pytesseract
                from PIL import Image
                
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                analysis.extracted_text = text.strip()
                analysis.confidence = 0.7
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                analysis.extracted_text = "(OCR 失败，请手动描述截图内容)"
                analysis.confidence = 0.1
        else:
            analysis.extracted_text = "(OCR 不可用，请手动描述截图内容)"
            analysis.confidence = 0.0
        
        # 结构化提取
        if analysis.extracted_text:
            self._extract_structure(analysis)
        
        analysis.summary = self._generate_summary(analysis)
        return analysis
    
    def analyze_text_description(self, description: str) -> ScreenAnalysis:
        """
        分析用户对屏幕的文字描述。
        
        当无法直接处理图片时，用户可以描述截图内容。
        """
        analysis = ScreenAnalysis(source="text_description", extracted_text=description)
        analysis.confidence = 0.8  # 用户描述通常比较准确
        
        self._extract_structure(analysis)
        analysis.summary = self._generate_summary(analysis)
        return analysis
    
    def _extract_structure(self, analysis: ScreenAnalysis):
        """从文本中提取结构化信息"""
        text = analysis.extracted_text
        
        # 提取错误信息
        error_patterns = [
            r'(?:Error|Exception|Traceback|错误|异常)[:\s]*(.*?)(?:\n|$)',
            r'(?:FATAL|CRITICAL|ERROR)[:\s]*(.*?)(?:\n|$)',
            r'(?:raise|throw)\s+(.*?)(?:\n|$)',
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis.error_messages.extend(matches)
        
        # 提取代码片段
        code_patterns = [
            r'```[\w]*\n(.*?)```',
            r'(?:def |class |import |from |const |let |var |function ).*?(?:\n|$)',
        ]
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            analysis.code_snippets.extend(matches)
        
        # 提取 URL
        url_pattern = r'https?://[^\s<>"]+'
        analysis.urls = re.findall(url_pattern, text)
        
        # 提取 UI 元素（按钮、输入框等）
        ui_patterns = [
            r'(?:button|按钮|输入框|菜单|下拉|checkbox|radio|tab|标签)',
        ]
        for pattern in ui_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analysis.ui_elements.extend(matches)
    
    def _generate_summary(self, analysis: ScreenAnalysis) -> str:
        """生成分析摘要"""
        parts = []
        
        if analysis.error_messages:
            parts.append(f"发现 {len(analysis.error_messages)} 个错误信息")
        if analysis.code_snippets:
            parts.append(f"包含 {len(analysis.code_snippets)} 个代码片段")
        if analysis.urls:
            parts.append(f"包含 {len(analysis.urls)} 个 URL")
        
        if parts:
            return "屏幕分析: " + ", ".join(parts)
        elif analysis.extracted_text:
            return f"已提取 {len(analysis.extracted_text)} 字符的文字内容"
        else:
            return "未能从屏幕中提取有效信息"
    
    def suggest_fix_from_error(self, error_text: str) -> str:
        """根据错误信息建议修复方案"""
        error_lower = error_text.lower()
        
        suggestions = {
            "modulenotfounderror": "缺少依赖包，运行 `pip install <包名>` 安装",
            "importerror": "导入失败，检查模块名是否正确或是否已安装",
            "syntaxerror": "语法错误，检查括号、引号、缩进是否正确",
            "indentationerror": "缩进错误，检查是否混用了 Tab 和空格",
            "typeerror": "类型错误，检查函数参数类型是否匹配",
            "keyerror": "键不存在，使用 .get() 方法提供默认值",
            "filenotfounderror": "文件不存在，检查文件路径是否正确",
            "permissionerror": "权限不足，检查文件权限或以管理员运行",
            "connectionerror": "网络连接失败，检查网络和目标服务器",
            "timeouterror": "操作超时，检查网络或增加超时时间",
            "memoryerror": "内存不足，尝试减小数据量或使用更小的模型",
            "indexerror": "索引越界，检查列表/数组长度",
            "attributeerror": "属性不存在，检查对象类型和属性名",
            "valueerror": "值错误，检查输入参数的格式和范围",
            "nameerror": "名称未定义，检查变量名是否拼写正确或是否已导入",
        }
        
        for error_type, suggestion in suggestions.items():
            if error_type in error_lower:
                return suggestion
        
        return "建议查看完整的错误堆栈信息，定位出错的代码行"


# ─── 全局实例 ─────────────────────────────────────

_global_reader: Optional[ScreenReader] = None


def get_screen_reader() -> ScreenReader:
    """获取全局屏幕阅读器实例"""
    global _global_reader
    if _global_reader is None:
        _global_reader = ScreenReader()
    return _global_reader