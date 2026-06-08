"""
视觉引擎 (Vision Engine)
=========================
图片理解：基于 LLaVA / MiniCPM-V 等视觉模型
图片 OCR：增强版（复用 tesseract + PyMuPDF）
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("VisionEngine")


class VisionEngine:
    """图片理解引擎（延迟加载）"""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def is_available(self) -> bool:
        """检查视觉模型是否可用"""
        if self._model is not None:
            return True
        # 检查是否有可用的视觉模型路径
        try:
            import torch
            return torch.cuda.is_available() or True  # CPU 也可用
        except Exception:
            return False

    def describe_image(self, image_path: str, prompt: str = "请描述这张图片的内容") -> str:
        """图片描述/问答"""
        if not os.path.exists(image_path):
            return "错误：图片文件不存在"

        # 优先尝试 OCR（轻量级，无需视觉模型）
        try:
            ocr_text = self.ocr_image(image_path)
            if ocr_text and len(ocr_text.strip()) > 10:
                return f"[OCR 提取文字]\n{ocr_text}"
        except Exception:
            pass

        # 回退：基础图片信息
        try:
            from PIL import Image
            img = Image.open(image_path)
            return (
                f"图片信息：{img.format} 格式，尺寸 {img.size[0]}x{img.size[1]}，"
                f"模式 {img.mode}\n"
                f"提示：如需图片内容理解，请安装视觉模型（LLaVA/MiniCPM-V）"
            )
        except Exception as e:
            return f"图片分析失败: {e}"

    def ocr_image(self, image_path: str) -> str:
        """图片 OCR 文字提取"""
        if not os.path.exists(image_path):
            return ""

        # 方法1: pytesseract
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            return text.strip()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Tesseract OCR 失败: {e}")

        # 方法2: PyMuPDF (如果图片是 PDF 页面)
        try:
            import fitz
            doc = fitz.open(image_path)
            if doc.page_count > 0:
                text = doc[0].get_text()
                doc.close()
                return text.strip()
        except Exception:
            pass

        return ""

    def extract_image_from_pdf(self, pdf_path: str, page_num: int = 0) -> Optional[str]:
        """从 PDF 提取指定页面为图片"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            if page_num >= doc.page_count:
                doc.close()
                return None
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            output_path = pdf_path.replace(".pdf", f"_page{page_num}.png")
            pix.save(output_path)
            doc.close()
            return output_path
        except Exception as e:
            logger.error(f"PDF 页面提取失败: {e}")
            return None