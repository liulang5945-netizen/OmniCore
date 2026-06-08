"""
态极 (Taiji) 视觉编码器
眼睛 — 集成 CLIP/SigLIP，让态极能看懂图像

将图像编码为态极的隐藏空间向量，实现图像理解。
"""
import os
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger("Taiji.Vision")


class TaijiVisionEncoder:
    """
    视觉编码器 — 图像理解

    集成 CLIP 视觉编码器，将图像映射到态极的隐藏空间。
    支持:
    - 单图理解
    - 多图对比
    - 图文匹配
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 hidden_size: int = 768, device: str = "cpu"):
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.device = device
        self._model = None
        self._processor = None
        self._projection = None
        self._loaded = False
        self._load_failed = False

    def _ensure_loaded(self):
        """延迟加载模型（节省内存）"""
        if self._loaded:
            return
        if self._load_failed:
            return  # 加载失败过，不重试（依赖不可用）
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading vision encoder: {self.model_name}")
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

            # 投影层: CLIP 维度 → 态极隐藏维度
            clip_dim = self._model.config.projection_dim  # 512 for base
            self._projection = nn.Linear(clip_dim, self.hidden_size)
            nn.init.normal_(self._projection.weight, std=0.02)

            self._model = self._model.to(self.device)
            self._projection = self._projection.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Vision encoder loaded: CLIP {clip_dim}D → Taiji {self.hidden_size}D")
        except ImportError:
            logger.warning("transformers/CLIP not available, vision will use fallback")
            self._load_failed = True  # 依赖不可用，不再重试
        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            # 运行时错误（如网络问题）允许重试，不设置任何标志

    def encode_image(self, image_path: str):
        """
        编码单张图像

        Args:
            image_path: 图像文件路径

        Returns:
            torch.Tensor [hidden_size] — 图像的隐藏向量表示
        """
        import torch
        self._ensure_loaded()

        if self._model is None:
            return torch.zeros(self.hidden_size)

        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                # 投影到态极隐藏空间
                projected = self._projection(image_features)
                return projected.squeeze(0)

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return torch.zeros(self.hidden_size)

    def encode_image_text_similarity(self, image_path: str, text: str) -> float:
        """
        计算图像和文本的相似度

        Returns:
            float: 0-1 的相似度分数
        """
        self._ensure_loaded()
        if self._model is None:
            return 0.0

        try:
            from PIL import Image
            import torch
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                similarity = outputs.logits_per_text.item()
                return float(similarity)

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def encode_multiple_images(self, image_paths: List[str]):
        """
        编码多张图像

        Returns:
            torch.Tensor [N, hidden_size]
        """
        import torch
        self._ensure_loaded()

        if self._model is None:
            return torch.zeros(len(image_paths), self.hidden_size)

        try:
            from PIL import Image
            images = []
            for path in image_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    continue

            if not images:
                return torch.zeros(len(image_paths), self.hidden_size)

            inputs = self._processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
                projected = self._projection(features)
                return projected

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return torch.zeros(len(image_paths), self.hidden_size)

    def image_to_token_text(self, image_path: str, description: str = "") -> str:
        """
        将图像编码为态极可理解的 token 文本

        输出: <image>图像描述</image>
        """
        if not description:
            description = f"[图像: {os.path.basename(image_path)}]"
        return f"<image>{description}</image>"

    def describe_image_simple(self, image_path: str) -> str:
        """
        简单图像描述（基于文件信息，不依赖模型）

        Returns:
            图像的基本描述文本
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            fmt = img.format or "unknown"
            mode = img.mode
            size_kb = os.path.getsize(image_path) / 1024
            return f"图像: {os.path.basename(image_path)}, {w}x{h}, {fmt}, {mode}, {size_kb:.0f}KB"
        except Exception as e:
            return f"图像: {os.path.basename(image_path)} (无法读取: {e})"

    def get_supported_formats(self) -> List[str]:
        """返回支持的图像格式"""
        return [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"]