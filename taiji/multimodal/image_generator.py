"""
态极 (Taiji) 图像生成模块
双手 — 让态极能创造图像

支持云端 API 和本地图像序列两种后端。
"""
import os
import logging
import json
import time
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger("Taiji.ImageGen")


class TaijiImageGenerator:
    """
    图像生成器 — 多后端支持

    后端优先级:
    1. 云端 API (效果最好，需网络)
    2. 本地图像序列 (CPU 可用，简单动画)
    """

    def __init__(self, output_dir: str = "./agent_workspace/generated_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, prompt: str, size: str = "512x512",
                 backend: str = "auto") -> Dict:
        """
        生成图像

        Args:
            prompt: 文字描述
            size: 图像尺寸 (如 "512x512", "1024x1024")
            backend: "cloud", "sequence", "auto"

        Returns:
            {"success": bool, "path": str, "backend": str, "error": str}
        """
        if backend in ("auto", "cloud"):
            result = self._generate_cloud(prompt, size)
            if result["success"]:
                return result

        # 回退到本地图像序列
        if backend in ("auto", "sequence"):
            return self._generate_sequence(prompt)

        return {"success": False, "path": "", "backend": "none",
                "error": f"后端 {backend} 不可用"}

    def _generate_cloud(self, prompt: str, size: str) -> Dict:
        """云端 API 生成"""
        try:
            settings_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "app_settings.json")
            if not os.path.exists(settings_path):
                return {"success": False, "path": "", "backend": "cloud",
                        "error": "未找到设置文件"}

            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            api_key = settings.get("cloud_api_key", "")
            api_base = settings.get("cloud_api_base", "")
            if not api_key or not api_base:
                return {"success": False, "path": "", "backend": "cloud",
                        "error": "未配置云端 API"}

            # 使用 OpenAI 兼容的图像生成 API
            import requests
            url = api_base.rstrip("/") + "/images/generations"
            w, h = size.split("x") if "x" in size else (512, 512)
            payload = {
                "prompt": prompt,
                "n": 1,
                "size": f"{w}x{h}",
                "response_format": "url",
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                image_url = data.get("data", [{}])[0].get("url", "")
                if image_url:
                    # 下载图像
                    img_resp = requests.get(image_url, timeout=30)
                    filename = f"img_{int(time.time())}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_resp.content)
                    return {"success": True, "path": filepath, "backend": "cloud",
                            "url": image_url}
            return {"success": False, "path": "", "backend": "cloud",
                    "error": f"API 返回 {resp.status_code}"}
        except Exception as e:
            return {"success": False, "path": "", "backend": "cloud",
                    "error": str(e)}

    def _generate_sequence(self, prompt: str) -> Dict:
        """本地图像序列生成（简单文字图片）"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(img)
            # 简单文字渲染
            text = prompt[:50] + ("..." if len(prompt) > 50 else "")
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
            draw.text((20, 20), f"态极生成:\n{text}", fill="black", font=font)
            draw.rectangle([10, 10, 502, 502], outline="gray", width=2)

            filename = f"img_seq_{int(time.time())}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            return {"success": True, "path": filepath, "backend": "sequence"}
        except Exception as e:
            return {"success": False, "path": "", "backend": "sequence",
                    "error": str(e)}

    def generate_gif(self, frames: list, fps: int = 4) -> Dict:
        """将图像列表保存为 GIF"""
        try:
            from PIL import Image
            imgs = [Image.open(f).convert("RGB") for f in frames if os.path.exists(f)]
            if not imgs:
                return {"success": False, "path": "", "error": "无有效帧"}
            filename = f"anim_{int(time.time())}.gif"
            filepath = os.path.join(self.output_dir, filename)
            duration = int(1000 / fps)
            imgs[0].save(filepath, save_all=True, append_images=imgs[1:],
                         duration=duration, loop=0)
            return {"success": True, "path": filepath, "frames": len(imgs)}
        except Exception as e:
            return {"success": False, "path": "", "error": str(e)}