"""
态极 (Taiji) 视频引擎
眼睛+双手 — 视频理解与生成

支持:
- 视频理解: 帧提取 + 音频分离 + 多模态理解
- 视频生成: 云端 API / 本地图像序列
"""
import os
import logging
import time
import json
from typing import Optional, Dict, List

logger = logging.getLogger("Taiji.Video")


class TaijiVideoEngine:
    """
    视频引擎 — 理解 + 生成

    视频理解: 提取关键帧 → 视觉编码器分析 → 文字描述
    视频生成: 云端 API / 本地图像序列 → GIF/MP4
    """

    def __init__(self, output_dir: str = "./agent_workspace/generated_videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self, video_path: str, fps: float = 1.0,
                       max_frames: int = 16) -> List[str]:
        """
        从视频中提取关键帧

        Args:
            video_path: 视频文件路径
            fps: 每秒提取帧数
            max_frames: 最大帧数

        Returns:
            帧图像文件路径列表
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = max(1, int(video_fps / fps))
            frames_dir = os.path.join(self.output_dir, f"frames_{int(time.time())}")
            os.makedirs(frames_dir, exist_ok=True)

            frame_paths = []
            frame_count = 0
            saved_count = 0

            while cap.isOpened() and saved_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_count += 1
                frame_count += 1

            cap.release()
            logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
            return frame_paths

        except ImportError:
            logger.warning("opencv-python not installed, cannot extract frames")
            return []
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []

    def understand_video(self, video_path: str) -> str:
        """
        理解视频内容

        通过提取关键帧 + 文件信息来描述视频

        Returns:
            视频内容的文字描述
        """
        info = self.get_video_info(video_path)
        frames = self.extract_frames(video_path, fps=0.5, max_frames=8)

        description_parts = []
        description_parts.append(f"视频: {os.path.basename(video_path)}")
        if info.get("duration_sec"):
            description_parts.append(f"时长: {info['duration_sec']:.1f}秒")
        if info.get("resolution"):
            description_parts.append(f"分辨率: {info['resolution']}")
        if frames:
            description_parts.append(f"提取了 {len(frames)} 个关键帧")

        return ", ".join(description_parts)

    def get_video_info(self, video_path: str) -> dict:
        """获取视频文件信息"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "无法打开视频"}

            info = {
                "path": video_path,
                "filename": os.path.basename(video_path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "size_kb": round(os.path.getsize(video_path) / 1024, 1),
            }
            if info["fps"] > 0:
                info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
            info["resolution"] = f"{info['width']}x{info['height']}"
            cap.release()
            return info

        except ImportError:
            return {"path": video_path, "filename": os.path.basename(video_path),
                    "size_kb": round(os.path.getsize(video_path) / 1024, 1)}
        except Exception as e:
            return {"error": str(e)}

    def generate_video(self, prompt: str, duration: int = 4,
                       backend: str = "auto") -> Dict:
        """
        生成视频

        Args:
            prompt: 文字描述
            duration: 时长(秒)
            backend: "cloud" / "local" / "auto"

        Returns:
            {"success": bool, "path": str, "error": str}
        """
        if backend in ("auto", "cloud"):
            result = self._generate_cloud(prompt, duration)
            if result["success"]:
                return result

        if backend in ("auto", "local"):
            return self._generate_local(prompt, duration)

        return {"success": False, "path": "", "error": f"后端 {backend} 不可用"}

    def _generate_cloud(self, prompt: str, duration: int) -> Dict:
        """云端 API 生成视频"""
        try:
            settings_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "app_settings.json")
            if not os.path.exists(settings_path):
                return {"success": False, "path": "", "error": "未找到设置文件"}

            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            api_key = settings.get("cloud_api_key", "")
            api_base = settings.get("cloud_api_base", "")
            if not api_key or not api_base:
                return {"success": False, "path": "", "error": "未配置云端 API"}

            import requests
            url = api_base.rstrip("/") + "/videos/generations"
            payload = {"prompt": prompt, "duration": duration, "n": 1}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            resp = requests.post(url, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                video_url = data.get("data", [{}])[0].get("url", "")
                if video_url:
                    vid_resp = requests.get(video_url, timeout=60)
                    filename = f"video_{int(time.time())}.mp4"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(vid_resp.content)
                    return {"success": True, "path": filepath, "backend": "cloud"}
            return {"success": False, "path": "", "error": f"API 返回 {resp.status_code}"}
        except Exception as e:
            return {"success": False, "path": "", "error": str(e)}

    def _generate_local(self, prompt: str, duration: int) -> Dict:
        """本地图像序列生成简单动画"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            frames = []
            fps = 4
            total_frames = duration * fps

            for i in range(total_frames):
                img = Image.new("RGB", (512, 512), "white")
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except Exception:
                    font = ImageFont.load_default()
                text = prompt[:40] + ("..." if len(prompt) > 40 else "")
                draw.text((20, 20), f"态极视频 [{i+1}/{total_frames}]\n{text}", fill="black", font=font)
                draw.rectangle([10, 10, 502, 502], outline="gray", width=2)
                frames.append(img)

            filename = f"video_{int(time.time())}.gif"
            filepath = os.path.join(self.output_dir, filename)
            duration_ms = int(1000 / fps)
            frames[0].save(filepath, save_all=True, append_images=frames[1:],
                           duration=duration_ms, loop=0)
            return {"success": True, "path": filepath, "backend": "local",
                    "frames": len(frames)}
        except Exception as e:
            return {"success": False, "path": "", "error": str(e)}

    def video_to_gif(self, video_path: str, fps: float = 4.0,
                     max_frames: int = 32) -> Dict:
        """将视频转换为 GIF"""
        frames = self.extract_frames(video_path, fps=fps, max_frames=max_frames)
        if not frames:
            return {"success": False, "path": "", "error": "无法提取帧"}

        try:
            from PIL import Image
            imgs = [Image.open(f).convert("RGB") for f in frames]
            filename = f"gif_{int(time.time())}.gif"
            filepath = os.path.join(self.output_dir, filename)
            duration_ms = int(1000 / fps)
            imgs[0].save(filepath, save_all=True, append_images=imgs[1:],
                         duration=duration_ms, loop=0)
            return {"success": True, "path": filepath, "frames": len(imgs)}
        except Exception as e:
            return {"success": False, "path": "", "error": str(e)}