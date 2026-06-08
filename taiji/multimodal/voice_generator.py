"""
态极 (Taiji) 语音生成模块
嘴巴 — 让态极能说话

支持 edge-tts (云端，质量好) 和 pyttsx3 (离线) 两种后端。
"""
import os
import logging
import time
import asyncio
from typing import Optional, Dict, List

logger = logging.getLogger("Taiji.Voice")


class TaijiVoiceGenerator:
    """
    语音合成器 — TTS
    
    后端:
    1. edge-tts: 微软免费 TTS，质量好，需网络
    2. pyttsx3: 离线 TTS，质量一般，CPU 可用
    """
    
    CHINESE_VOICES = [
        "zh-CN-XiaoxiaoNeural",    # 女声，温柔
        "zh-CN-YunxiNeural",       # 男声，年轻
        "zh-CN-YunjianNeural",     # 男声，成熟
        "zh-CN-XiaoyiNeural",      # 女声，活泼
    ]
    
    def __init__(self, output_dir: str = "./agent_workspace/audio",
                 engine: str = "edge_tts"):
        self.output_dir = output_dir
        self.engine = engine
        os.makedirs(output_dir, exist_ok=True)
    
    def synthesize(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural",
                   rate: str = "+0%", volume: str = "+0%") -> Dict:
        """
        文字转语音
        
        Args:
            text: 要合成的文字
            voice: 语音名称
            rate: 语速 (如 "+20%", "-10%")
            volume: 音量
            
        Returns:
            {"success": bool, "path": str, "duration_estimate": float}
        """
        if self.engine == "edge_tts":
            return self._synthesize_edge_tts(text, voice, rate, volume)
        elif self.engine == "pyttsx3":
            return self._synthesize_pyttsx3(text)
        else:
            return {"success": False, "path": "", "error": f"未知引擎: {self.engine}"}
    
    def _synthesize_edge_tts(self, text: str, voice: str, rate: str, volume: str) -> Dict:
        """使用 edge-tts 合成"""
        try:
            import edge_tts
            
            filename = f"speech_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
            asyncio.run(communicate.save(filepath))
            
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                return {
                    "success": True,
                    "path": filepath,
                    "voice": voice,
                    "size_kb": round(size_kb, 1),
                    "engine": "edge_tts",
                }
            return {"success": False, "path": "", "error": "文件未生成"}
        except ImportError:
            logger.warning("edge-tts not installed, falling back to pyttsx3")
            return self._synthesize_pyttsx3(text)
        except Exception as e:
            logger.error(f"edge-tts error: {e}")
            return {"success": False, "path": "", "error": str(e)}
    
    def _synthesize_pyttsx3(self, text: str) -> Dict:
        """使用 pyttsx3 离线合成"""
        try:
            import pyttsx3
            
            filename = f"speech_{int(time.time())}.wav"
            filepath = os.path.join(self.output_dir, filename)
            
            engine = pyttsx3.init()
            engine.save_to_file(text, filepath)
            engine.runAndWait()
            
            if os.path.exists(filepath):
                return {"success": True, "path": filepath, "engine": "pyttsx3"}
            return {"success": False, "path": "", "error": "文件未生成"}
        except ImportError:
            return {"success": False, "path": "", "error": "需要安装 pyttsx3: pip install pyttsx3"}
        except Exception as e:
            return {"success": False, "path": "", "error": str(e)}
    
    def get_voices(self) -> List[str]:
        """获取可用语音列表"""
        return self.CHINESE_VOICES
    
    def is_available(self) -> bool:
        """检查 TTS 是否可用"""
        try:
            if self.engine == "edge_tts":
                import edge_tts
                return True
            elif self.engine == "pyttsx3":
                import pyttsx3
                return True
        except ImportError:
            pass
        return False