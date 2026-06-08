"""
态极语音交互能力 (Voice Interface)
====================================

态极的新能力 #8：听和说。

支持：
- 语音识别（Speech-to-Text）
- 语音合成（Text-to-Speech）
- 实时语音对话

依赖：
- speech_recognition（语音识别，可选）
- pyttsx3 或 edge-tts（语音合成，可选）
"""
import os
import logging
import tempfile
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("VoiceInterface")


@dataclass
class VoiceConfig:
    """语音配置"""
    language: str = "zh-CN"         # 语言
    rate: int = 150                  # 语速（字/分钟）
    volume: float = 0.8              # 音量 0-1
    voice_id: Optional[str] = None  # 语音 ID
    auto_listen: bool = False        # 自动监听


class VoiceInterface:
    """
    态极的语音交互引擎
    
    让态极能听懂用户说话，并用语音回答。
    
    工作流程：
    1. 用户按下录音按钮（或自动检测语音活动）
    2. 录音结束后，发送给语音识别引擎
    3. 识别结果作为文字输入传递给态极
    4. 态极生成文字回答
    5. 文字回答发送给语音合成引擎
    6. 播放语音回答
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._stt_available = False
        self._tts_available = False
        self._recorder = None
        
        # 检查 STT 可用性
        try:
            import speech_recognition
            self._stt_available = True
            logger.info("VoiceInterface: STT available (speech_recognition)")
        except ImportError:
            logger.info("VoiceInterface: STT not available")
        
        # 检查 TTS 可用性
        try:
            import pyttsx3
            self._tts_available = True
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.config.rate)
            self._engine.setProperty('volume', self.config.volume)
            logger.info("VoiceInterface: TTS available (pyttsx3)")
        except Exception:
            logger.info("VoiceInterface: TTS not available")
    
    # ─── 语音识别（STT）────────────────────────────
    
    def listen(self, timeout: int = 5, phrase_limit: int = 30) -> Optional[str]:
        """
        监听麦克风并识别语音。
        
        Args:
            timeout: 等待说话的超时秒数
            phrase_limit: 最长语音秒数
            
        Returns:
            识别出的文字，失败返回 None
        """
        if not self._stt_available:
            logger.warning("STT not available")
            return None
        
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                logger.info("Listening...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            
            text = recognizer.recognize_google(audio, language=self.config.language)
            logger.info(f"Recognized: {text}")
            return text
            
        except Exception as e:
            logger.warning(f"Speech recognition failed: {e}")
            return None
    
    def recognize_file(self, audio_path: str) -> Optional[str]:
        """
        识别音频文件中的语音。
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别出的文字
        """
        if not self._stt_available:
            return None
        
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio, language=self.config.language)
            return text
            
        except Exception as e:
            logger.warning(f"Audio file recognition failed: {e}")
            return None
    
    # ─── 语音合成（TTS）────────────────────────────
    
    def speak(self, text: str) -> bool:
        """
        将文字转换为语音并播放。
        
        Args:
            text: 要朗读的文字
            
        Returns:
            是否成功
        """
        if not self._tts_available:
            logger.warning("TTS not available")
            return False
        
        try:
            # 清理文字中的特殊字符
            clean_text = self._clean_for_speech(text)
            self._engine.say(clean_text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
            return False
    
    def save_speech(self, text: str, output_path: str) -> bool:
        """
        将文字转语音并保存到文件。
        
        Args:
            text: 要转换的文字
            output_path: 输出音频文件路径
            
        Returns:
            是否成功
        """
        if not self._tts_available:
            return False
        
        try:
            clean_text = self._clean_for_speech(text)
            self._engine.save_to_file(clean_text, output_path)
            self._engine.runAndWait()
            return True
        except Exception as e:
            logger.warning(f"Save speech failed: {e}")
            return False
    
    # ─── 语音对话 ──────────────────────────────────
    
    def voice_conversation(
        self,
        process_func: Callable[[str], str],
        max_turns: int = 10,
    ):
        """
        进行语音对话。
        
        Args:
            process_func: 处理用户输入的函数，接收文字返回回答
            max_turns: 最大对话轮数
        """
        self.speak("你好，我是态极。请说话。")
        
        for turn in range(max_turns):
            user_text = self.listen(timeout=10)
            
            if user_text is None:
                self.speak("我没有听清楚，请再说一次。")
                continue
            
            if any(word in user_text for word in ["再见", "结束", "退出"]):
                self.speak("好的，再见！")
                break
            
            response = process_func(user_text)
            self.speak(response)
    
    # ─── 工具方法 ──────────────────────────────────
    
    def _clean_for_speech(self, text: str) -> str:
        """清理文字，使其更适合语音朗读"""
        import re
        
        # 移除 Markdown 格式
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 粗体
        text = re.sub(r'\*(.*?)\*', r'\1', text)       # 斜体
        text = re.sub(r'`(.*?)`', r'\1', text)         # 代码
        text = re.sub(r'#{1,6}\s+', '', text)          # 标题
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 链接
        
        # 移除多余空白
        text = re.sub(r'\n+', '。', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def get_status(self) -> dict:
        """获取语音接口状态"""
        return {
            "stt_available": self._stt_available,
            "tts_available": self._tts_available,
            "language": self.config.language,
            "rate": self.config.rate,
        }
    
    def get_summary(self) -> str:
        """获取状态摘要"""
        status = self.get_status()
        stt = "✅ 可用" if status["stt_available"] else "❌ 不可用"
        tts = "✅ 可用" if status["tts_available"] else "❌ 不可用"
        
        return (
            f"🎵 语音接口状态\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"语音识别 (STT): {stt}\n"
            f"语音合成 (TTS): {tts}\n"
            f"语言: {status['language']}\n"
            f"语速: {status['rate']} 字/分钟"
        )


# ─── 全局实例 ─────────────────────────────────────

_global_voice: Optional[VoiceInterface] = None


def get_voice_interface(config: Optional[VoiceConfig] = None) -> VoiceInterface:
    """获取全局语音接口实例"""
    global _global_voice
    if _global_voice is None:
        _global_voice = VoiceInterface(config)
    return _global_voice