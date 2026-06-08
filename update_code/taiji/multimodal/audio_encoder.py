"""
态极 (Taiji) 音频编码器
耳朵 — 集成 Whisper，让态极能听懂语音

将音频编码为态极的隐藏空间向量，实现语音识别和音频理解。
"""
import os
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger("Taiji.Audio")


class TaijiAudioEncoder:
    """
    音频编码器 — 语音识别 + 音频理解

    集成 Whisper 语音识别模型。
    支持:
    - 语音转文字 (ASR)
    - 音频特征编码
    - 多语言识别
    """

    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"]

    def __init__(self, model_name: str = "openai/whisper-tiny",
                 hidden_size: int = 768, device: str = "cpu",
                 language: str = "zh"):
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.device = device
        self.language = language
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_failed = False

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._loaded:
            return
        if self._load_failed:
            return  # 依赖不可用，不重试
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            logger.info(f"Loading audio encoder: {self.model_name}")
            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Audio encoder loaded: {self.model_name}")
        except ImportError:
            logger.warning("transformers/Whisper not available, audio will use fallback")
            self._load_failed = True  # 依赖不可用，不再重试
        except Exception as e:
            logger.error(f"Failed to load audio encoder: {e}")
            # 运行时错误允许重试

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        语音转文字

        Args:
            audio_path: 音频文件路径
            language: 语言代码 (zh, en, ja 等)，None=自动检测

        Returns:
            识别出的文字
        """
        self._ensure_loaded()
        if self._model is None:
            return f"[语音识别不可用: {os.path.basename(audio_path)}]"

        try:
            import torch
            import librosa

            lang = language or self.language
            audio, sr = librosa.load(audio_path, sr=16000)

            input_features = self._processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)

            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=lang, task="transcribe"
            )

            with torch.no_grad():
                predicted_ids = self._model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                )

            transcription = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            return transcription.strip()

        except ImportError:
            logger.warning("librosa not installed, cannot load audio")
            return f"[需要安装 librosa: pip install librosa]"
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"[语音识别失败: {e}]"

    def get_audio_info(self, audio_path: str) -> dict:
        """获取音频文件信息"""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            return {
                "path": audio_path,
                "filename": os.path.basename(audio_path),
                "duration_sec": round(duration, 2),
                "sample_rate": sr,
                "samples": len(audio),
                "size_kb": round(os.path.getsize(audio_path) / 1024, 1),
            }
        except ImportError:
            return {
                "path": audio_path,
                "filename": os.path.basename(audio_path),
                "size_kb": round(os.path.getsize(audio_path) / 1024, 1),
            }
        except Exception as e:
            return {"path": audio_path, "error": str(e)}

    def audio_to_token_text(self, audio_path: str, transcription: str = "") -> str:
        """将音频编码为态极可理解的 token 文本"""
        if not transcription:
            transcription = f"[音频: {os.path.basename(audio_path)}]"
        return f"<speech>{transcription}</speech>"

    def get_supported_formats(self) -> List[str]:
        return self.SUPPORTED_FORMATS