"""
态极 (Taiji) 多模态配置
定义多模态 Token 空间和模态类型
"""
from taiji.config import SPECIAL_TOKENS


# 多模态特殊 Token 扩展 (接在工具名范围 32150-32899 之后，避免 ID 冲突)
MULTIMODAL_TOKENS = {
    # === 图像模态 (32900-32909) ===
    "image_start":      32900,  # <image>
    "image_end":        32901,  # </image>
    "img_gen":          32902,  # <img_gen> — 图像生成请求
    "img_edit":         32903,  # <img_edit> — 图像编辑请求
    "img_result":       32904,  # <img_result> — 图像生成结果

    # === 音频模态 (32910-32919) ===
    "audio_start":      32910,  # <audio>
    "audio_end":        32911,  # </audio>
    "speech":           32912,  # <speech> — 语音识别结果
    "tts_request":      32913,  # <tts> — 语音合成请求
    "tts_result":       32914,  # <tts_result> — 语音合成结果

    # === 视频模态 (32920-32929) ===
    "video_start":      32920,  # <video>
    "video_end":        32921,  # </video>
    "video_gen":        32922,  # <video_gen> — 视频生成请求
    "frames":           32923,  # <frames> — 视频帧序列
}


# 合并到主 SPECIAL_TOKENS
SPECIAL_TOKENS.update(MULTIMODAL_TOKENS)


# 模态类型枚举
class Modality:
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# 多模态工具注册
MULTIMODAL_TOOLS = {
    # 图像
    "generate_image": {
        "description": "根据文字描述生成图像",
        "input_format": '{"prompt": "描述", "size": "512x512"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
    "edit_image": {
        "description": "编辑现有图像",
        "input_format": '{"image_path": "图片路径", "instruction": "指令"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
    "describe_image": {
        "description": "描述图像内容",
        "input_format": '{"image_path": "图片路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    # 音频
    "transcribe_audio": {
        "description": "将音频转为文字（语音识别）",
        "input_format": '{"audio_path": "音频路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    "text_to_speech": {
        "description": "将文字转为语音（语音合成）",
        "input_format": '{"text": "文本", "voice": "声音"}',
        "modality": Modality.AUDIO,
        "output": "audio_path",
    },
    # 视频
    "understand_video": {
        "description": "理解视频内容并描述",
        "input_format": '{"video_path": "视频路径"}',
        "modality": Modality.TEXT,
        "output": "text",
    },
    "generate_video": {
        "description": "根据文字描述生成视频",
        "input_format": '{"prompt": "描述", "duration": "时长"}',
        "modality": Modality.VIDEO,
        "output": "video_path",
    },
    "video_to_gif": {
        "description": "将视频转换为GIF动图",
        "input_format": '{"video_path": "视频路径", "fps": "帧率"}',
        "modality": Modality.IMAGE,
        "output": "image_path",
    },
}


# 态极版本信息
TAIJI_VERSION = "4.0.0"
TAIJI_NAME = "态极"
TAIJI_NAME_EN = "Taiji"
TAIJI_DESCRIPTION = "OmniCore 原生多模态 Agent 模型 — 天生完整的智能体"