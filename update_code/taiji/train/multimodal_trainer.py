"""
态极多模态训练器
=================

训练多模态投影层，让态极能"看图说话"、"听声理解"。

训练流程：
  阶段一：冻结 backbone，只训练投影层
    - CLIP 特征 → 投影层 → 与文本拼接 → Transformer → 语言损失
    - 目标：投影层学会将 CLIP embedding 映射到态极的语义空间

  阶段二：解冻 backbone + 投影层，联合微调
    - 目标：整个模型学会处理多模态输入
"""
import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Generator, Tuple

logger = logging.getLogger("Taiji.MultimodalTrainer")


class MultimodalTrainer:
    """
    多模态训练器

    训练 CLIP/Whisper → 态极 的投影层。
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            model: 态极模型（ModelSelf）
            tokenizer: 分词器
            device: 设备
            learning_rate: 学习率
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = learning_rate

        # 延迟加载编码器
        self._vision_encoder = None
        self._audio_encoder = None

    def _get_vision_encoder(self):
        """获取 CLIP 视觉编码器"""
        if self._vision_encoder is None:
            try:
                from taiji.vision_encoder import TaijiVisionEncoder
                hidden_size = self.model.config.hidden_size
                self._vision_encoder = TaijiVisionEncoder(
                    hidden_size=hidden_size,
                    device=self.device,
                )
                logger.info("CLIP 视觉编码器已加载")
            except Exception as e:
                logger.warning(f"无法加载视觉编码器: {e}")
        return self._vision_encoder

    def _get_audio_encoder(self):
        """获取 Whisper 音频编码器"""
        if self._audio_encoder is None:
            try:
                from taiji.audio_encoder import TaijiAudioEncoder
                hidden_size = self.model.config.hidden_size
                self._audio_encoder = TaijiAudioEncoder(
                    hidden_size=hidden_size,
                    device=self.device,
                )
                logger.info("Whisper 音频编码器已加载")
            except Exception as e:
                logger.warning(f"无法加载音频编码器: {e}")
        return self._audio_encoder

    def train_projection(
        self,
        dataset_path: str,
        num_epochs: int = 10,
        batch_size: int = 4,
        save_dir: str = None,
        freeze_backbone: bool = True,
    ) -> Generator:
        """
        训练多模态投影层。

        Args:
            dataset_path: 多模态数据集路径（JSONL 格式）
            num_epochs: 训练轮数
            batch_size: 批次大小
            save_dir: 保存目录
            freeze_backbone: 是否冻结 backbone

        Yields:
            (epoch, step, loss, metrics)
        """
        # 加载数据
        data = self._load_dataset(dataset_path)
        if not data:
            logger.error("数据集为空")
            return

        logger.info(f"加载了 {len(data)} 条多模态训练数据")

        # 冻结策略
        if freeze_backbone:
            self._freeze_backbone()
            logger.info("Backbone 已冻结，只训练投影层")
        else:
            logger.info("联合训练：backbone + 投影层")

        # 优化器（只优化未冻结的参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        self.model.train()
        step = 0
        total_steps = len(data) * num_epochs

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_loss = 0.0

                for item in batch:
                    try:
                        loss = self._train_single(item, optimizer)
                        batch_loss += loss
                        step += 1
                    except Exception as e:
                        logger.warning(f"训练样本失败: {e}")
                        continue

                if batch_loss > 0:
                    batch_loss = batch_loss / len(batch)
                    epoch_loss += batch_loss
                    num_batches += 1

                    yield (
                        epoch,
                        step,
                        batch_loss,
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "total_steps": total_steps,
                            "loss": batch_loss,
                            "avg_loss": epoch_loss / max(num_batches, 1),
                        },
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Multimodal Epoch {epoch+1}/{num_epochs} done, avg loss: {avg_loss:.4f}")

        # 保存
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            from taiji.loader import save_model
            save_path = os.path.join(save_dir, "multimodal_best")
            save_model(self.model, self.tokenizer, save_path)
            logger.info(f"多模态模型已保存到 {save_path}")

    def _train_single(self, item: dict, optimizer) -> float:
        """训练单个样本"""
        modality = item.get("modality", "text")
        messages = item.get("messages", [])
        media_path = item.get("media_path", "")

        if not messages:
            return 0.0

        # 构建目标文本
        target_text = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                target_text = msg.get("content", "")
                break

        if not target_text:
            return 0.0

        # 构建输入 prompt
        prompt_parts = []
        for msg in messages:
            if msg.get("role") != "assistant":
                prompt_parts.append(f"[{msg['role']}] {msg['content']}")
        prompt = "\n".join(prompt_parts) + "\n[assistant] "

        # 获取多模态特征
        vision_features = None
        audio_features = None

        if modality == "vision" and media_path:
            encoder = self._get_vision_encoder()
            if encoder:
                vision_features = self._encode_image(encoder, media_path)

        elif modality == "audio" and media_path:
            encoder = self._get_audio_encoder()
            if encoder:
                audio_features = self._encode_audio(encoder, media_path)

        # Tokenize
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt")["input_ids"].to(self.device)

        # 拼接
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)

        # 构建 labels（只计算 target 部分的损失）
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_ids.shape[1]:] = target_ids

        # 前向传播
        outputs = self.model(
            input_ids,
            targets=labels,
            vision_features=vision_features,
            audio_features=audio_features,
        )

        loss = outputs.loss
        if loss is None:
            return 0.0

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    def _encode_image(self, encoder, image_path: str) -> torch.Tensor:
        """编码图像"""
        try:
            if os.path.exists(image_path):
                features = encoder.encode_image(image_path)
                return features.to(self.device)
        except Exception as e:
            logger.warning(f"图像编码失败: {e}")
        return None

    def _encode_audio(self, encoder, audio_path: str) -> torch.Tensor:
        """编码音频"""
        try:
            if os.path.exists(audio_path):
                features = encoder.encode_audio(audio_path)
                return features.to(self.device)
        except Exception as e:
            logger.warning(f"音频编码失败: {e}")
        return None

    def _freeze_backbone(self):
        """冻结 backbone，只训练投影层"""
        # 冻结 backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # 冻结所有头
        for head_name in ["lm_head", "tool_head", "perception_head", "memory_head", "plan_head"]:
            head = getattr(self.model, head_name, None)
            if head:
                for param in head.parameters():
                    param.requires_grad = False

        # 只训练投影层
        if hasattr(self.model, "vision_projector"):
            for param in self.model.vision_projector.parameters():
                param.requires_grad = True
        if hasattr(self.model, "audio_projector"):
            for param in self.model.audio_projector.parameters():
                param.requires_grad = True

    def _load_dataset(self, path: str) -> List[dict]:
        """加载多模态数据集"""
        data = []

        if os.path.isdir(path):
            # 目录：加载所有 JSONL 文件
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".jsonl"):
                    fpath = os.path.join(path, fname)
                    data.extend(self._load_jsonl(fpath))
        elif os.path.isfile(path):
            data = self._load_jsonl(path)

        return data

    def _load_jsonl(self, path: str) -> List[dict]:
        """加载单个 JSONL 文件"""
        data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        # 只保留有 modality 字段的样本
                        if "modality" in item or "messages" in item:
                            data.append(item)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"加载 {path} 失败: {e}")
        return data
