"""
训练配置推荐 API
================
提供硬件自适应训练配置推荐和数据集质量检查端点
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("ApiServer.Training.Recommend")
router = APIRouter()


class RecommendRequest(BaseModel):
    model_path: str = ""
    dataset_size: int = 100
    preset: str = "mid"


class DatasetCheckRequest(BaseModel):
    file_path: str = ""


@router.post("/api/training/recommend")
async def get_training_recommendation(req: RecommendRequest):
    """获取硬件自适应训练推荐配置"""
    try:
        from model.training_recommender import TrainingRecommender
        rec = TrainingRecommender()
        if not req.model_path:
            hw = rec.detect_hardware()
            presets = {k: {**v, "hardware": hw} for k, v in rec.PRESETS.items()}
            return {"status": "success", "hardware": hw, "presets": presets}
        result = rec.recommend(req.model_path, req.dataset_size, req.preset)
        all_presets = rec.get_all_presets(req.model_path, req.dataset_size)
        return {"status": "success", "recommendation": result, "all_presets": all_presets}
    except Exception as e:
        logger.error(f"获取训练推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/check_dataset")
async def check_dataset_quality(req: DatasetCheckRequest):
    """检查数据集质量"""
    try:
        from model.dataset_checker import DatasetQualityChecker
        checker = DatasetQualityChecker()
        result = checker.check(req.file_path)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"数据集检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))