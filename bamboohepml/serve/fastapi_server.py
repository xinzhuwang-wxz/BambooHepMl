"""
FastAPI 推理服务

提供 RESTful API 接口用于模型推理。
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import logger
from ..engine import Predictor
from ..metadata import load_model_metadata
from ..models import get_model


# 请求模型
class PredictRequest(BaseModel):
    """预测请求模型。"""

    features: list[list[float]] | None = Field(None, description="特征向量列表（向后兼容）")
    event: list[list[float]] | None = Field(None, description="Event-level 特征（新格式）")
    object: list[list[list[float]]] | None = Field(None, description="Object-level 特征（新格式）")
    mask: list[list[bool]] | None = Field(None, description="Object mask（新格式，仅在使用 object 时需要）")
    return_probabilities: bool = Field(False, description="是否返回概率")


class BatchPredictRequest(BaseModel):
    """批量预测请求模型。"""

    samples: list[dict[str, Any]] = Field(..., description="样本列表")
    return_probabilities: bool = Field(False, description="是否返回概率")


# 响应模型
class PredictResponse(BaseModel):
    """预测响应模型。"""

    predictions: list[Any] = Field(..., description="预测结果列表")
    probabilities: list[list[float]] | None = Field(None, description="概率列表（如果请求）")


class HealthResponse(BaseModel):
    """健康检查响应模型。"""

    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    model_info: dict[str, Any] | None = Field(None, description="模型信息")


def create_app(
    model_path: str | None = None,
    metadata_path: str | None = None,
    onnx_path: str | None = None,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    pipeline_config_path: str | None = None,  # 向后兼容，已废弃
) -> FastAPI:
    """
    创建 FastAPI 应用。

    Args:
        model_path: PyTorch 模型文件路径（.pt）
        metadata_path: 元数据文件路径（默认为 model_path 同目录下的 metadata.json）
        onnx_path: ONNX 模型文件路径（如果提供，将优先使用 ONNX 模型）
        model_name: 模型名称（如果 metadata 未提供，向后兼容用）
        model_params: 模型参数（如果 metadata 未提供，向后兼容用）
        pipeline_config_path: Pipeline 配置文件路径（已废弃，向后兼容用）

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title="BambooHepMl Inference Service",
        description="高能物理机器学习推理服务",
        version="0.1.0",
    )

    # 全局变量存储模型和预测器
    predictor: Predictor | None = None
    model_info: dict[str, Any] = {}

    @app.on_event("startup")
    async def startup_event():
        """启动时加载模型。"""
        nonlocal predictor, model_info
        try:
            from pathlib import Path

            # 优先使用 ONNX 模型
            if onnx_path and Path(onnx_path).exists():
                try:
                    from ..serve.onnx_predictor import ONNXPredictor

                    predictor = ONNXPredictor(onnx_path)
                    logger.info(f"ONNX model loaded from {onnx_path}")
                    model_info = {"model_type": "onnx", "onnx_path": onnx_path}
                    return
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model: {e}, falling back to PyTorch")

            # 加载 PyTorch 模型
            if not model_path:
                raise ValueError("model_path is required when onnx_path is not provided")

            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # 加载 metadata
            resolved_metadata_path = metadata_path
            if resolved_metadata_path is None:
                resolved_metadata_path = str(model_path_obj.parent / "metadata.json")
            else:
                resolved_metadata_path = str(Path(resolved_metadata_path))

            if Path(resolved_metadata_path).exists():
                # 从 metadata 加载
                metadata = load_model_metadata(resolved_metadata_path)
                model_config = metadata["model_config"]
                model_name = model_config.get("name")
                model_params = model_config.get("params", {}).copy()
                # model_params 已包含 event_input_dim/object_input_dim/embed_dim
            elif pipeline_config_path:
                # 从 pipeline 配置加载
                from ..pipeline import PipelineOrchestrator

                orchestrator = PipelineOrchestrator(pipeline_config_path)
                model_config = orchestrator.get_model_config()
                model_name = model_config.get("name")
                model_params = model_config.get("params", {})
                # setup_model 会自动从 FeatureGraph 推断维度，但这里我们只需要参数
            else:
                raise ValueError(
                    "Must provide metadata_path (or metadata.json in model directory), "
                    "or pipeline_config_path (deprecated), "
                    "or (model_name and model_params)"
                )

            # 创建模型
            model = get_model(model_name, **model_params)

            # 加载权重
            state_dict = torch.load(str(model_path), map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"PyTorch model loaded from {model_path}")

            # 创建预测器
            predictor = Predictor(model)
            model_info = {
                "model_type": "pytorch",
                "model_name": model_name,
                "model_params": model_params,
            }
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # 不抛出异常，允许应用启动（用于健康检查）
            predictor = None
            model_info = {"error": str(e)}

    @app.get("/")
    async def health_check() -> HealthResponse:
        """健康检查。"""
        if predictor is None:
            return HealthResponse(
                status="unavailable",
                message="Model not loaded",
                model_info=None,
            )
        return HealthResponse(
            status="healthy",
            message="Model is ready",
            model_info=model_info,
        )

    @app.post("/predict")
    async def predict(request: PredictRequest) -> PredictResponse:
        """单样本预测。"""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # 构建 batch 字典（支持新格式和向后兼容）
            batch_dict = {}

            # 优先使用新格式（event/object）
            if request.event is not None:
                batch_dict["event"] = torch.tensor(request.event, dtype=torch.float32)
            if request.object is not None:
                batch_dict["object"] = torch.tensor(request.object, dtype=torch.float32)
                if request.mask is not None:
                    batch_dict["mask"] = torch.tensor(request.mask, dtype=torch.bool)

            # 向后兼容：如果没有新格式，使用旧的 features
            if not batch_dict and request.features is not None:
                batch_dict["features"] = torch.tensor(request.features, dtype=torch.float32)

            if not batch_dict:
                raise ValueError("Must provide either (event/object) or features")

            # 创建数据集
            from torch.utils.data import DataLoader

            from ..utils import DictDataset

            dataset = DictDataset([batch_dict])
            batch_size = (
                len(request.event)
                if request.event is not None
                else len(request.object) if request.object is not None else len(request.features) if request.features is not None else 1
            )
            dataloader = DataLoader(dataset, batch_size=batch_size)

            # 预测
            results = predictor.predict(dataloader, return_probabilities=request.return_probabilities)

            predictions = [r["prediction"] for r in results]
            probabilities = None
            if request.return_probabilities and results and "probabilities" in results[0]:
                probabilities = [r["probabilities"] for r in results]

            return PredictResponse(predictions=predictions, probabilities=probabilities)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictRequest) -> PredictResponse:
        """批量预测。"""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # 收集所有样本（支持新格式和向后兼容）
            batch_dict = {}

            # 检查第一个样本来确定格式
            if not request.samples:
                raise ValueError("No samples provided")

            first_sample = request.samples[0]

            # 优先使用新格式
            if "event" in first_sample:
                all_event = [sample.get("event") for sample in request.samples]
                batch_dict["event"] = torch.tensor(all_event, dtype=torch.float32)
            if "object" in first_sample:
                all_object = [sample.get("object") for sample in request.samples]
                batch_dict["object"] = torch.tensor(all_object, dtype=torch.float32)
                if "mask" in first_sample:
                    all_mask = [sample.get("mask") for sample in request.samples]
                    batch_dict["mask"] = torch.tensor(all_mask, dtype=torch.bool)

            # 向后兼容：如果没有新格式，使用旧的 features
            if not batch_dict:
                if "features" not in first_sample:
                    raise ValueError("Each sample must have 'features', 'event', or 'object' key")
                all_features = [sample["features"] for sample in request.samples]
                batch_dict["features"] = torch.tensor(all_features, dtype=torch.float32)

            # 创建数据集
            from torch.utils.data import DataLoader

            from ..utils import DictDataset

            dataset = DictDataset([batch_dict])
            dataloader = DataLoader(dataset, batch_size=len(request.samples))

            # 预测
            results = predictor.predict(dataloader, return_probabilities=request.return_probabilities)

            predictions = [r["prediction"] for r in results]
            probabilities = None
            if request.return_probabilities and results and "probabilities" in results[0]:
                probabilities = [r["probabilities"] for r in results]

            return PredictResponse(predictions=predictions, probabilities=probabilities)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def serve_fastapi(
    model_path: str | None = None,
    metadata_path: str | None = None,
    onnx_path: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    pipeline_config_path: str | None = None,  # 向后兼容，已废弃
    **kwargs,
):
    """
    启动 FastAPI 服务。

    Args:
        model_path: PyTorch 模型文件路径（.pt）
        metadata_path: 元数据文件路径（默认为 model_path 同目录下的 metadata.json）
        onnx_path: ONNX 模型文件路径（如果提供，将优先使用 ONNX 模型）
        host: 主机地址
        port: 端口号
        pipeline_config_path: Pipeline 配置文件路径（已废弃，向后兼容用）
        **kwargs: 其他参数（传递给 create_app）
    """
    import uvicorn

    app = create_app(
        model_path=model_path,
        metadata_path=metadata_path,
        onnx_path=onnx_path,
        pipeline_config_path=pipeline_config_path,  # 向后兼容
        **kwargs,
    )

    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
