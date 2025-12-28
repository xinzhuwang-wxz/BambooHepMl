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
from ..models import get_model
from ..utils.metadata import load_model_metadata


# 请求模型
class PredictRequest(BaseModel):
    """预测请求模型。"""

    features: list[list[float]] = Field(..., description="特征向量列表")
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
    input_key: str | None = None

    @app.on_event("startup")
    async def startup_event():
        """启动时加载模型。"""
        nonlocal predictor, model_info, input_key
        try:
            from pathlib import Path

            # 优先使用 ONNX 模型
            if onnx_path and Path(onnx_path).exists():
                try:
                    from ..serve.onnx_predictor import ONNXPredictor

                    predictor = ONNXPredictor(onnx_path)
                    logger.info(f"ONNX model loaded from {onnx_path}")
                    model_info = {"model_type": "onnx", "onnx_path": onnx_path}

                    # 尝试加载 metadata 获取 input_key
                    resolved_metadata_path = metadata_path
                    if resolved_metadata_path is None and model_path:
                        resolved_metadata_path = str(Path(model_path).parent / "metadata.json")
                    if resolved_metadata_path and Path(resolved_metadata_path).exists():
                        metadata = load_model_metadata(resolved_metadata_path)
                        input_key = metadata.get("input_key", "event")
                    else:
                        input_key = "event"  # 默认
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
                input_dim = metadata["input_dim"]
                input_key = metadata["input_key"]
                model_name = model_config.get("name")
                model_params = model_config.get("params", {}).copy()
                model_params["input_dim"] = input_dim
            elif pipeline_config_path:
                # 向后兼容：从 pipeline 配置加载（已废弃）
                logger.warning(f"Metadata file not found at {resolved_metadata_path}. " "Falling back to pipeline_config_path (deprecated).")
                from ..pipeline import PipelineOrchestrator

                orchestrator = PipelineOrchestrator(pipeline_config_path)
                model_config = orchestrator.get_model_config()
                model_name = model_config.get("name")
                model_params = model_config.get("params", {})

                # 从数据推断输入维度
                dataset = orchestrator.setup_data()
                sample = next(iter(dataset))
                input_key = None
                for key in sample.keys():
                    if key.startswith("_") and key != "_label_":
                        input_key = key
                        break

                if input_key is None:
                    raise ValueError("Could not find input key in dataset")

                input_value = sample[input_key]
                if isinstance(input_value, torch.Tensor):
                    if len(input_value.shape) == 1:
                        input_dim = input_value.shape[0]
                    elif len(input_value.shape) == 2:
                        input_dim = input_value.shape[1]
                    else:
                        input_dim = int(torch.prod(torch.tensor(input_value.shape)))
                else:
                    raise ValueError(f"Unexpected input type: {type(input_value)}")

                model_params["input_dim"] = input_dim
            elif model_name and model_params:
                # 向后兼容：直接使用提供的参数
                input_key = "event"  # 默认
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
                "input_key": input_key,
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
            # 转换为 torch.Tensor
            features_tensor = torch.tensor(request.features, dtype=torch.float32)

            # 创建数据集（使用正确的 input_key）
            from torch.utils.data import DataLoader

            from ..utils import DictDataset

            # 使用 metadata 中的 input_key（默认为 "event"）
            actual_input_key = input_key or "event"
            dataset = DictDataset([{actual_input_key: features_tensor}])
            dataloader = DataLoader(dataset, batch_size=len(request.features))

            # 预测
            results = predictor.predict(dataloader, return_probabilities=request.return_probabilities)

            predictions = [r["prediction"] for r in results]
            probabilities = None
            if request.return_probabilities and "probabilities" in results[0]:
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
            # 提取所有样本的特征
            all_features = []
            for sample in request.samples:
                if "features" in sample:
                    all_features.append(sample["features"])
                else:
                    raise ValueError("Each sample must have 'features' key")

            # 转换为 torch.Tensor
            features_tensor = torch.tensor(all_features, dtype=torch.float32)

            # 创建数据集（使用正确的 input_key）
            from torch.utils.data import DataLoader

            from ..utils import DictDataset

            # 使用 metadata 中的 input_key（默认为 "event"）
            actual_input_key = input_key or "event"
            dataset = DictDataset([{actual_input_key: features_tensor}])
            dataloader = DataLoader(dataset, batch_size=len(all_features))

            # 预测
            results = predictor.predict(dataloader, return_probabilities=request.return_probabilities)

            predictions = [r["prediction"] for r in results]
            probabilities = None
            if request.return_probabilities and "probabilities" in results[0]:
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
