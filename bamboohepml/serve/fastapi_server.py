"""
FastAPI 推理服务

提供 RESTful API 接口用于模型推理。
"""

from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import logger
from ..engine import Predictor
from ..models import get_model
from ..pipeline import PipelineOrchestrator


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
    pipeline_config_path: str | None = None,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用。

    Args:
        model_path: 模型文件路径
        pipeline_config_path: Pipeline 配置文件路径（用于加载模型配置）
        model_name: 模型名称（如果 pipeline_config_path 未提供）
        model_params: 模型参数（如果 pipeline_config_path 未提供）

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
            if pipeline_config_path:
                # 从 pipeline 配置加载
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
                model = orchestrator.setup_model(input_dim=input_dim)
            elif model_name and model_params:
                # 直接使用提供的参数
                model = get_model(model_name, **model_params)
            else:
                raise ValueError("Must provide either pipeline_config_path or (model_name and model_params)")

            # 加载权重
            if model_path:
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded from {model_path}")

            # 创建预测器
            predictor = Predictor(model)
            # Set model_info - handle both cases
            if pipeline_config_path:
                final_model_name = model_name or model_config.get("name", "unknown")
                final_model_params = model_params or model_config.get("params", {})
            else:
                final_model_name = model_name or "unknown"
                final_model_params = model_params or {}
            model_info = {
                "model_name": final_model_name,
                "model_params": final_model_params,
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

            # 创建临时数据集
            from torch.utils.data import Dataset

            class DictDataset(Dataset):
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            dataset = DictDataset([{"features": features_tensor}])
            from torch.utils.data import DataLoader

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
            # 转换为 torch.Tensor
            from torch.utils.data import Dataset

            class DictDataset(Dataset):
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            # 提取所有样本的特征
            all_features = []
            for sample in request.samples:
                if "features" in sample:
                    all_features.append(sample["features"])
                else:
                    raise ValueError("Each sample must have 'features' key")

            features_tensor = torch.tensor(all_features, dtype=torch.float32)
            dataset = DictDataset([{"features": features_tensor}])
            from torch.utils.data import DataLoader

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


def serve_fastapi(model_path: str, pipeline_config_path: str | None = None, host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """
    启动 FastAPI 服务。

    Args:
        model_path: 模型文件路径
        pipeline_config_path: Pipeline 配置文件路径
        host: 主机地址
        port: 端口号
        **kwargs: 其他参数（传递给 create_app）
    """
    import uvicorn

    app = create_app(model_path=model_path, pipeline_config_path=pipeline_config_path, **kwargs)

    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
