"""
Ray Serve 集成

使用 Ray Serve 部署模型服务。
"""

from http import HTTPStatus
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from ..config import logger
from ..engine import Predictor
from ..models import get_model
from ..pipeline import PipelineOrchestrator

# 定义 FastAPI 应用
app = FastAPI(
    title="BambooHepMl Ray Serve",
    description="高能物理机器学习推理服务（Ray Serve）",
    version="0.1.0",
)


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 0})
@serve.ingress(app)
class RayServeDeployment:
    """
    Ray Serve 部署类

    使用 Ray Serve 部署模型服务。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline_config_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化部署。

        Args:
            model_path: 模型文件路径
            pipeline_config_path: Pipeline 配置文件路径
            run_id: MLflow run ID（如果使用 MLflow）
            model_name: 模型名称
            model_params: 模型参数
        """
        self.model_path = model_path
        self.pipeline_config_path = pipeline_config_path
        self.run_id = run_id

        # 加载模型
        self._load_model(model_name, model_params)

    def _load_model(self, model_name: Optional[str], model_params: Optional[Dict[str, Any]]):
        """加载模型。"""
        try:
            if self.pipeline_config_path:
                # 从 pipeline 配置加载
                orchestrator = PipelineOrchestrator(self.pipeline_config_path)
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
            if self.model_path:
                state_dict = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded from {self.model_path}")

            # 创建预测器
            self.predictor = Predictor(model)
            logger.info("Model loaded successfully in Ray Serve")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @app.get("/")
    def _index(self) -> Dict[str, Any]:
        """健康检查。"""
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model_path": self.model_path,
                "run_id": self.run_id,
            },
        }

    @app.get("/run_id")
    def _run_id(self) -> Dict[str, str]:
        """获取 run ID。"""
        return {"run_id": self.run_id or "N/A"}

    @app.post("/predict")
    async def _predict(self, request: Request) -> Dict[str, Any]:
        """预测。"""
        data = await request.json()
        features = data.get("features", [])
        return_probabilities = data.get("return_probabilities", False)

        if not features:
            return {
                "error": "No features provided",
                "status-code": HTTPStatus.BAD_REQUEST,
            }

        try:
            # 转换为 tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # 创建临时 dataloader
            from torch.utils.data import DataLoader, TensorDataset

            dataset = TensorDataset(features_tensor)
            dataloader = DataLoader(dataset, batch_size=len(features))

            # 预测
            results = self.predictor.predict(
                dataloader,
                return_probabilities=return_probabilities,
            )

            return {
                "results": results,
                "status-code": HTTPStatus.OK,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "error": str(e),
                "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            }


def serve_ray(
    model_path: Optional[str] = None,
    pipeline_config_path: Optional[str] = None,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    启动 Ray Serve 服务。

    Args:
        model_path: 模型文件路径
        pipeline_config_path: Pipeline 配置文件路径
        run_id: MLflow run ID
        model_name: 模型名称
        model_params: 模型参数
        **kwargs: 其他参数（传递给 RayServeDeployment）
    """
    import ray

    # 初始化 Ray（如果未初始化）
    if not ray.is_initialized():
        ray.init()

    # 绑定部署
    deployment = RayServeDeployment.bind(
        model_path=model_path,
        pipeline_config_path=pipeline_config_path,
        run_id=run_id,
        model_name=model_name,
        model_params=model_params,
        **kwargs,
    )

    # 启动服务
    logger.info("Starting Ray Serve deployment...")
    serve.run(deployment)
