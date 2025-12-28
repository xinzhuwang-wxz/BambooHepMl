"""
服务模块

提供：
- FastAPI 推理服务
- Ray Serve 集成
- ONNX 推理接口
"""
from .fastapi_server import create_app, serve_fastapi
from .ray_serve import RayServeDeployment, serve_ray
from .onnx_predictor import ONNXPredictor

__all__ = [
    'create_app',
    'serve_fastapi',
    'RayServeDeployment',
    'serve_ray',
    'ONNXPredictor',
]

