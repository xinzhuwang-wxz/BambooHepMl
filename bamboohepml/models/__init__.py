"""
模型模块

提供：
- BaseModel: 任务无关的模型基类
- 模型注册和发现机制
- get_model: 模型工厂函数
"""

# 导入通用模型以触发注册
from . import common  # noqa: F401
from .base import BaseModel, ClassificationModel, MultitaskModel, RegressionModel
from .registry import ModelRegistry, get_model, register_model

__all__ = [
    "BaseModel",
    "ClassificationModel",
    "RegressionModel",
    "MultitaskModel",
    "ModelRegistry",
    "get_model",
    "register_model",
]
