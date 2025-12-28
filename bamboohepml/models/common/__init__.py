"""
通用模型模块

提供：
- MLP: 多层感知机（支持分类和回归）
- Transformer: Transformer 模型（后续实现）
"""

# 导入 MLP 模型以触发注册装饰器
from .mlp import MLPClassifier, MLPRegressor  # noqa: F401

__all__ = [
    "MLPClassifier",
    "MLPRegressor",
]
