"""
特征工程模块

提供：
- ExpressionEngine: 表达式引擎
- FeatureGraph: 特征依赖图（DAG）
- FeatureProcessor: 特征处理器
- OperatorRegistry: 函数注册表
"""

from .expression import ExpressionEngine, OperatorRegistry
from .feature_graph import FeatureGraph, FeatureNode
from .processors import Clipper, FeatureProcessor, Normalizer, Padder

__all__ = [
    "ExpressionEngine",
    "OperatorRegistry",
    "FeatureGraph",
    "FeatureNode",
    "FeatureProcessor",
    "Normalizer",
    "Clipper",
    "Padder",
]
