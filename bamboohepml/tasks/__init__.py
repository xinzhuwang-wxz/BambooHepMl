"""
Tasks 子系统

包含：
- train: 训练任务
- predict: 预测任务
- export: 导出任务（ONNX）
- inspect: 检查任务（数据/特征检查）
"""
from .export import export_task
from .inspect import inspect_task
from .predict import predict_task
from .train import train_task

__all__ = ["train_task", "predict_task", "export_task", "inspect_task"]
