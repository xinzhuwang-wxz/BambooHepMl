"""
模型元数据工具

提供保存和加载模型元数据的功能。
模型元数据包含：
- feature_spec: 特征规范（来自 FeatureGraph.output_spec()）
- task_type: 任务类型
- model_config: 模型配置（包含 event_input_dim/object_input_dim/embed_dim 等）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import logger

__all__ = ["save_model_metadata", "load_model_metadata"]


def save_model_metadata(
    metadata_path: str | Path,
    feature_spec: dict[str, Any],
    task_type: str,
    model_config: dict[str, Any],
    **kwargs,
) -> None:
    """
    保存模型元数据到 JSON 文件。

    Args:
        metadata_path: 元数据文件路径
        feature_spec: 特征规范（来自 FeatureGraph.output_spec()）
        task_type: 任务类型（classification/regression）
        model_config: 模型配置（应包含 event_input_dim/object_input_dim/embed_dim）
        **kwargs: 其他元数据（如 feature_state, experiment_name 等）
    """
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_spec": feature_spec,
        "task_type": task_type,
        "model_config": model_config,
        **kwargs,  # 其他字段
    }

    # 序列化（处理不可序列化的类型）
    def default_serializer(obj):
        """默认序列化器，处理 numpy/torch 类型。"""
        import numpy as np
        import torch

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=default_serializer)

    logger.info(f"Model metadata saved to {metadata_path}")


def load_model_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """
    从 JSON 文件加载模型元数据。

    Args:
        metadata_path: 元数据文件路径

    Returns:
        dict: 模型元数据
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(f"Model metadata loaded from {metadata_path}")
    return metadata
