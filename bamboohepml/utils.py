"""
工具函数模块

借鉴 Made-With-ML 的工具函数，提供：
- 随机种子设置
- 字典 I/O
- 数据预处理工具
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.data import DatasetContext
from ray.train.torch import get_device

from bamboohepml.config import logger, mlflow

DatasetContext.get_current().execution_options.preserve_order = True


def set_seeds(seed: int = 42):
    """设置随机种子以确保可重复性。

    Args:
        seed (int): 随机种子值。默认为 42。
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> dict:
    """从 JSON 文件加载字典。

    Args:
        path (str): 文件路径。

    Returns:
        Dict: 加载的 JSON 数据。
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """将字典保存到指定位置。

    Args:
        d (Dict): 要保存的数据。
        path (str): 保存位置。
        cls (optional): 用于编码字典数据的编码器。默认为 None。
        sortkeys (bool, optional): 是否按字母顺序排序键。默认为 False。
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """将 2D 数组用零填充，直到所有行的长度与最长行相同。

    Args:
        arr (np.ndarray): 输入数组。
        dtype: 数据类型。默认为 np.int32。

    Returns:
        np.ndarray: 零填充后的数组。
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """将一批 numpy 数组转换为张量（带适当的填充）。

    Args:
        batch (Dict[str, np.ndarray]): 输入批次，作为 numpy 数组字典。

    Returns:
        Dict[str, torch.Tensor]: 输出批次，作为张量字典。
    """
    # 对于 HEP 数据，可能需要不同的填充策略
    # 这里提供一个通用版本，具体任务可以覆盖
    tensor_batch = {}
    for key, array in batch.items():
        if isinstance(array, np.ndarray):
            if array.dtype == object:  # 不规则数组
                array = pad_array(array)
            tensor_batch[key] = torch.as_tensor(array, device=get_device())
        else:
            tensor_batch[key] = torch.as_tensor(array, device=get_device())
    return tensor_batch


def get_run_id(experiment_name: str, trial_id: str) -> str:
    """获取特定 Ray trial ID 的 MLflow run ID。

    Args:
        experiment_name (str): 实验名称。
        trial_id (str): trial ID。

    Returns:
        str: trial 的 run ID。
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.trial_name = '{trial_name}'",
    ).iloc[0]
    return run.run_id


class DictDataset(torch.utils.data.Dataset):
    """简单的字典数据集，用于将字典数据转换为 DataLoader 输入。"""

    def __init__(self, data: list[dict[str, Any]]):
        """
        初始化数据集。

        Args:
            data: 字典列表，每个字典包含一个样本的数据
        """
        self.data = data

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取指定索引的样本。"""
        return self.data[idx]


def save_model_metadata(
    metadata_path: str | Path,
    feature_spec: dict[str, Any],
    task_type: str,
    model_config: dict[str, Any],
    input_dim: int,
    input_key: str,
    **kwargs,
) -> None:
    """
    保存模型元数据到 JSON 文件。

    Args:
        metadata_path: 元数据文件路径
        feature_spec: 特征规范（来自 FeatureGraph.output_spec()）
        task_type: 任务类型（classification/regression）
        model_config: 模型配置
        input_dim: 输入维度
        input_key: 输入键名（event/object）
        **kwargs: 其他元数据（如 num_classes, output_dir 等）
    """
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_spec": feature_spec,
        "task_type": task_type,
        "model_config": model_config,
        "input_dim": input_dim,
        "input_key": input_key,
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
