"""
工具函数模块

提供：
- 随机种子设置
- 字典 I/O
- 数据预处理工具
"""

import json
import os
import random
from typing import Any

import numpy as np
import torch

from .config import mlflow

# 可选依赖：Ray
try:
    from ray.data import DatasetContext
    from ray.train.torch import get_device

    DatasetContext.get_current().execution_options.preserve_order = True
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

    def get_device():
        """当 Ray 不可用时，返回 CPU 设备。"""
        return torch.device("cpu")


__all__ = [
    "set_seeds",
    "load_dict",
    "save_dict",
    "pad_array",
    "collate_fn",
    "get_run_id",
    "DictDataset",
]


def set_seeds(seed: int = 42):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> dict:
    """从 JSON 文件加载字典。

    Args:
        path: 文件路径

    Returns:
        Dict: 加载的 JSON 数据
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """
    将字典保存到指定位置。

    Args:
        d: 要保存的数据
        path: 保存位置
        cls: 用于编码字典数据的编码器。默认为 None
        sortkeys: 是否按字母顺序排序键。默认为 False
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    将 2D 数组用零填充，直到所有行的长度与最长行相同。

    Args:
        arr: 输入数组
        dtype: 数据类型。默认为 np.int32

    Returns:
        零填充后的数组
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """
    将一批 numpy 数组转换为张量（带适当的填充）。

    Args:
        batch: 输入批次，作为 numpy 数组字典

    Returns:
        输出批次，作为张量字典
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
    """
    获取特定 Ray trial ID 的 MLflow run ID。

    Args:
        experiment_name: 实验名称
        trial_id: trial ID

    Returns:
        trial 的 run ID
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
