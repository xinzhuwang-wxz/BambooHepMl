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


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    将一批样本合并成批次张量。

    Args:
        batch: 输入批次，作为样本字典列表（每个样本是一个字典）

    Returns:
        输出批次，作为张量字典
    """
    if not batch:
        return {}

    # 获取所有样本的键（假设所有样本有相同的键）
    keys = batch[0].keys()
    tensor_batch = {}

    for key in keys:
        # 收集该键的所有值
        values = [sample[key] for sample in batch if key in sample]

        if not values:
            continue

        # 检查第一个值的类型
        first_value = values[0]

        if isinstance(first_value, torch.Tensor):
            # 如果已经是张量，直接堆叠
            tensor_batch[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            # 如果是 numpy 数组，转换为张量并堆叠
            if first_value.dtype == object:  # 不规则数组，需要填充
                # 转换为列表进行填充
                padded_values = [pad_array(v) if isinstance(v, np.ndarray) else v for v in values]
                tensor_batch[key] = torch.as_tensor(np.stack(padded_values), device=get_device())
            else:
                tensor_batch[key] = torch.as_tensor(np.stack(values), device=get_device())
        else:
            # 尝试导入 awkward（可选依赖）
            try:
                import awkward as ak

                if isinstance(first_value, ak.Array):
                    # 如果是 awkward array，转换为 numpy 然后堆叠
                    numpy_values = [ak.to_numpy(v) for v in values]
                    tensor_batch[key] = torch.as_tensor(np.stack(numpy_values), device=get_device())
                    continue
            except ImportError:
                pass

            # 标量值或其他类型
            if isinstance(first_value, (int, float)):
                # 标量值，转换为张量
                tensor_batch[key] = torch.tensor(values, device=get_device())
            else:
                # 其他类型，尝试转换为张量
                try:
                    tensor_batch[key] = torch.as_tensor(np.array(values), device=get_device())
                except Exception:
                    # 如果无法转换，保持原样（可能是其他特殊类型）
                    tensor_batch[key] = values

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
