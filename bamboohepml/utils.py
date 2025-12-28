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
from typing import Any, Dict, List

import numpy as np
import torch
from ray.data import DatasetContext
from ray.train.torch import get_device

from bamboohepml.config import mlflow

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


def load_dict(path: str) -> Dict:
    """从 JSON 文件加载字典。

    Args:
        path (str): 文件路径。

    Returns:
        Dict: 加载的 JSON 数据。
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
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


def collate_fn(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
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


def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    """将字典转换为字典列表。

    Args:
        data (Dict): 输入字典。
        keys (List[str]): 要包含在输出字典列表中的键。

    Returns:
        List[Dict[str, Any]]: 输出字典列表。
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
