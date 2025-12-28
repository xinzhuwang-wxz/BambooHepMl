"""
学习范式基类

定义学习范式的统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...config import logger


class LearningParadigm(ABC):
    """
    学习范式基类

    所有学习范式都应该继承此类，实现：
    - compute_loss(): 计算损失
    - prepare_data(): 准备数据（如果需要特殊处理）
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        初始化学习范式。

        Args:
            config: 范式特定配置
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        outputs: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        计算损失。

        Args:
            model: 模型
            inputs: 输入张量
            labels: 标签（如果有）
            outputs: 模型输出
            loss_fn: 损失函数（可选）

        Returns:
            torch.Tensor: 损失值
        """
        pass

    def prepare_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        准备批次数据（如果需要特殊处理）。

        默认实现：提取 inputs 和 labels。

        Args:
            batch: 批次字典

        Returns:
            tuple: (inputs, labels)
        """
        # 找到输入键
        input_key = None
        for key in batch.keys():
            if key in ["event", "object"] or (key.startswith("_") and key != "_label_"):
                input_key = key
                break

        if input_key is None:
            raise ValueError(f"Could not find input key in batch. Available keys: {list(batch.keys())}")

        inputs = batch[input_key]
        labels = batch.get("_label_", None)

        return inputs, labels

    def get_default_loss_fn(self, task_type: str = "classification") -> nn.Module:
        """
        获取默认损失函数。

        Args:
            task_type: 任务类型（classification, regression）

        Returns:
            nn.Module: 损失函数
        """
        if task_type == "classification":
            return nn.CrossEntropyLoss()
        elif task_type == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        验证配置。

        Returns:
            tuple[bool, list[str]]: (是否有效, 错误列表)
        """
        return True, []

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
