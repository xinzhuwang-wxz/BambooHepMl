"""
有监督学习范式

标准的有监督学习，使用标签计算损失。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import LearningParadigm


class SupervisedParadigm(LearningParadigm):
    """
    有监督学习范式

    使用标签计算标准的有监督损失（分类或回归）。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        初始化有监督学习范式。

        Args:
            config: 配置字典，可包含：
                - task_type: "classification" 或 "regression"（默认从数据推断）
        """
        super().__init__(config)
        self.task_type = config.get("task_type") if config else None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        outputs: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        计算有监督损失。

        Args:
            model: 模型
            inputs: 输入张量（未使用，但保留接口一致性）
            labels: 标签（必需）
            outputs: 模型输出
            loss_fn: 损失函数（如果为 None，将从 task_type 推断）

        Returns:
            torch.Tensor: 损失值

        Raises:
            ValueError: 如果 labels 为 None
        """
        if labels is None:
            raise ValueError("Supervised learning requires labels. Got None.")

        # 确定任务类型
        if self.task_type:
            task_type = self.task_type
        else:
            # 从 outputs 形状推断：如果是分类，outputs.shape[1] > 1
            task_type = "classification" if outputs.shape[1] > 1 else "regression"

        # 如果没有提供 loss_fn，使用默认的
        if loss_fn is None:
            loss_fn = self.get_default_loss_fn(task_type)

        # 计算损失
        # 根据 loss_fn 类型决定如何处理 labels
        # CrossEntropyLoss 需要 Long 类型的 labels，MSELoss 需要 Float 类型
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # 分类任务：确保 labels 是 Long 类型
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = loss_fn(outputs, labels)
        elif isinstance(loss_fn, nn.MSELoss):
            # 回归任务：确保 labels 是 Float 类型
            if labels.dtype != torch.float32:
                labels = labels.float()
            loss = loss_fn(outputs.squeeze(), labels)
        else:
            # 对于其他损失函数，根据 task_type 处理
            if task_type == "classification":
                loss = loss_fn(outputs, labels)
            else:  # regression
                loss = loss_fn(outputs.squeeze(), labels.float())

        return loss
