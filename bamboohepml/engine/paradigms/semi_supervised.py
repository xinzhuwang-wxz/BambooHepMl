"""
半监督学习范式

支持多种半监督学习策略：
- Self-training: 使用模型预测生成伪标签
- Consistency regularization: 对无标签数据添加噪声，要求输出一致
- Pseudo-labeling: 基于置信度的伪标签
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import logger
from .base import LearningParadigm


class SemiSupervisedParadigm(LearningParadigm):
    """
    半监督学习范式

    支持多种策略：
    - self-training: 使用模型预测的 argmax 作为伪标签
    - consistency: 对无标签数据添加噪声，要求输出一致
    - pseudo-labeling: 基于置信度的伪标签（只对高置信度样本使用）

    标签约定（重要）：
    - 有标签样本：labels >= 0（例如：0, 1, 2, ... 用于分类；实际值用于回归）
    - 无标签样本：labels == -1
    - 这是标准的半监督学习约定，与 scikit-learn 的 LabelSpreading 等保持一致

    示例：
        labels = torch.tensor([0, 1, -1, 0, -1])  # 前两个和后一个样本有标签，中间两个无标签
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        初始化半监督学习范式。

        Args:
            config: 配置字典，可包含：
                - strategy: "self-training", "consistency", "pseudo-labeling" (默认: "self-training")
                - unsupervised_weight: 无监督损失的权重（默认: 0.1）
                - confidence_threshold: 伪标签的置信度阈值（默认: 0.9，仅用于 pseudo-labeling）
                - task_type: "classification" 或 "regression"（默认: "classification"）
        """
        super().__init__(config)
        self.strategy = config.get("strategy", "self-training") if config else "self-training"
        self.unsupervised_weight = config.get("unsupervised_weight", 0.1) if config else 0.1
        self.confidence_threshold = config.get("confidence_threshold", 0.9) if config else 0.9
        self.task_type = config.get("task_type", "classification") if config else "classification"

        valid_strategies = ["self-training", "consistency", "pseudo-labeling"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        outputs: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        计算半监督损失。

        Args:
            model: 模型
            inputs: 输入张量
            labels: 标签（对于有标签数据）
            outputs: 模型输出
            loss_fn: 损失函数（如果为 None，将从 task_type 推断）

        Returns:
            torch.Tensor: 组合损失（supervised + unsupervised）
        """
        if loss_fn is None:
            loss_fn = self.get_default_loss_fn(self.task_type)

        # 分离有标签和无标签数据
        # P1-4 修复：明确标签约定
        # 约定：labels >= 0 表示有标签样本，labels == -1 表示无标签样本
        # 这是标准的半监督学习约定（与 scikit-learn 的 LabelSpreading 等一致）
        if labels is not None:
            labeled_mask = labels >= 0  # 有标签样本：label >= 0，无标签样本：label == -1
            labeled_inputs = inputs[labeled_mask]
            labeled_labels = labels[labeled_mask]
            labeled_outputs = outputs[labeled_mask]
            unlabeled_inputs = inputs[~labeled_mask]
        else:
            # 如果没有提供 labels，所有数据都是无标签的
            labeled_inputs = None
            labeled_labels = None
            labeled_outputs = None
            unlabeled_inputs = inputs

        # 有监督损失
        if labeled_labels is not None and len(labeled_labels) > 0:
            if self.task_type == "classification":
                supervised_loss = loss_fn(labeled_outputs, labeled_labels)
            else:  # regression
                supervised_loss = loss_fn(labeled_outputs.squeeze(), labeled_labels.float())
        else:
            supervised_loss = torch.tensor(0.0, device=inputs.device)

        # 无监督损失（根据策略）
        if unlabeled_inputs is not None and len(unlabeled_inputs) > 0:
            unlabeled_outputs = model({"features": unlabeled_inputs})

            if self.strategy == "self-training":
                unsupervised_loss = self._self_training_loss(unlabeled_outputs, loss_fn)
            elif self.strategy == "consistency":
                # 对输入添加噪声，要求输出一致
                unsupervised_loss = self._consistency_loss(model, unlabeled_inputs, unlabeled_outputs)
            elif self.strategy == "pseudo-labeling":
                unsupervised_loss = self._pseudo_labeling_loss(unlabeled_outputs, loss_fn)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        else:
            unsupervised_loss = torch.tensor(0.0, device=inputs.device)

        # 组合损失
        total_loss = supervised_loss + self.unsupervised_weight * unsupervised_loss

        return total_loss

    def _self_training_loss(self, outputs: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        Self-training 策略：使用模型预测的 argmax 作为伪标签。

        Args:
            outputs: 模型输出
            loss_fn: 损失函数

        Returns:
            torch.Tensor: 无监督损失
        """
        if self.task_type == "classification":
            # 使用 argmax 作为伪标签
            pseudo_labels = torch.argmax(outputs.detach(), dim=1)
            loss = loss_fn(outputs, pseudo_labels)
        else:  # regression
            # 对于回归，使用模型输出本身作为目标（重构误差）
            loss = loss_fn(outputs.squeeze(), outputs.detach().squeeze())

        return loss

    def _consistency_loss(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Consistency regularization 策略：对输入添加噪声，要求输出一致。

        Args:
            model: 模型
            inputs: 输入张量
            outputs: 原始输出

        Returns:
            torch.Tensor: 一致性损失
        """
        # 添加噪声（高斯噪声）
        noise = torch.randn_like(inputs) * 0.1
        noisy_inputs = inputs + noise
        noisy_outputs = model({"features": noisy_inputs})

        # 计算输出的一致性损失（MSE）
        if self.task_type == "classification":
            # 对分类任务，使用概率分布的一致性
            probs_original = F.softmax(outputs, dim=1)
            probs_noisy = F.softmax(noisy_outputs, dim=1)
            loss = F.mse_loss(probs_noisy, probs_original.detach())
        else:  # regression
            loss = F.mse_loss(noisy_outputs.squeeze(), outputs.detach().squeeze())

        return loss

    def _pseudo_labeling_loss(self, outputs: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        Pseudo-labeling 策略：只对高置信度的预测使用伪标签。

        Args:
            outputs: 模型输出
            loss_fn: 损失函数

        Returns:
            torch.Tensor: 无监督损失
        """
        if self.task_type == "classification":
            # 计算置信度（最大概率）
            probs = F.softmax(outputs, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)

            # 只对高置信度的样本计算损失
            high_confidence_mask = confidence > self.confidence_threshold

            if high_confidence_mask.sum() > 0:
                high_conf_outputs = outputs[high_confidence_mask]
                high_conf_labels = pseudo_labels[high_confidence_mask]
                loss = loss_fn(high_conf_outputs, high_conf_labels)
            else:
                loss = torch.tensor(0.0, device=outputs.device)
        else:
            # 对于回归，使用 self-training 策略
            loss = self._self_training_loss(outputs, loss_fn)

        return loss
