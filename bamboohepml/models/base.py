"""
模型基类

定义任务无关的模型接口，支持：
- Classification / Regression / Multitask
- Finetune（冻结/解冻层）
- 模型保存和加载
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):
    """
    模型基类（任务无关）

    所有模型都应该继承此类，实现：
    - forward(): 前向传播
    - predict(): 预测（分类返回类别，回归返回值）
    - predict_proba(): 预测概率（仅分类任务）

    支持：
    - Finetune（冻结/解冻层）
    - 模型保存和加载
    - 任务无关设计（不直接处理特征工程）
    """

    def __init__(self, **kwargs):
        """
        初始化模型。

        Args:
            **kwargs: 模型特定参数
        """
        super().__init__()
        self.config = kwargs
        self._frozen_layers = set()

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: 输入批次字典，包含模型需要的所有输入
                - 对于分类/回归：通常包含特征张量
                - 对于序列模型：可能包含 x, mask, v 等

        Returns:
            torch.Tensor: 模型输出
                - 分类任务：logits (batch_size, num_classes)
                - 回归任务：预测值 (batch_size, num_outputs)
                - 多任务：字典 {task_name: output}
        """
        pass

    @torch.inference_mode()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        预测（推理模式）。

        Args:
            batch: 输入批次字典

        Returns:
            - 分类任务：预测类别 (batch_size,)
            - 回归任务：预测值 (batch_size, num_outputs)
            - 多任务：字典 {task_name: predictions}
        """
        self.eval()
        output = self(batch)

        # 如果是字典（多任务），对每个任务分别处理
        if isinstance(output, dict):
            predictions = {}
            for task_name, task_output in output.items():
                predictions[task_name] = self._predict_single_task(task_output)
            return predictions
        else:
            return self._predict_single_task(output)

    def _predict_single_task(self, output: torch.Tensor) -> torch.Tensor:
        """
        单个任务的预测逻辑。

        Args:
            output: 模型输出

        Returns:
            预测结果
        """
        # 默认实现：对于多类输出，返回 argmax；对于单输出，直接返回
        if output.dim() > 1 and output.size(1) > 1:
            # 分类任务：返回类别索引
            return torch.argmax(output, dim=1).cpu()
        else:
            # 回归任务：返回预测值
            return output.squeeze().cpu()

    @torch.inference_mode()
    def predict_proba(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        预测概率（仅分类任务）。

        Args:
            batch: 输入批次字典

        Returns:
            - 分类任务：概率分布 (batch_size, num_classes)
            - 多任务：字典 {task_name: probabilities}
        """
        self.eval()
        output = self(batch)

        # 如果是字典（多任务），对每个任务分别处理
        if isinstance(output, dict):
            probabilities = {}
            for task_name, task_output in output.items():
                probabilities[task_name] = F.softmax(task_output, dim=1).cpu()
            return probabilities
        else:
            return F.softmax(output, dim=1).cpu()

    def freeze_layers(self, layer_names: list[str] | None = None, freeze_all: bool = False):
        """
        冻结层（用于 finetune）。

        Args:
            layer_names: 要冻结的层名称列表。如果为 None，则根据 freeze_all 决定
            freeze_all: 如果为 True，冻结所有层（除了分类/回归头）
        """
        if freeze_all:
            # 冻结所有层，除了分类/回归头
            for name, param in self.named_parameters():
                if "head" not in name.lower() and "classifier" not in name.lower():
                    param.requires_grad = False
                    self._frozen_layers.add(name)
        elif layer_names:
            # 冻结指定的层
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
                    self._frozen_layers.add(name)

    def unfreeze_layers(self, layer_names: list[str] | None = None, unfreeze_all: bool = False):
        """
        解冻层。

        Args:
            layer_names: 要解冻的层名称列表。如果为 None，则根据 unfreeze_all 决定
            unfreeze_all: 如果为 True，解冻所有层
        """
        if unfreeze_all:
            # 解冻所有层
            for name, param in self.named_parameters():
                param.requires_grad = True
            self._frozen_layers.clear()
        elif layer_names:
            # 解冻指定的层
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    self._frozen_layers.discard(name)

    def get_frozen_layers(self) -> list[str]:
        """
        获取当前冻结的层名称列表。

        Returns:
            冻结的层名称列表
        """
        return list(self._frozen_layers)

    def save(self, save_dir: str | Path, model_name: str = "model"):
        """
        保存模型。

        Args:
            save_dir: 保存目录
            model_name: 模型文件名（不含扩展名）
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型配置
        config_path = save_dir / f"{model_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        # 保存模型权重
        weights_path = save_dir / f"{model_name}.pt"
        torch.save(self.state_dict(), weights_path)

        # 保存冻结层信息
        frozen_path = save_dir / f"{model_name}_frozen.json"
        with open(frozen_path, "w") as f:
            json.dump(self._frozen_layers, f, indent=2)

    @classmethod
    @abstractmethod
    def load(cls, save_dir: str | Path, model_name: str = "model", **kwargs):
        """
        加载模型。

        Args:
            save_dir: 保存目录
            model_name: 模型文件名（不含扩展名）
            **kwargs: 额外的加载参数

        Returns:
            加载的模型实例
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """
        获取模型信息（用于日志和调试）。

        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_layers": list(self._frozen_layers),
            "config": self.config,
        }


class ClassificationModel(BaseModel):
    """
    分类模型基类（可选，提供分类特定的辅助方法）
    """

    def __init__(self, num_classes: int, **kwargs):
        """
        初始化分类模型。

        Args:
            num_classes: 类别数量
            **kwargs: 其他模型参数
        """
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes

    def _predict_single_task(self, output: torch.Tensor) -> torch.Tensor:
        """分类任务的预测：返回类别索引"""
        return torch.argmax(output, dim=1).cpu()


class RegressionModel(BaseModel):
    """
    回归模型基类（可选，提供回归特定的辅助方法）
    """

    def __init__(self, num_outputs: int = 1, **kwargs):
        """
        初始化回归模型。

        Args:
            num_outputs: 输出数量（默认为 1）
            **kwargs: 其他模型参数
        """
        super().__init__(num_outputs=num_outputs, **kwargs)
        self.num_outputs = num_outputs

    def _predict_single_task(self, output: torch.Tensor) -> torch.Tensor:
        """回归任务的预测：直接返回预测值"""
        return output.squeeze().cpu()

    def predict_proba(self, batch: dict[str, torch.Tensor]) -> None:
        """回归任务不支持 predict_proba"""
        raise NotImplementedError("Regression models do not support predict_proba")


class MultitaskModel(BaseModel):
    """
    多任务模型基类

    输出格式：Dict[str, torch.Tensor]，每个键对应一个任务
    """

    def __init__(self, tasks: dict[str, dict[str, Any]], **kwargs):
        """
        初始化多任务模型。

        Args:
            tasks: 任务配置字典，格式为 {task_name: task_config}
                task_config 包含任务类型和参数
            **kwargs: 其他模型参数
        """
        super().__init__(tasks=tasks, **kwargs)
        self.tasks = tasks

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        多任务前向传播。

        Returns:
            字典 {task_name: output}
        """
        raise NotImplementedError("Subclasses must implement forward()")
