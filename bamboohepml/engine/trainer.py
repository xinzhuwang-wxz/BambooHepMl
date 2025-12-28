"""
统一 Trainer

支持：
- Supervised: 有监督学习 ✅
- Semi-supervised: 半监督学习 ✅ (支持 self-training, consistency, pseudo-labeling)
- Unsupervised: 无监督学习 ✅ (支持 autoencoder, VAE, contrastive)
- Loss 与 Model 解耦 ✅
- Callback 系统（logging / early stop）✅
- 多任务 Loss ⚠️ (部分支持，需要自定义 loss_fn)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import logger
from ..models import BaseModel
from .callbacks import Callback, EarlyStoppingCallback
from .evaluator import Evaluator
from .paradigms import LearningParadigm, SemiSupervisedParadigm, SupervisedParadigm, UnsupervisedParadigm


def get_default_device() -> torch.device:
    """获取默认设备（支持 CUDA, MPS, CPU）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Trainer:
    """
    统一训练器，支持多种学习范式和任务类型。

    Trainer 通过 LearningParadigm 系统实现学习逻辑的解耦，支持：
    - 监督学习：标准分类/回归任务
    - 半监督学习：self-training、consistency regularization、pseudo-labeling
    - 无监督学习：autoencoder、VAE、contrastive learning

    设计特点：
    - Loss 计算完全委托给 LearningParadigm，实现范式与训练器的解耦
    - 自动设备检测（CUDA/MPS/CPU）
    - 支持回调系统（logging、early stopping 等）
    - 自动识别输入键（event/object）以适配 FeatureGraph 的输出格式
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader | Iterable,
        val_loader: DataLoader | Iterable | None = None,
        test_loader: DataLoader | Iterable | None = None,
        loss_fn: nn.Module | Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: torch.device | str | None = None,
        task_type: str = "supervised",
        learning_paradigm: str | LearningParadigm | None = None,
        paradigm_config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
    ):
        """
        初始化训练器。

        Args:
            model: 要训练的模型实例
            train_loader: 训练数据加载器（DataLoader 或任意可迭代对象）
            val_loader: 验证数据加载器（可选）
            test_loader: 测试数据加载器（可选）
            loss_fn: 损失函数（可选，某些范式会使用自己的损失函数）
            optimizer: 优化器（默认使用 Adam，lr=1e-3）
            scheduler: 学习率调度器（可选）
            device: 计算设备（None 时自动检测：CUDA > MPS > CPU）
            task_type: 任务类型（'classification'/'regression'/'multitask'），用于评估器
            learning_paradigm: 学习范式名称或实例（'supervised'/'semi-supervised'/'unsupervised'）
            paradigm_config: 学习范式的配置字典
            callbacks: 回调函数列表（用于日志记录、early stopping 等）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设备处理
        if device is None:
            self.device = get_default_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.task_type = task_type
        self.callbacks = callbacks or []

        # 创建或设置学习范式
        if learning_paradigm is None:
            # 默认使用有监督学习
            learning_paradigm = "supervised"

        if isinstance(learning_paradigm, str):
            self.paradigm = self._create_paradigm(learning_paradigm, paradigm_config)
            self.learning_paradigm_name = learning_paradigm
        elif isinstance(learning_paradigm, LearningParadigm):
            self.paradigm = learning_paradigm
            self.learning_paradigm_name = learning_paradigm.name
        else:
            raise ValueError(
                f"Invalid learning_paradigm: {learning_paradigm}. "
                "Must be a string ('supervised', 'semi-supervised', 'unsupervised') "
                "or a LearningParadigm instance."
            )

        # 验证范式配置
        is_valid, errors = self.paradigm.validate_config()
        if not is_valid:
            raise ValueError(f"Paradigm config validation failed: {errors}")

        # 设置模型到设备
        self.model.to(self.device)

        # 设置损失函数（可选，paradigm 可能使用自己的损失函数）
        self.loss_fn = loss_fn

        # 设置优化器
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        # 设置调度器
        self.scheduler = scheduler

        # 设置评估器
        self.evaluator = Evaluator(task_type="classification" if task_type == "classification" else "regression")

        # 找到输入键（从 FeatureGraph.build_batch 返回的键：event 或 object）
        sample = next(iter(train_loader))
        self.input_key = None
        # 优先查找 event，然后是 object
        if "event" in sample:
            self.input_key = "event"
        elif "object" in sample:
            self.input_key = "object"
        else:
            # 向后兼容：查找以 _ 开头的键
            for key in sample.keys():
                if key.startswith("_") and key != "_label_":
                    self.input_key = key
                    break

        if self.input_key is None:
            raise ValueError(f"Could not find input key in train_loader. Available keys: {list(sample.keys())}")

        # 设置 early stopping callback 的模型引用
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                callback.set_model(self.model)

    def _create_paradigm(self, learning_paradigm: str, paradigm_config: dict[str, Any] | None) -> LearningParadigm:
        """
        创建学习范式实例。

        Args:
            learning_paradigm: 学习范式名称
            paradigm_config: 范式配置

        Returns:
            LearningParadigm: 学习范式实例
        """
        if paradigm_config is None:
            paradigm_config = {}

        # 将 task_type 传递给 paradigm_config（如果范式需要）
        if "task_type" not in paradigm_config:
            paradigm_config["task_type"] = self.task_type

        if learning_paradigm == "supervised":
            return SupervisedParadigm(paradigm_config)
        elif learning_paradigm == "semi-supervised":
            return SemiSupervisedParadigm(paradigm_config)
        elif learning_paradigm == "unsupervised":
            return UnsupervisedParadigm(paradigm_config)
        else:
            raise ValueError(f"Unknown learning paradigm: {learning_paradigm}")

    def train_epoch(self) -> dict[str, float]:
        """
        训练一个 epoch。

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Callback: batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, {"batch": batch_idx})

            # 准备输入（batch 必须是字典格式）
            if not isinstance(batch, dict):
                raise ValueError(f"Batch must be a dict, got {type(batch)}")

            # 使用 paradigm 准备数据（如果需要特殊处理）
            inputs, labels = self.paradigm.prepare_batch(batch)
            inputs = inputs.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model({"features": inputs})

            # 使用 paradigm 计算损失
            loss = self.paradigm.compute_loss(
                model=self.model,
                inputs=inputs,
                labels=labels,
                outputs=outputs,
                loss_fn=self.loss_fn,
            )

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Callback: batch_end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, {"loss": loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def validate(self) -> dict[str, float]:
        """
        验证。

        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}

        return self.evaluator.evaluate(
            self.model,
            self.val_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    def test(self) -> dict[str, float]:
        """
        测试。

        Returns:
            测试指标字典
        """
        if self.test_loader is None:
            return {}

        return self.evaluator.evaluate(
            self.model,
            self.test_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    def fit(
        self,
        num_epochs: int,
        save_dir: str | None = None,
        save_best: bool = True,
        monitor: str = "val_loss",
    ) -> dict[str, Any]:
        """
        训练模型。

        Args:
            num_epochs: 训练轮数
            save_dir: 保存目录
            save_best: 是否保存最佳模型
            monitor: 监控的指标（用于保存最佳模型）

        Returns:
            训练历史字典
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_value = float("inf")
        best_epoch = 0

        # Callback: train_begin（传递配置信息）
        train_begin_logs = {
            "num_epochs": num_epochs,
            "config": {
                "model": self.model.config if hasattr(self.model, "config") else {},
                "task_type": self.task_type,
                "device": str(self.device),
            },
        }
        for callback in self.callbacks:
            callback.on_train_begin(train_begin_logs)

        for epoch in range(num_epochs):
            # Callback: epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, {"epoch": epoch})

            # 训练
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])

            # 验证
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics.get("loss", 0.0))
                history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()

            # 合并指标
            epoch_logs = {**train_metrics, **val_metrics}

            # Callback: epoch_end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_logs)

            # 保存最佳模型
            if save_best and save_dir:
                current_value = epoch_logs.get(monitor, float("inf"))
                if current_value < best_value:
                    best_value = current_value
                    best_epoch = epoch
                    self.save_checkpoint(save_dir, "best_model.pt")

            # 检查早停
            should_stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStoppingCallback):
                    if callback.should_stop():
                        should_stop = True
                        break

            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Callback: train_end
        for callback in self.callbacks:
            callback.on_train_end({"history": history})

        # 保存最终模型
        if save_dir:
            self.save_checkpoint(save_dir, "final_model.pt")

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_value": best_value,
        }

    def save_checkpoint(self, save_dir: str, filename: str = "checkpoint.pt"):
        """
        保存检查点。

        Args:
            save_dir: 保存目录
            filename: 文件名
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }

        torch.save(checkpoint, save_path / filename)
        logger.info(f"Checkpoint saved to {save_path / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点。

        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
