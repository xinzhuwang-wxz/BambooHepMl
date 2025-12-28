"""
统一 Trainer

支持：
- Supervised / Semi-supervised / Unsupervised
- Loss 与 Model 解耦
- Callback 系统（logging / early stop）
- 多任务 Loss
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import logger
from ..models import BaseModel
from .callbacks import Callback, EarlyStoppingCallback
from .evaluator import Evaluator


class Trainer:
    """
    统一训练器

    支持：
    - Supervised: 有监督学习
    - Semi-supervised: 半监督学习
    - Unsupervised: 无监督学习
    - Loss 与 Model 解耦
    - Callback 系统
    - 多任务 Loss
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Union[nn.Module, Callable]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        task_type: str = "supervised",
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        初始化训练器。

        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            test_loader: 测试数据加载器（可选）
            loss_fn: 损失函数（可选，如果为 None，需要从 task_type 推断）
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            device: 设备
            task_type: 任务类型（'supervised', 'semi-supervised', 'unsupervised'）
            callbacks: 回调函数列表
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_type = task_type
        self.callbacks = callbacks or []

        # 设置模型到设备
        self.model.to(self.device)

        # 设置损失函数
        if loss_fn is None:
            self.loss_fn = self._get_default_loss_fn()
        else:
            self.loss_fn = loss_fn

        # 设置优化器
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        # 设置调度器
        self.scheduler = scheduler

        # 设置评估器
        self.evaluator = Evaluator(task_type="classification" if task_type == "supervised" else "regression")

        # 找到输入键
        sample = next(iter(train_loader))
        self.input_key = None
        for key in sample.keys():
            if key.startswith("_") and key != "_label_":
                self.input_key = key
                break

        if self.input_key is None:
            raise ValueError("Could not find input key in train_loader")

        # 设置 early stopping callback 的模型引用
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                callback.set_model(self.model)

    def _get_default_loss_fn(self) -> nn.Module:
        """获取默认损失函数。"""
        if self.task_type == "supervised":
            # 从数据推断是分类还是回归
            sample = next(iter(self.train_loader))
            labels = sample["_label_"]
            if labels.dtype == torch.long:
                return nn.CrossEntropyLoss()
            else:
                return nn.MSELoss()
        elif self.task_type == "semi-supervised":
            # 半监督：组合有监督和无监督损失
            return nn.MSELoss()  # 简化，实际应该组合
        elif self.task_type == "unsupervised":
            # 无监督：重构损失
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def train_epoch(self) -> Dict[str, float]:
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

            # 准备输入
            inputs = batch[self.input_key].to(self.device)
            labels = batch.get("_label_", None)
            if labels is not None:
                labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model({"features": inputs})

            # 计算损失
            if self.task_type == "supervised" and labels is not None:
                if outputs.shape[1] > 1:  # 分类
                    loss = self.loss_fn(outputs, labels)
                else:  # 回归
                    loss = self.loss_fn(outputs.squeeze(), labels.float())
            elif self.task_type == "semi-supervised":
                # 半监督：组合有监督和无监督损失
                if labels is not None:
                    supervised_loss = self.loss_fn(outputs.squeeze(), labels.float())
                else:
                    supervised_loss = torch.tensor(0.0, device=self.device)
                # 无监督损失（简化：重构误差）
                unsupervised_loss = torch.tensor(0.0, device=self.device)
                loss = supervised_loss + 0.1 * unsupervised_loss  # 权重可配置
            elif self.task_type == "unsupervised":
                # 无监督：重构损失
                loss = self.loss_fn(outputs, inputs)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

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

    def validate(self) -> Dict[str, float]:
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

    def test(self) -> Dict[str, float]:
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
        save_dir: Optional[str] = None,
        save_best: bool = True,
        monitor: str = "val_loss",
    ) -> Dict[str, Any]:
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
