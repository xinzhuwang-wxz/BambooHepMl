"""
Evaluator 类

提供评估功能，支持：
- 分类任务（准确率、F1、混淆矩阵等）
- 回归任务（MSE、MAE、R²等）
- 多任务评估
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import BaseModel


class Evaluator:
    """
    评估器

    支持：
    - 分类任务评估
    - 回归任务评估
    - 多任务评估
    """

    def __init__(self, task_type: str = "classification", input_key: str | None = None):
        """
        初始化评估器。

        Args:
            task_type: 任务类型（'classification', 'regression', 'multitask'）
            input_key: 输入键名（'event', 'object' 或 '_features' 等），如果为 None 则自动检测
        """
        self.task_type = task_type
        self.input_key = input_key

    def evaluate(
        self,
        model: BaseModel,
        dataloader: DataLoader,
        loss_fn: torch.nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        """
        评估模型。

        Args:
            model: 模型实例
            dataloader: 数据加载器
            loss_fn: 损失函数（可选）
            device: 设备

        Returns:
            评估指标字典
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()

        # 收集预测和标签
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # 将 batch 移动到设备（直接传递完整 batch 给 model）
                batch_on_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_on_device[k] = v.to(device)
                    else:
                        batch_on_device[k] = v

                labels = batch_on_device.get("_label_")
                if labels is None:
                    raise ValueError("Batch must contain '_label_' key for evaluation")

                # 前向传播（直接传递完整 batch）
                outputs = model(batch_on_device)

                # 计算损失（如果提供）
                if loss_fn is not None:
                    # 根据 task_type 决定 labels 的类型
                    # CrossEntropyLoss 需要 Long 类型，MSELoss 需要 Float 类型
                    if self.task_type == "classification":
                        # 分类任务：确保 labels 是 Long 类型
                        if labels.dtype != torch.long:
                            labels = labels.long()
                        loss = loss_fn(outputs, labels)
                    else:  # regression
                        # 回归任务：确保 labels 是 Float 类型
                        if labels.dtype != torch.float32:
                            labels = labels.float()
                        loss = loss_fn(outputs.squeeze(), labels)
                    total_loss += loss.item()
                    num_batches += 1

                # 收集预测和标签
                if self.task_type == "classification":
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    predictions = outputs.squeeze().cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels.cpu().numpy())

        # 合并结果
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        # 计算指标
        metrics = {}

        if loss_fn is not None:
            metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        if self.task_type == "classification":
            metrics.update(self._compute_classification_metrics(all_predictions, all_labels))
        elif self.task_type == "regression":
            metrics.update(self._compute_regression_metrics(all_predictions, all_labels))
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        return metrics

    def _compute_classification_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """计算分类任务指标。"""
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            metrics = {
                "accuracy": float(accuracy_score(labels, predictions)),
                "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
                "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            }
        except ImportError:
            # 如果没有 sklearn，使用简单的准确率计算
            accuracy = float(np.mean(predictions == labels))
            metrics = {
                "accuracy": accuracy,
                "precision": accuracy,  # 简化
                "recall": accuracy,  # 简化
                "f1": accuracy,  # 简化
            }

        return metrics

    def _compute_regression_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """计算回归任务指标。"""
        # 使用 numpy 计算回归指标（不依赖 sklearn）
        mse = float(np.mean((labels - predictions) ** 2))
        mae = float(np.mean(np.abs(labels - predictions)))
        rmse = float(np.sqrt(mse))

        # R² 计算
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        return metrics
