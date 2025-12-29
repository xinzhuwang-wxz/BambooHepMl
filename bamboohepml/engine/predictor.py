"""
Predictor 类

提供预测功能，支持：
- 批量预测
- 概率预测
- 多任务预测
"""

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..models import BaseModel


class Predictor:
    """
    预测器

    支持：
    - 批量预测
    - 概率预测
    - 多任务预测
    """

    def __init__(self, model: BaseModel, device: torch.device | None = None, input_key: str | None = None):
        """
        初始化预测器。

        Args:
            model: 模型实例
            device: 设备
            input_key: 输入键名（'event', 'object' 或 '_features' 等），如果为 None 则自动检测
        """
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.input_key = input_key

    def predict(
        self,
        dataloader: DataLoader,
        return_probabilities: bool = False,
    ) -> list[dict[str, Any]]:
        """
        预测。

        Args:
            dataloader: 数据加载器
            return_probabilities: 是否返回概率

        Returns:
            预测结果列表
        """
        # 确定输入键
        if self.input_key is None:
            # 自动检测输入键（与 Trainer 逻辑一致）
            sample = next(iter(dataloader))
            input_key = None
            # 优先查找 event，然后是 object
            if "event" in sample:
                input_key = "event"
            elif "object" in sample:
                input_key = "object"
            else:
                # 向后兼容：查找以 _ 开头的键
                for key in sample.keys():
                    if key.startswith("_") and key != "_label_":
                        input_key = key
                        break

            if input_key is None:
                raise ValueError(f"Could not find input key in dataloader. Available keys: {list(sample.keys())}")
        else:
            input_key = self.input_key

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[input_key].to(self.device)

                # 前向传播
                outputs = self.model({"features": inputs})

                # 获取预测
                if outputs.shape[1] > 1:  # 分类任务
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                    if return_probabilities:
                        probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()
                        all_probabilities.extend(probs)
                else:  # 回归任务
                    predictions = outputs.squeeze().cpu().numpy().tolist()

                all_predictions.extend(predictions)

        # 构建结果
        results = []
        for i, pred in enumerate(all_predictions):
            result = {"prediction": pred}
            if return_probabilities and i < len(all_probabilities):
                result["probabilities"] = all_probabilities[i]
            results.append(result)

        return results

    def predict_proba(self, dataloader: DataLoader) -> list[dict[str, Any]]:
        """
        预测概率（仅分类任务）。

        Args:
            dataloader: 数据加载器

        Returns:
            预测概率列表
        """
        return self.predict(dataloader, return_probabilities=True)
