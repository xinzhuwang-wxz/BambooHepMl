"""
Predictor 类

提供预测功能，支持：
- 批量预测
- 概率预测
- 多任务预测
"""

from typing import Any, Optional

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

    def __init__(self, model: BaseModel, device: Optional[torch.device] = None):
        """
        初始化预测器。

        Args:
            model: 模型实例
            device: 设备
        """
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()

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
        # 找到输入键
        sample = next(iter(dataloader))
        input_key = None
        for key in sample.keys():
            if key.startswith("_") and key != "_label_":
                input_key = key
                break

        if input_key is None:
            raise ValueError("Could not find input key in dataloader")

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
