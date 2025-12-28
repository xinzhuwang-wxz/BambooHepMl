"""
MLP 模型

提供：
- MLPClassifier: 分类任务
- MLPRegressor: 回归任务
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from ..base import ClassificationModel, RegressionModel
from ..registry import register_model


@register_model("mlp_classifier")
class MLPClassifier(ClassificationModel):
    """
    多层感知机分类器

    示例：
        model = MLPClassifier(
            input_dim=128,
            hidden_dims=[256, 128, 64],
            num_classes=10,
            dropout=0.1,
            activation='relu'
        )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = False,
        **kwargs,
    ):
        """
        初始化 MLP 分类器。

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
            dropout: Dropout 比例
            activation: 激活函数（'relu', 'gelu', 'tanh'）
            batch_norm: 是否使用 BatchNorm
            **kwargs: 其他参数
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            **kwargs,
        )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 分类头
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: 输入批次字典，必须包含 'features' 键
                - features: (batch_size, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        if "features" not in batch:
            raise ValueError("Batch must contain 'features' key")

        x = batch["features"]
        return self.network(x)

    @classmethod
    def load(cls, save_dir, model_name: str = "model", **kwargs):
        """
        加载模型。

        Args:
            save_dir: 保存目录
            model_name: 模型文件名（不含扩展名）
            **kwargs: 额外的加载参数

        Returns:
            加载的模型实例
        """
        save_dir = Path(save_dir)
        config_path = save_dir / f"{model_name}_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建模型实例
        model = cls(**config)

        # 加载权重
        weights_path = save_dir / f"{model_name}.pt"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        # 加载冻结层信息
        frozen_path = save_dir / f"{model_name}_frozen.json"
        if frozen_path.exists():
            with open(frozen_path, "r") as f:
                frozen_layers = json.load(f)
            model._frozen_layers = set(frozen_layers)
            # 恢复冻结状态
            for name, param in model.named_parameters():
                if name in frozen_layers:
                    param.requires_grad = False

        return model


@register_model("mlp_regressor")
class MLPRegressor(RegressionModel):
    """
    多层感知机回归器

    示例：
        model = MLPRegressor(
            input_dim=128,
            hidden_dims=[256, 128, 64],
            num_outputs=1,
            dropout=0.1,
            activation='relu'
        )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_outputs: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = False,
        **kwargs,
    ):
        """
        初始化 MLP 回归器。

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_outputs: 输出数量（默认为 1）
            dropout: Dropout 比例
            activation: 激活函数（'relu', 'gelu', 'tanh'）
            batch_norm: 是否使用 BatchNorm
            **kwargs: 其他参数
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_outputs=num_outputs,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            **kwargs,
        )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 回归头（无激活函数）
        layers.append(nn.Linear(prev_dim, num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: 输入批次字典，必须包含 'features' 键
                - features: (batch_size, input_dim)

        Returns:
            预测值: (batch_size, num_outputs)
        """
        if "features" not in batch:
            raise ValueError("Batch must contain 'features' key")

        x = batch["features"]
        return self.network(x)

    @classmethod
    def load(cls, save_dir, model_name: str = "model", **kwargs):
        """
        加载模型。

        Args:
            save_dir: 保存目录
            model_name: 模型文件名（不含扩展名）
            **kwargs: 额外的加载参数

        Returns:
            加载的模型实例
        """
        save_dir = Path(save_dir)
        config_path = save_dir / f"{model_name}_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建模型实例
        model = cls(**config)

        # 加载权重
        weights_path = save_dir / f"{model_name}.pt"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        # 加载冻结层信息
        frozen_path = save_dir / f"{model_name}_frozen.json"
        if frozen_path.exists():
            with open(frozen_path, "r") as f:
                frozen_layers = json.load(f)
            model._frozen_layers = set(frozen_layers)
            # 恢复冻结状态
            for name, param in model.named_parameters():
                if name in frozen_layers:
                    param.requires_grad = False

        return model
