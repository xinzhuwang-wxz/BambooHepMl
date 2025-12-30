"""
MLP 模型

提供：
- MLPClassifier: 分类任务
- MLPRegressor: 回归任务
"""

import json
from pathlib import Path
from typing import Optional

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
        hidden_dims: list[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = False,
        event_input_dim: Optional[int] = None,
        object_input_dim: Optional[int] = None,
        embed_dim: int = 64,
        object_pooling_mode: str = "mean",
        **kwargs,
    ):
        """
        初始化 MLP 分类器。

        Args:
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
            dropout: Dropout 比例
            activation: 激活函数（'relu', 'gelu', 'tanh'）
            batch_norm: 是否使用 BatchNorm
            event_input_dim: event-level 特征维度（如果为 None，则不使用 event 特征）
            object_input_dim: object-level 特征维度（如果为 None，则不使用 object 特征）
            embed_dim: embedding 维度（event 和 object 嵌入后的统一维度，默认 64）
            object_pooling_mode: object-level 特征的池化方式（'mean', 'sum', 'max'）
            **kwargs: 其他参数

        注意：
            - 必须至少提供一个 event_input_dim 或 object_input_dim
            - batch 应包含 "event" 或 "object" 键（根据提供的维度）
        """
        # 确保至少有一个输入维度
        if event_input_dim is None and object_input_dim is None:
            raise ValueError("At least one of event_input_dim or object_input_dim must be provided")

        # 计算融合后的维度（用于 MLP 主干输入）
        fused_dim = 0
        if event_input_dim is not None:
            fused_dim += embed_dim
        if object_input_dim is not None:
            fused_dim += embed_dim
        # 使用融合维度作为 MLP 主干的输入维度
        mlp_input_dim = fused_dim

        super().__init__(
            input_dim=mlp_input_dim,  # 传递给父类的 input_dim 是融合后的维度
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            **kwargs,
        )

        # 保存所有参数
        self.event_input_dim = event_input_dim
        self.object_input_dim = object_input_dim
        self.embed_dim = embed_dim
        self.object_pooling_mode = object_pooling_mode
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # 构建 embedding 网络
        # Event-level embedding
        if event_input_dim is not None:
            event_embed_layers = [nn.Linear(event_input_dim, embed_dim)]
            if activation == "relu":
                event_embed_layers.append(nn.ReLU())
            elif activation == "gelu":
                event_embed_layers.append(nn.GELU())
            elif activation == "tanh":
                event_embed_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            self.event_embed = nn.Sequential(*event_embed_layers)
        else:
            self.event_embed = None

        # Object-level embedding
        if object_input_dim is not None:
            object_embed_layers = [nn.Linear(object_input_dim, embed_dim)]
            if activation == "relu":
                object_embed_layers.append(nn.ReLU())
            elif activation == "gelu":
                object_embed_layers.append(nn.GELU())
            elif activation == "tanh":
                object_embed_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            self.object_embed = nn.Sequential(*object_embed_layers)
        else:
            self.object_embed = None

        # 构建 MLP 主干网络
        layers = []
        prev_dim = mlp_input_dim

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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: 输入批次字典，包含以下键：
                - event: (batch_size, event_input_dim) [可选，如果 event_input_dim 已提供]
                - object: (batch_size, max_length, object_input_dim) [可选，如果 object_input_dim 已提供]
                - mask: (batch_size, max_length) [可选，仅在使用 object 时需要]

        Returns:
            logits: (batch_size, num_classes)
        """
        embeddings = []

        # Event-level embedding
        if self.event_embed is not None:
            if "event" not in batch:
                raise ValueError("Batch must contain 'event' key when event_input_dim is provided")
            event_features = batch["event"]  # (B, event_input_dim)
            event_emb = self.event_embed(event_features)  # (B, embed_dim)
            embeddings.append(event_emb)

        # Object-level embedding
        if self.object_embed is not None:
            if "object" not in batch:
                raise ValueError("Batch must contain 'object' key when object_input_dim is provided")
            object_features = batch["object"]  # (B, N, object_input_dim)
            object_emb = self.object_embed(object_features)  # (B, N, embed_dim)

            # Pooling: 将 object-level 特征池化为 event-level
            if self.object_pooling_mode == "mean":
                # 如果有 mask，使用 masked mean；否则使用普通 mean
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    object_emb_masked = object_emb * mask_expanded
                    object_emb_pooled = object_emb_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.mean(dim=1)  # (B, embed_dim)
            elif self.object_pooling_mode == "sum":
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    object_emb_pooled = (object_emb * mask_expanded).sum(dim=1)  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.sum(dim=1)  # (B, embed_dim)
            elif self.object_pooling_mode == "max":
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    # 将 masked 位置设为负无穷，这样 max 会忽略它们
                    object_emb_masked = object_emb.masked_fill(~mask_expanded.bool(), float("-inf"))
                    object_emb_pooled = object_emb_masked.max(dim=1)[0]  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.max(dim=1)[0]  # (B, embed_dim)
            else:
                raise ValueError(f"Unknown pooling mode: {self.object_pooling_mode}")

            embeddings.append(object_emb_pooled)

        # 融合 embeddings（concat）
        if len(embeddings) == 0:
            raise ValueError("At least one of event_input_dim or object_input_dim must be provided")
        elif len(embeddings) == 1:
            x = embeddings[0]
        else:
            x = torch.cat(embeddings, dim=-1)  # (B, embed_dim * num_embeddings)

        # 通过 MLP 主干网络
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

        with open(config_path) as f:
            config = json.load(f)

        # 创建模型实例
        model = cls(**config)

        # 加载权重
        weights_path = save_dir / f"{model_name}.pt"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        # 加载冻结层信息
        frozen_path = save_dir / f"{model_name}_frozen.json"
        if frozen_path.exists():
            with open(frozen_path) as f:
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
            event_input_dim=10,
            object_input_dim=5,
            embed_dim=64,
            hidden_dims=[256, 128, 64],
            num_outputs=1,
            dropout=0.1,
            activation='relu'
        )
    """

    def __init__(
        self,
        hidden_dims: list[int] = [256, 128, 64],
        num_outputs: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = False,
        event_input_dim: Optional[int] = None,
        object_input_dim: Optional[int] = None,
        embed_dim: int = 64,
        object_pooling_mode: str = "mean",
        **kwargs,
    ):
        """
        初始化 MLP 回归器。

        Args:
            hidden_dims: 隐藏层维度列表
            num_outputs: 输出数量（默认为 1）
            dropout: Dropout 比例
            activation: 激活函数（'relu', 'gelu', 'tanh'）
            batch_norm: 是否使用 BatchNorm
            event_input_dim: event-level 特征维度（如果为 None，则不使用 event 特征）
            object_input_dim: object-level 特征维度（如果为 None，则不使用 object 特征）
            embed_dim: embedding 维度（event 和 object 嵌入后的统一维度，默认 64）
            object_pooling_mode: object-level 特征的池化方式（'mean', 'sum', 'max'）
            **kwargs: 其他参数

        注意：
            - 必须至少提供一个 event_input_dim 或 object_input_dim
            - batch 应包含 "event" 或 "object" 键（根据提供的维度）
        """
        # 确保至少有一个输入维度
        if event_input_dim is None and object_input_dim is None:
            raise ValueError("At least one of event_input_dim or object_input_dim must be provided")

        # 计算融合后的维度（用于 MLP 主干输入）
        fused_dim = 0
        if event_input_dim is not None:
            fused_dim += embed_dim
        if object_input_dim is not None:
            fused_dim += embed_dim
        # 使用融合维度作为 MLP 主干的输入维度
        mlp_input_dim = fused_dim

        super().__init__(
            input_dim=mlp_input_dim,  # 传递给父类的 input_dim 是融合后的维度
            hidden_dims=hidden_dims,
            num_outputs=num_outputs,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            **kwargs,
        )

        # 保存所有参数
        self.event_input_dim = event_input_dim
        self.object_input_dim = object_input_dim
        self.embed_dim = embed_dim
        self.object_pooling_mode = object_pooling_mode
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # 构建 embedding 网络
        # Event-level embedding
        if event_input_dim is not None:
            event_embed_layers = [nn.Linear(event_input_dim, embed_dim)]
            if activation == "relu":
                event_embed_layers.append(nn.ReLU())
            elif activation == "gelu":
                event_embed_layers.append(nn.GELU())
            elif activation == "tanh":
                event_embed_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            self.event_embed = nn.Sequential(*event_embed_layers)
        else:
            self.event_embed = None

        # Object-level embedding
        if object_input_dim is not None:
            object_embed_layers = [nn.Linear(object_input_dim, embed_dim)]
            if activation == "relu":
                object_embed_layers.append(nn.ReLU())
            elif activation == "gelu":
                object_embed_layers.append(nn.GELU())
            elif activation == "tanh":
                object_embed_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            self.object_embed = nn.Sequential(*object_embed_layers)
        else:
            self.object_embed = None

        # 构建 MLP 主干网络
        layers = []
        prev_dim = mlp_input_dim

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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: 输入批次字典，包含以下键：
                - event: (batch_size, event_input_dim) [可选，如果 event_input_dim 已提供]
                - object: (batch_size, max_length, object_input_dim) [可选，如果 object_input_dim 已提供]
                - mask: (batch_size, max_length) [可选，仅在使用 object 时需要]

        Returns:
            预测值: (batch_size, num_outputs)
        """
        embeddings = []

        # Event-level embedding
        if self.event_embed is not None:
            if "event" not in batch:
                raise ValueError("Batch must contain 'event' key when event_input_dim is provided")
            event_features = batch["event"]  # (B, event_input_dim)
            event_emb = self.event_embed(event_features)  # (B, embed_dim)
            embeddings.append(event_emb)

        # Object-level embedding
        if self.object_embed is not None:
            if "object" not in batch:
                raise ValueError("Batch must contain 'object' key when object_input_dim is provided")
            object_features = batch["object"]  # (B, N, object_input_dim)
            object_emb = self.object_embed(object_features)  # (B, N, embed_dim)

            # Pooling: 将 object-level 特征池化为 event-level
            if self.object_pooling_mode == "mean":
                # 如果有 mask，使用 masked mean；否则使用普通 mean
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    object_emb_masked = object_emb * mask_expanded
                    object_emb_pooled = object_emb_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.mean(dim=1)  # (B, embed_dim)
            elif self.object_pooling_mode == "sum":
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    object_emb_pooled = (object_emb * mask_expanded).sum(dim=1)  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.sum(dim=1)  # (B, embed_dim)
            elif self.object_pooling_mode == "max":
                if "mask" in batch:
                    mask = batch["mask"]  # (B, N)
                    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                    # 将 masked 位置设为负无穷，这样 max 会忽略它们
                    object_emb_masked = object_emb.masked_fill(~mask_expanded.bool(), float("-inf"))
                    object_emb_pooled = object_emb_masked.max(dim=1)[0]  # (B, embed_dim)
                else:
                    object_emb_pooled = object_emb.max(dim=1)[0]  # (B, embed_dim)
            else:
                raise ValueError(f"Unknown pooling mode: {self.object_pooling_mode}")

            embeddings.append(object_emb_pooled)

        # 融合 embeddings（concat）
        if len(embeddings) == 0:
            raise ValueError("At least one of event_input_dim or object_input_dim must be provided")
        elif len(embeddings) == 1:
            x = embeddings[0]
        else:
            x = torch.cat(embeddings, dim=-1)  # (B, embed_dim * num_embeddings)

        # 通过 MLP 主干网络
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

        with open(config_path) as f:
            config = json.load(f)

        # 创建模型实例
        model = cls(**config)

        # 加载权重
        weights_path = save_dir / f"{model_name}.pt"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        # 加载冻结层信息
        frozen_path = save_dir / f"{model_name}_frozen.json"
        if frozen_path.exists():
            with open(frozen_path) as f:
                frozen_layers = json.load(f)
            model._frozen_layers = set(frozen_layers)
            # 恢复冻结状态
            for name, param in model.named_parameters():
                if name in frozen_layers:
                    param.requires_grad = False

        return model
