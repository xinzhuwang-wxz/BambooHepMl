"""
无监督学习范式

支持多种无监督学习方法：
- Autoencoder: 自编码器重构
- Variational Autoencoder (VAE): 变分自编码器
- Contrastive Learning: 对比学习
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import logger
from .base import LearningParadigm


class UnsupervisedParadigm(LearningParadigm):
    """
    无监督学习范式

    支持多种方法：
    - autoencoder: 自编码器重构损失
    - vae: 变分自编码器（需要模型支持）
    - contrastive: 对比学习（需要模型支持）
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        初始化无监督学习范式。

        Args:
            config: 配置字典，可包含：
                - method: "autoencoder", "vae", "contrastive" (默认: "autoencoder")
                - reconstruction_weight: 重构损失的权重（默认: 1.0）
                - kl_weight: KL 散度权重（仅用于 VAE，默认: 0.001）
        """
        super().__init__(config)
        self.method = config.get("method", "autoencoder") if config else "autoencoder"
        self.reconstruction_weight = config.get("reconstruction_weight", 1.0) if config else 1.0
        self.kl_weight = config.get("kl_weight", 0.001) if config else 0.001

        valid_methods = ["autoencoder", "vae", "contrastive"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        outputs: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        计算无监督损失。

        Args:
            model: 模型（应该是一个自编码器）
            inputs: 输入张量
            labels: 标签（未使用，但保留接口一致性）
            outputs: 模型输出（重构的输出）
            loss_fn: 损失函数（如果为 None，使用 MSE）

        Returns:
            torch.Tensor: 无监督损失
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        if self.method == "autoencoder":
            loss = self._autoencoder_loss(inputs, outputs, loss_fn)
        elif self.method == "vae":
            loss = self._vae_loss(model, inputs, outputs, loss_fn)
        elif self.method == "contrastive":
            loss = self._contrastive_loss(model, inputs, outputs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return loss

    def _autoencoder_loss(self, inputs: torch.Tensor, outputs: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        自编码器重构损失。

        Args:
            inputs: 原始输入
            outputs: 重构输出
            loss_fn: 损失函数

        Returns:
            torch.Tensor: 重构损失
        """
        # 对于自编码器，输出应该与输入形状相同
        # 如果 outputs 形状不同，可能需要 reshape
        if outputs.shape != inputs.shape:
            # 尝试展平后比较
            if outputs.numel() == inputs.numel():
                outputs = outputs.view(inputs.shape)
            else:
                logger.warning(f"Output shape {outputs.shape} != input shape {inputs.shape}. " "Using flatten comparison.")
                outputs_flat = outputs.flatten()
                inputs_flat = inputs.flatten()
                # 截断或填充到相同长度
                min_len = min(len(outputs_flat), len(inputs_flat))
                outputs_flat = outputs_flat[:min_len]
                inputs_flat = inputs_flat[:min_len]
                return loss_fn(outputs_flat, inputs_flat) * self.reconstruction_weight

        return loss_fn(outputs, inputs) * self.reconstruction_weight

    def _vae_loss(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        变分自编码器损失（重构损失 + KL 散度）。

        Args:
            model: VAE 模型（需要支持返回 mu, logvar）
            inputs: 原始输入
            outputs: 重构输出
            loss_fn: 损失函数

        Returns:
            torch.Tensor: VAE 损失
        """
        # 重构损失
        reconstruction_loss = self._autoencoder_loss(inputs, outputs, loss_fn) / self.reconstruction_weight

        # KL 散度（需要模型支持返回 mu 和 logvar）
        # 这里假设模型有 get_latent_params 方法
        if hasattr(model, "get_latent_params"):
            mu, logvar = model.get_latent_params()
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        else:
            logger.warning(
                "Model does not support get_latent_params(). "
                "Using autoencoder loss only. "
                "For VAE, model should return (mu, logvar) from encoder."
            )
            kl_loss = torch.tensor(0.0, device=inputs.device)

        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        return total_loss

    def _contrastive_loss(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        对比学习损失（SimCLR 风格）。

        Args:
            model: 模型
            inputs: 原始输入
            outputs: 模型输出（特征表示）

        Returns:
            torch.Tensor: 对比损失
        """
        # 生成增强版本（添加噪声）
        noise = torch.randn_like(inputs) * 0.1
        augmented_inputs = inputs + noise
        augmented_outputs = model({"features": augmented_inputs})

        # 归一化特征
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        augmented_norm = F.normalize(augmented_outputs, p=2, dim=1)

        # 计算相似度矩阵（温度参数）
        temperature = 0.1
        similarity_matrix = torch.matmul(outputs_norm, augmented_norm.t()) / temperature

        # 对角线元素是正样本对
        labels = torch.arange(outputs_norm.size(0), device=outputs.device)

        # 对比损失（交叉熵）
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(similarity_matrix, labels)

        return loss
