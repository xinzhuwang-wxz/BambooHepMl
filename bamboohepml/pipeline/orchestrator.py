"""
from __future__ import annotations
Pipeline Orchestrator

统一入口，协调整个 ML pipeline。
"""

from pathlib import Path
from typing import Any

import yaml

from ..config import logger
from ..data import DataConfig, DataSourceFactory, HEPDataset
from ..data.features import ExpressionEngine, FeatureGraph
from ..models import get_model


class PipelineOrchestrator:
    """
    Pipeline Orchestrator

    功能：
    - 加载 pipeline.yaml
    - 解析 data / feature / model / train 配置
    - 构建 dataset
    - 构建 model
    - 提供统一入口
    """

    def __init__(self, pipeline_config_path: str):
        """
        初始化 Pipeline Orchestrator。

        Args:
            pipeline_config_path: pipeline.yaml 文件路径
        """
        self.pipeline_config_path = Path(pipeline_config_path)
        self.config = self._load_config()
        self.data_config = None
        self.model_config = None
        self.train_config = None
        self.feature_graph = None
        self.expression_engine = None

    def _load_config(self) -> dict[str, Any]:
        """加载 pipeline.yaml 配置。"""
        if not self.pipeline_config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {self.pipeline_config_path}")

        with open(self.pipeline_config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded pipeline config from {self.pipeline_config_path}")
        return config

    def setup_data(self) -> HEPDataset:
        """
        设置数据系统。

        Returns:
            HEPDataset: 配置好的数据集
        """
        # 加载 DataConfig
        data_config_path = self.config.get("data", {}).get("config_path")
        if data_config_path:
            self.data_config = DataConfig.load(data_config_path)
        else:
            # 从 pipeline.yaml 直接读取 data 配置
            data_dict = self.config.get("data", {})
            self.data_config = DataConfig(**data_dict)

        # 创建数据源
        data_source_path = self.config.get("data", {}).get("source_path")
        if not data_source_path:
            raise ValueError("data.source_path must be specified in pipeline.yaml")

        treename = self.config.get("data", {}).get("treename", "tree")
        load_range = self.config.get("data", {}).get("load_range")

        data_source = DataSourceFactory.create(
            data_source_path,
            treename=treename,
            load_range=load_range,
        )

        # 设置特征系统（可选）
        feature_config_path = self.config.get("features", {}).get("config_path")
        if feature_config_path:
            self.expression_engine = ExpressionEngine()
            self.feature_graph = FeatureGraph.from_yaml(
                feature_config_path,
                self.expression_engine,
            )

        # 创建数据集
        dataset = HEPDataset(
            data_source=data_source,
            data_config=self.data_config,
            feature_graph=self.feature_graph,
            expression_engine=self.expression_engine,
            for_training=True,
            shuffle=True,
        )

        logger.info("Data system setup complete")
        return dataset

    def setup_model(self, input_dim: int | None = None) -> Any:
        """
        设置模型系统。

        Args:
            input_dim: 输入维度（如果为 None，将从数据中推断）

        Returns:
            模型实例
        """
        model_config = self.config.get("model", {})
        if not model_config:
            raise ValueError("model config must be specified in pipeline.yaml")

        model_name = model_config.get("name")
        if not model_name:
            raise ValueError("model.name must be specified in pipeline.yaml")

        # 获取模型参数
        model_kwargs = model_config.get("params", {})

        # 如果提供了 input_dim，使用它；否则从配置中获取
        if input_dim is not None:
            model_kwargs["input_dim"] = input_dim

        # 创建模型
        model = get_model(model_name, **model_kwargs)

        self.model_config = model_config
        logger.info(f"Model '{model_name}' created with params: {model_kwargs}")
        return model

    def get_train_config(self) -> dict[str, Any]:
        """
        获取训练配置。

        Returns:
            训练配置字典
        """
        train_config = self.config.get("train", {})
        self.train_config = train_config
        return train_config

    def get_config(self) -> dict[str, Any]:
        """
        获取完整配置。

        Returns:
            完整配置字典
        """
        return self.config

    def get_data_config(self) -> DataConfig | None:
        """获取 DataConfig。"""
        return self.data_config

    def get_model_config(self) -> dict[str, Any] | None:
        """获取模型配置。"""
        return self.model_config

    def get_feature_graph(self) -> FeatureGraph | None:
        """获取特征图。"""
        return self.feature_graph
