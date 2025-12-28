"""
Pipeline State

管理 pipeline 的完整状态，包括配置验证和一致性检查。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import logger
from ..data.features import FeatureGraph


@dataclass
class PipelineState:
    """
    Pipeline 状态

    包含 pipeline 的完整状态信息，用于：
    - 配置一致性验证
    - 状态持久化
    - 训练/推理/导出时使用同一状态
    """

    feature_spec: dict[str, Any]
    model_config: dict[str, Any]
    task_type: str
    input_dim: int
    input_key: str
    pipeline_config_path: str | None = None
    feature_config_path: str | None = None
    data_config_path: str | None = None

    def validate(self) -> tuple[bool, list[str]]:
        """
        验证配置一致性。

        Returns:
            tuple[bool, list[str]]: (是否有效, 错误列表)
        """
        errors = []

        # 验证 feature_spec
        if not self.feature_spec:
            errors.append("feature_spec is empty")
        else:
            if "event" not in self.feature_spec and "object" not in self.feature_spec:
                errors.append("feature_spec must contain at least 'event' or 'object' features")

        # 验证 model_config
        if not self.model_config:
            errors.append("model_config is empty")
        else:
            if "name" not in self.model_config:
                errors.append("model_config must contain 'name'")
            if "params" not in self.model_config:
                errors.append("model_config must contain 'params'")

        # 验证 task_type
        valid_task_types = ["classification", "regression", "multitask"]
        if self.task_type not in valid_task_types:
            errors.append(f"task_type must be one of {valid_task_types}, got {self.task_type}")

        # 验证 input_dim
        if self.input_dim <= 0:
            errors.append(f"input_dim must be positive, got {self.input_dim}")

        # 验证 input_key
        valid_input_keys = ["event", "object"]
        if self.input_key not in valid_input_keys:
            errors.append(f"input_key must be one of {valid_input_keys}, got {self.input_key}")

        # 验证 input_dim 与 feature_spec 的一致性
        if "event" in self.feature_spec:
            expected_dim = self.feature_spec["event"]["dim"]
            if self.input_key == "event" and self.input_dim != expected_dim:
                errors.append(f"input_dim mismatch: input_dim={self.input_dim}, " f"but feature_spec['event']['dim']={expected_dim}")
        elif "object" in self.feature_spec:
            if self.input_key == "object":
                object_dim = self.feature_spec["object"]["dim"]
                max_length = self.feature_spec["object"]["max_length"]
                expected_dim = object_dim * max_length
                if self.input_dim != expected_dim:
                    errors.append(
                        f"input_dim mismatch: input_dim={self.input_dim}, "
                        f"but feature_spec['object'] suggests {expected_dim} "
                        f"(dim={object_dim}, max_length={max_length})"
                    )

        # 验证 model_config 中的 input_dim（如果存在）
        if "params" in self.model_config:
            model_params = self.model_config["params"]
            if "input_dim" in model_params:
                if model_params["input_dim"] != self.input_dim:
                    errors.append(
                        f"input_dim mismatch: state.input_dim={self.input_dim}, " f"but model_config.params.input_dim={model_params['input_dim']}"
                    )

        # 验证 task_type 与 model 的一致性
        if "params" in self.model_config:
            model_params = self.model_config["params"]
            if "task_type" in model_params:
                if model_params["task_type"] != self.task_type:
                    logger.warning(
                        f"task_type mismatch: state.task_type={self.task_type}, " f"but model_config.params.task_type={model_params['task_type']}"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def save(self, path: str | Path) -> None:
        """
        保存状态到文件。

        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = asdict(self)

        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

        logger.info(f"Pipeline state saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PipelineState:
        """
        从文件加载状态。

        Args:
            path: 文件路径

        Returns:
            PipelineState: 加载的状态
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline state file not found: {path}")

        with open(path) as f:
            state_dict = json.load(f)

        return cls(**state_dict)

    @classmethod
    def from_configs(
        cls,
        feature_graph: FeatureGraph,
        model_config: dict[str, Any],
        task_type: str,
        pipeline_config_path: str | None = None,
        feature_config_path: str | None = None,
        data_config_path: str | None = None,
    ) -> PipelineState:
        """
        从配置创建 PipelineState。

        Args:
            feature_graph: 特征图
            model_config: 模型配置
            task_type: 任务类型
            pipeline_config_path: Pipeline 配置文件路径
            feature_config_path: 特征配置文件路径
            data_config_path: 数据配置文件路径

        Returns:
            PipelineState: 创建的状态
        """
        # 获取 feature_spec
        feature_spec = feature_graph.output_spec()

        # 获取 input_dim 和 input_key
        if "event" in feature_spec:
            input_dim = feature_spec["event"]["dim"]
            input_key = "event"
        elif "object" in feature_spec:
            object_dim = feature_spec["object"]["dim"]
            max_length = feature_spec["object"]["max_length"]
            input_dim = object_dim * max_length
            input_key = "object"
        else:
            raise ValueError("No features found in feature_spec")

        return cls(
            feature_spec=feature_spec,
            model_config=model_config,
            task_type=task_type,
            input_dim=input_dim,
            input_key=input_key,
            pipeline_config_path=pipeline_config_path,
            feature_config_path=feature_config_path,
            data_config_path=data_config_path,
        )

    def get_model_info(self) -> dict[str, Any]:
        """获取模型信息摘要。"""
        return {
            "model_name": self.model_config.get("name", "unknown"),
            "task_type": self.task_type,
            "input_dim": self.input_dim,
            "input_key": self.input_key,
            "feature_types": list(self.feature_spec.keys()),
        }
