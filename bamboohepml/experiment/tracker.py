"""
统一实验跟踪器

提供统一的接口来管理 MLflow 和 TensorBoard。
"""

from __future__ import annotations

from typing import Any

from ..engine.callbacks import MLflowCallback, TensorBoardCallback


class ExperimentTracker:
    """
    统一实验跟踪器

    自动管理：
    - MLflow 跟踪
    - TensorBoard 日志
    - Config 记录
    - Metrics 记录
    - Artifacts 保存
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        use_mlflow: bool = True,
        use_tensorboard: bool = True,
        mlflow_tracking_uri: str | None = None,
        tensorboard_log_dir: str | None = None,
        log_config: bool = True,
        log_artifacts: bool = True,
    ):
        """
        初始化实验跟踪器。

        Args:
            experiment_name: 实验名称
            use_mlflow: 是否使用 MLflow
            use_tensorboard: 是否使用 TensorBoard
            mlflow_tracking_uri: MLflow tracking URI
            tensorboard_log_dir: TensorBoard 日志目录
            log_config: 是否自动记录配置
            log_artifacts: 是否自动保存 artifacts
        """
        self.experiment_name = experiment_name
        self.log_config = log_config
        self.log_artifacts = log_artifacts

        # 初始化 Callbacks
        self.callbacks = []

        if use_mlflow:
            self.mlflow_callback = MLflowCallback(
                experiment_name=experiment_name,
                tracking_uri=mlflow_tracking_uri,
                log_config=log_config,
                log_artifacts=log_artifacts,
            )
            self.callbacks.append(self.mlflow_callback)
        else:
            self.mlflow_callback = None

        if use_tensorboard:
            self.tensorboard_callback = TensorBoardCallback(
                log_dir=tensorboard_log_dir or "./logs/tensorboard",
                log_config=log_config,
            )
            self.callbacks.append(self.tensorboard_callback)
        else:
            self.tensorboard_callback = None

    def start_run(self, config: dict[str, Any] | None = None, model=None):
        """
        开始实验 run。

        Args:
            config: 配置字典
            model: 模型实例（用于 TensorBoard 模型图）
        """
        logs = {"config": config or {}}
        if model is not None:
            logs["sample_input"] = self._get_sample_input(model)
            if self.tensorboard_callback:
                self.tensorboard_callback.set_model(model)

        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def log_metrics(self, metrics: dict[str, float], step: int):
        """
        记录指标。

        Args:
            metrics: 指标字典
            step: 步骤（epoch）
        """
        for callback in self.callbacks:
            callback.on_epoch_end(step, metrics)

    def log_artifact(self, artifact_path: str, artifact_path_in_mlflow: str | None = None):
        """
        记录 artifact。

        Args:
            artifact_path: artifact 路径
            artifact_path_in_mlflow: MLflow 中的 artifact 路径
        """
        if self.mlflow_callback:
            self.mlflow_callback.log_artifact(artifact_path, artifact_path_in_mlflow)

    def end_run(self, logs: dict[str, Any] | None = None):
        """
        结束实验 run。

        Args:
            logs: 额外的日志信息（如 model_path, config_path）
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def _get_sample_input(self, model) -> Any:
        """获取示例输入（用于模型图）。"""
        # 简化实现：返回 None，实际应该从模型配置推断
        return None

    def get_callbacks(self) -> list:
        """获取所有 callbacks（用于 Trainer）。"""
        return self.callbacks
