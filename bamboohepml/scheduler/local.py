"""
本地调度器

直接在本地执行任务，不使用集群调度系统。
"""

from __future__ import annotations

from typing import Any

from ..config import logger
from ..tasks import export_task, inspect_task, predict_task, train_task
from .base import BaseScheduler


class LocalScheduler(BaseScheduler):
    """
    本地调度器

    直接在本地执行任务，不使用集群调度系统。
    """

    def submit_train(
        self,
        pipeline_config_path: str,
        experiment_name: str | None = None,
        num_epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        output_dir: str | None = None,
        use_ray: bool = False,
        num_workers: int = 1,
        gpu_per_worker: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """提交训练任务（本地执行）。"""
        logger.info("Using Local Scheduler for training")

        return train_task(
            pipeline_config_path=pipeline_config_path,
            experiment_name=experiment_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            use_ray=use_ray,
            num_workers=num_workers,
            gpu_per_worker=gpu_per_worker,
        )

    def submit_predict(
        self,
        pipeline_config_path: str,
        model_path: str,
        output_path: str | None = None,
        batch_size: int = 32,
        return_probabilities: bool = False,
        **kwargs,
    ) -> list:
        """提交预测任务（本地执行）。"""
        logger.info("Using Local Scheduler for prediction")

        return predict_task(
            pipeline_config_path=pipeline_config_path,
            model_path=model_path,
            output_path=output_path,
            batch_size=batch_size,
            return_probabilities=return_probabilities,
        )

    def submit_export(
        self, pipeline_config_path: str, model_path: str, output_path: str, input_shape: tuple | None = None, opset_version: int = 11, **kwargs
    ) -> dict[str, Any]:
        """提交导出任务（本地执行）。"""
        logger.info("Using Local Scheduler for export")

        return export_task(
            pipeline_config_path=pipeline_config_path,
            model_path=model_path,
            output_path=output_path,
            input_shape=input_shape,
            opset_version=opset_version,
        )

    def submit_inspect(
        self,
        pipeline_config_path: str,
        output_path: str | None = None,
        num_samples: int = 1000,
        inspect_data: bool = True,
        inspect_features: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """提交检查任务（本地执行）。"""
        logger.info("Using Local Scheduler for inspection")

        return inspect_task(
            pipeline_config_path=pipeline_config_path,
            output_path=output_path,
            num_samples=num_samples,
            inspect_data=inspect_data,
            inspect_features=inspect_features,
        )
