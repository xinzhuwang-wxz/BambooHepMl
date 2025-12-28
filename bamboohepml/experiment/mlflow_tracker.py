"""
MLflow 跟踪器

提供 MLflow 专用的实验跟踪功能。
"""

from typing import Any, Optional

from ..config import MLFLOW_TRACKING_URI, logger


class MLflowTracker:
    """
    MLflow 跟踪器

    提供 MLflow 专用的功能：
    - 搜索最佳 run
    - 获取 run 信息
    - 下载 artifacts
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        初始化 MLflow 跟踪器。

        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self.mlflow = None

        try:
            import mlflow

            self.mlflow = mlflow
            if self.tracking_uri:
                self.mlflow.set_tracking_uri(self.tracking_uri)
        except ImportError:
            logger.warning("MLflow not available")

    def search_runs(
        self,
        experiment_name: str,
        filter_string: Optional[str] = None,
        order_by: Optional[list] = None,
        max_results: int = 100,
    ):
        """
        搜索 runs。

        Args:
            experiment_name: 实验名称
            filter_string: 过滤字符串
            order_by: 排序字段列表（如 ["metrics.val_loss ASC"]）
            max_results: 最大结果数

        Returns:
            DataFrame: 搜索结果
        """
        if self.mlflow is None:
            raise ImportError("MLflow is not installed")

        return self.mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results,
        )

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "val_loss",
        mode: str = "min",
    ) -> Optional[dict[str, Any]]:
        """
        获取最佳 run。

        Args:
            experiment_name: 实验名称
            metric: 指标名称
            mode: "min" 或 "max"

        Returns:
            最佳 run 的信息字典
        """
        if self.mlflow is None:
            return None

        try:
            order_by = [f"metrics.{metric} {mode.upper()}"]
            runs = self.search_runs(experiment_name, order_by=order_by, max_results=1)

            if len(runs) == 0:
                return None

            best_run = runs.iloc[0]
            return {
                "run_id": best_run.run_id,
                "experiment_id": best_run.experiment_id,
                "metrics": {k.replace("metrics.", ""): v for k, v in best_run.items() if k.startswith("metrics.")},
                "params": {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")},
                "artifact_uri": best_run.artifact_uri,
            }
        except Exception as e:
            logger.warning(f"Failed to get best run: {e}")
            return None

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """
        获取指定 run 的信息。

        Args:
            run_id: Run ID

        Returns:
            Run 信息字典
        """
        if self.mlflow is None:
            return None

        try:
            run = self.mlflow.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        except Exception as e:
            logger.warning(f"Failed to get run: {e}")
            return None

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str):
        """
        下载 artifact。

        Args:
            run_id: Run ID
            artifact_path: Artifact 路径（在 MLflow 中）
            dst_path: 目标路径
        """
        if self.mlflow is None:
            raise ImportError("MLflow is not installed")

        return self.mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path,
        )
