"""
调度器基类

定义调度器接口。
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseScheduler(ABC):
    """
    调度器基类

    所有调度器都应该继承此类，实现：
    - submit_train: 提交训练任务
    - submit_predict: 提交预测任务
    - submit_export: 提交导出任务
    - submit_inspect: 提交检查任务
    """

    @abstractmethod
    def submit_train(
        self,
        pipeline_config_path: str,
        experiment_name: Optional[str] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        output_dir: Optional[str] = None,
        use_ray: bool = False,
        num_workers: int = 1,
        gpu_per_worker: int = 0,
        **kwargs
    ) -> Any:
        """提交训练任务。"""
        pass

    @abstractmethod
    def submit_predict(
        self,
        pipeline_config_path: str,
        model_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 32,
        return_probabilities: bool = False,
        **kwargs
    ) -> Any:
        """提交预测任务。"""
        pass

    @abstractmethod
    def submit_export(
        self,
        pipeline_config_path: str,
        model_path: str,
        output_path: str,
        input_shape: Optional[tuple] = None,
        opset_version: int = 11,
        **kwargs
    ) -> Any:
        """提交导出任务。"""
        pass

    @abstractmethod
    def submit_inspect(
        self,
        pipeline_config_path: str,
        output_path: Optional[str] = None,
        num_samples: int = 1000,
        inspect_data: bool = True,
        inspect_features: bool = True,
        **kwargs
    ) -> Any:
        """提交检查任务。"""
        pass
