"""
Callback 系统

支持：
- Logging: 日志记录
- Early Stopping: 早停
- MLflow: 实验跟踪
- TensorBoard: 可视化
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from ..config import logger


class Callback(ABC):
    """
    Callback 基类
    
    所有 callback 都应该继承此类，实现：
    - on_train_begin: 训练开始时调用
    - on_train_end: 训练结束时调用
    - on_epoch_begin: 每个 epoch 开始时调用
    - on_epoch_end: 每个 epoch 结束时调用
    - on_batch_begin: 每个 batch 开始时调用
    - on_batch_end: 每个 batch 结束时调用
    """
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时调用。"""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时调用。"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 开始时调用。"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 epoch 结束时调用。"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 batch 开始时调用。"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """每个 batch 结束时调用。"""
        pass


class LoggingCallback(Callback):
    """
    日志记录 Callback
    
    记录训练过程中的指标到日志。
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        初始化日志记录 Callback。
        
        Args:
            log_level: 日志级别
        """
        self.log_level = log_level
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """记录 epoch 结束时的指标。"""
        if logs:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                    for k, v in logs.items()])
            logger.log(self.log_level, f"Epoch {epoch} - {metrics_str}")


class EarlyStoppingCallback(Callback):
    """
    早停 Callback
    
    当验证指标不再改善时停止训练。
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        restore_best_weights: bool = True,
    ):
        """
        初始化早停 Callback。
        
        Args:
            monitor: 监控的指标名称
            min_delta: 最小改善量
            patience: 容忍的 epoch 数（无改善）
            mode: 'min' 或 'max'（指标越小越好还是越大越好）
            restore_best_weights: 是否恢复最佳权重
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.model = None
    
    def set_model(self, model):
        """设置模型（用于恢复权重）。"""
        self.model = model
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """初始化。"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        if self.model is not None:
            self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """检查是否需要早停。"""
        if logs is None or self.monitor not in logs:
            return
        
        current_value = logs[self.monitor]
        
        # 判断是否改善
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights and self.model is not None:
                # 保存最佳权重
                import torch
                self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else:
            self.wait += 1
        
        # 检查是否需要停止
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"Early stopping triggered at epoch {epoch}")
            if self.restore_best_weights and self.model is not None and self.best_weights is not None:
                logger.info("Restoring best weights")
                self.model.load_state_dict(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时恢复最佳权重。"""
        if self.stopped_epoch > 0:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
    
    def should_stop(self) -> bool:
        """检查是否应该停止训练。"""
        return self.wait >= self.patience


class MLflowCallback(Callback):
    """
    MLflow Callback
    
    自动记录：
    - Config（训练开始时）
    - Metrics（每个 epoch）
    - Artifacts（训练结束时）
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_config: bool = True,
        log_artifacts: bool = True,
        artifact_paths: Optional[list] = None,
    ):
        """
        初始化 MLflow Callback。
        
        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow tracking URI（如果为 None，使用 config 中的）
            log_config: 是否自动记录配置
            log_artifacts: 是否自动保存 artifacts
            artifact_paths: 要保存的 artifact 路径列表（如果为 None，自动保存模型和配置）
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_config = log_config
        self.log_artifacts = log_artifacts
        self.artifact_paths = artifact_paths or []
        self.mlflow = None
        self.run_id = None
        self.config_dict = {}
        
        # 延迟导入 MLflow
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            logger.warning("MLflow not available, MLflowCallback will be disabled")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """开始 MLflow run，自动记录 config。"""
        if self.mlflow is None:
            return
        
        try:
            # 设置 tracking URI
            if self.tracking_uri:
                self.mlflow.set_tracking_uri(self.tracking_uri)
            else:
                # 使用 config 中的 tracking URI
                from ..config import MLFLOW_TRACKING_URI
                if MLFLOW_TRACKING_URI:
                    self.mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # 设置实验
            if self.experiment_name:
                self.mlflow.set_experiment(self.experiment_name)
            
            # 开始 run
            self.mlflow.start_run()
            self.run_id = self.mlflow.active_run().info.run_id
            logger.info(f"MLflow run started: {self.run_id}")
            
            # 自动记录 config
            if self.log_config and logs:
                self.config_dict = logs.get('config', {})
                if self.config_dict:
                    # 记录参数（扁平化嵌套字典）
                    params = self._flatten_dict(self.config_dict)
                    self.mlflow.log_params(params)
                    logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """自动记录 epoch 指标。"""
        if self.mlflow is None or logs is None:
            return
        
        try:
            # 过滤并转换指标
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    # 如果是列表，记录最后一个值
                    metrics[k] = float(v[-1]) if isinstance(v[-1], (int, float)) else v[-1]
            
            if metrics:
                metrics['epoch'] = epoch
                self.mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """结束 MLflow run，自动保存 artifacts。"""
        if self.mlflow is None:
            return
        
        try:
            # 自动保存 artifacts
            if self.log_artifacts:
                # 保存模型（如果提供）
                if logs and 'model_path' in logs:
                    model_path = logs['model_path']
                    if model_path and Path(model_path).exists():
                        self.mlflow.log_artifact(model_path, artifact_path="model")
                        logger.info(f"Logged model artifact: {model_path}")
                
                # 保存配置文件（如果提供）
                if logs and 'config_path' in logs:
                    config_path = logs['config_path']
                    if config_path and Path(config_path).exists():
                        self.mlflow.log_artifact(config_path, artifact_path="config")
                        logger.info(f"Logged config artifact: {config_path}")
                
                # 保存其他 artifacts
                for artifact_path in self.artifact_paths:
                    if Path(artifact_path).exists():
                        self.mlflow.log_artifact(artifact_path)
                        logger.info(f"Logged artifact: {artifact_path}")
            
            # 结束 run
            self.mlflow.end_run()
            logger.info(f"MLflow run ended: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """扁平化嵌套字典。"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # 转换为字符串（MLflow 参数必须是字符串）
                items.append((new_key, str(v)))
        return dict(items)
    
    def log_artifact(self, artifact_path: str, artifact_path_in_mlflow: Optional[str] = None):
        """手动记录 artifact。"""
        if self.mlflow is None:
            return
        
        try:
            self.mlflow.log_artifact(artifact_path, artifact_path=artifact_path_in_mlflow)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")


class TensorBoardCallback(Callback):
    """
    TensorBoard Callback
    
    自动记录：
    - Config（训练开始时，作为 text）
    - Metrics（每个 epoch）
    - Model graph（训练开始时，如果可能）
    """
    
    def __init__(
        self,
        log_dir: str = "./logs/tensorboard",
        log_config: bool = True,
        log_model_graph: bool = True,
    ):
        """
        初始化 TensorBoard Callback。
        
        Args:
            log_dir: 日志目录
            log_config: 是否记录配置
            log_model_graph: 是否记录模型图
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_config = log_config
        self.log_model_graph = log_model_graph
        self.writer = None
        self.model = None
        
        # 延迟导入 TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, TensorBoardCallback will be disabled")
    
    def set_model(self, model):
        """设置模型（用于记录模型图）。"""
        self.model = model
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """记录配置和模型图。"""
        if self.writer is None:
            return
        
        try:
            # 记录配置（作为 text）
            if self.log_config and logs:
                config_dict = logs.get('config', {})
                if config_dict:
                    config_text = self._dict_to_text(config_dict)
                    self.writer.add_text('config', config_text, 0)
                    logger.info("Logged config to TensorBoard")
            
            # 记录模型图（如果可能）
            if self.log_model_graph and self.model is not None:
                try:
                    # 尝试获取一个示例输入
                    if logs and 'sample_input' in logs:
                        sample_input = logs['sample_input']
                        self.writer.add_graph(self.model, sample_input)
                        logger.info("Logged model graph to TensorBoard")
                except Exception as e:
                    logger.debug(f"Could not log model graph: {e}")
        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """自动记录 epoch 指标。"""
        if self.writer is None or logs is None:
            return
        
        try:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # 如果是列表，记录最后一个值
                    if isinstance(value[-1], (int, float)):
                        self.writer.add_scalar(key, value[-1], epoch)
        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """关闭 TensorBoard writer。"""
        if self.writer is not None:
            try:
                self.writer.close()
                logger.info(f"TensorBoard logs saved to {self.log_dir}")
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard writer: {e}")
    
    def _dict_to_text(self, d: Dict[str, Any], indent: int = 0) -> str:
        """将字典转换为文本格式。"""
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append("  " * indent + f"{k}:")
                lines.append(self._dict_to_text(v, indent + 1))
            else:
                lines.append("  " * indent + f"{k}: {v}")
        return "\n".join(lines)

