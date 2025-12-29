"""
训练任务

借鉴 Made-With-ML 的训练流程，支持：
- LocalBackend: 本地训练 (CPU/GPU/MPS)，直接使用 Trainer
- RayBackend: Ray 分布式训练，在 worker 中复用 Trainer 逻辑
- MLflow 实验跟踪
- Checkpoint 管理
"""

from __future__ import annotations

import abc
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import EFS_DIR, logger
from ..engine import Trainer
from ..metadata import save_model_metadata
from ..models import get_model
from ..pipeline import PipelineOrchestrator
from ..utils import collate_fn, set_seeds

# Ray imports (optional)
try:
    import ray
    import ray.train as train
    from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from torch.nn.parallel.distributed import DistributedDataParallel

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class TrainingBackend(abc.ABC):
    """训练后端基类"""

    @abc.abstractmethod
    def run(
        self,
        pipeline_config_path: str,
        experiment_name: str | None = None,
        num_epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        output_dir: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """运行训练任务"""
        pass


class LocalBackend(TrainingBackend):
    """本地训练后端 (CPU/GPU/MPS)"""

    def run(
        self,
        model: Any,
        train_dataset: Any,
        val_dataset: Any | None = None,
        train_config: dict[str, Any] | None = None,
        output_dir: str | None = None,
        orchestrator: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info("Using LocalBackend for training...")

        if train_config is None:
            train_config = {}

        # 4. 准备 DataLoaders
        # LocalBackend 使用标准的 PyTorch DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config.get("batch_size", 32), collate_fn=collate_fn, num_workers=0, pin_memory=True  # 简化起见，可配置
        )

        # 验证集
        val_loader = None
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=train_config.get("batch_size", 32), collate_fn=collate_fn, num_workers=0, pin_memory=True
            )

        # 6. 初始化 Trainer
        # 从 train_config 中提取 paradigm 配置
        learning_paradigm = train_config.get("learning_paradigm", "supervised")
        paradigm_config = train_config.get("paradigm_config", {})
        task_type = train_config.get("task_type", "classification")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=None,  # 使用默认 Adam
            task_type=task_type,
            learning_paradigm=learning_paradigm,
            paradigm_config=paradigm_config,
            # device=None, # 自动检测
        )

        # 更新学习率 (如果 Trainer 创建了默认优化器)
        learning_rate = train_config.get("learning_rate")
        if learning_rate is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # 7. 开始训练
        logger.info(f"Starting training with paradigm: {learning_paradigm}")
        history = trainer.fit(
            num_epochs=train_config.get("num_epochs", 10),
            save_dir=output_dir,
        )

        # 保存模型
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path / "model.pt")
            logger.info(f"Model saved to {output_path / 'model.pt'}")

            # 6. 保存 PipelineState 和 Metadata
            if orchestrator:
                # 保存 PipelineState
                if orchestrator.pipeline_state:
                    orchestrator.save_pipeline_state(output_path / "pipeline_state.json")

                # 保存 Metadata
                feature_graph = orchestrator.feature_graph
                feature_spec = feature_graph.output_spec() if feature_graph else {}
                feature_state = feature_graph.export_state() if feature_graph else {}

                # 获取模型配置和输入维度
                model_config = orchestrator.config.get("model", {})
                # input_dim 可以在 PipelineState 中找到，或者重新推断
                input_dim = None
                input_key = "event"  # 默认
                if orchestrator.pipeline_state:
                    input_dim = orchestrator.pipeline_state.input_dim
                    input_key = orchestrator.pipeline_state.input_key

                save_model_metadata(
                    output_path / "metadata.json",
                    feature_spec=feature_spec,
                    task_type=train_config.get("task_type", "classification"),
                    model_config=model_config,
                    input_dim=input_dim,
                    input_key=input_key,
                    feature_state=feature_state,
                    experiment_name=kwargs.get("experiment_name", "default"),
                )
                logger.info(f"Saved pipeline state and metadata to {output_dir}")

        return history


# Ray 相关的辅助函数
def _convert_dataset_to_ray(dataset, max_samples: int | None = None):
    """将 HEPDataset 转换为 Ray Dataset"""
    import ray.data

    # 收集数据样本（限制数量以避免内存问题）
    # 注意：这对于大数据集不是最优的，但在当前架构下 HEPDataset 是 Iterable，
    # 且通常是流式的。为了 Ray 分布式，我们需要将其物化或使用 Ray 的读取器。
    # 这里保持原有的简化实现。
    samples = []
    logger.info(f"Converting dataset to Ray format (max_samples={max_samples})...")
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        sample_dict = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample_dict[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                sample_dict[key] = value
            else:
                try:
                    sample_dict[key] = np.array(value)
                except Exception:
                    sample_dict[key] = value
        samples.append(sample_dict)

        if (i + 1) % 1000 == 0:
            logger.info(f"Converted {i + 1} samples...")

    return ray.data.from_items(samples)


def train_loop_per_worker(config: dict) -> None:
    """Ray Worker 训练循环：复用 Trainer"""
    # 1. 获取配置
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    task_type = config["task_type"]
    model_name = config["model_name"]
    model_params = config["model_params"]
    learning_paradigm = config.get("learning_paradigm", "supervised")
    paradigm_config = config.get("paradigm_config", {})

    # 2. 设置随机种子
    set_seeds()

    # 3. 准备数据 Iterator
    # Ray Train 会自动分片
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # 将 Ray Dataset 转换为 Trainer 可用的 Iterable (yields batches)
    # 注意：batch_size 在这里是 per_worker
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers

    train_loader = train_ds.iter_torch_batches(batch_size=batch_size_per_worker, collate_fn=collate_fn)

    val_loader = None
    if val_ds:
        val_loader = val_ds.iter_torch_batches(batch_size=batch_size_per_worker, collate_fn=collate_fn)

    # 4. 准备模型
    model = get_model(model_name, **model_params)
    # Ray Prepare Model (DDP wrapping)
    model = train.torch.prepare_model(model)

    # 5. 实例化 Trainer
    # 注意：我们传递已包装的 model 和 iterator
    # 必须指定 device，Ray worker 会自动设置 CUDA_VISIBLE_DEVICES，
    # 但我们需要显式告诉 Trainer 使用当前 device
    device = train.torch.get_device()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=None,  # 让 Trainer 创建默认优化器
        task_type=task_type,
        learning_paradigm=learning_paradigm,
        paradigm_config=paradigm_config,
        device=device,
    )

    # 更新学习率
    if learning_rate is not None:
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = learning_rate

    # 6. 训练循环
    # 我们不能直接调用 trainer.fit()，因为它控制了 epoch 循环和保存逻辑，
    # 而 Ray 通常希望我们在外部控制 epoch 以便 report metrics。
    # 但我们可以调用 trainer.train_epoch()。

    for epoch in range(num_epochs):
        # 训练一个 epoch
        train_metrics = trainer.train_epoch()

        # 验证
        val_metrics = {}
        if val_loader:
            val_metrics = trainer.validate()

        # 汇总指标
        metrics = {"epoch": epoch, **train_metrics, **val_metrics}

        # Checkpoint (仅在主 worker 或通过 Ray 机制)
        # Ray Train 会处理 checkpoint 上传
        with tempfile.TemporaryDirectory() as dp:
            # 获取原始模型状态 (解包 DDP)
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save(state_dict, f"{dp}/model.pt")

            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


class RayBackend(TrainingBackend):
    """Ray 分布式训练后端"""

    def __init__(self, num_workers: int = 1, gpu_per_worker: int = 0):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Install with: pip install 'ray[default]' 'ray[train]'")
        self.num_workers = num_workers
        self.gpu_per_worker = gpu_per_worker

    def run(
        self,
        model: Any,
        train_dataset: Any,
        val_dataset: Any | None = None,
        train_config: dict[str, Any] | None = None,
        output_dir: str | None = None,
        orchestrator: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info("Using RayBackend for distributed training...")

        if not ray.is_initialized():
            ray.init()

        # 2. 转换数据
        # TODO: 优化大数据集处理
        ray_train_ds = _convert_dataset_to_ray(train_dataset)

        ray_val_ds = None
        if val_dataset:
            ray_val_ds = _convert_dataset_to_ray(val_dataset)

        if train_config is None:
            train_config = {}

        # 获取模型参数以便在 worker 中重建
        # 注意：Ray Trainer 需要在 worker 中重建模型，或者我们可以传递 state_dict
        # 但这里为了简单，我们重新从 config 构建
        if orchestrator:
            model_config = orchestrator.config["model"]
            input_dim = orchestrator.get_input_dim_from_spec()
            model_params = {**model_config.get("params", {}), "input_dim": input_dim}
            model_name = model_config["name"]
        else:
            # Fallback if orchestrator not provided (should not happen in current train_task)
            raise ValueError("RayBackend requires orchestrator to reconstruct model in workers")

        # 4. Ray Trainer 配置
        train_loop_config = {
            "num_epochs": train_config.get("num_epochs", 10),
            "batch_size": train_config.get("batch_size", 32),
            "learning_rate": train_config.get("learning_rate", 1e-3),
            "task_type": train_config.get("task_type", "classification"),
            "learning_paradigm": train_config.get("learning_paradigm", "supervised"),
            "paradigm_config": train_config.get("paradigm_config", {}),
            "model_name": model_name,
            "model_params": model_params,
        }

        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=bool(self.gpu_per_worker),
            resources_per_worker={"CPU": 1, "GPU": self.gpu_per_worker},
        )

        run_config = RunConfig(
            storage_path=str(EFS_DIR),
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        )

        datasets = {"train": ray_train_ds}
        if ray_val_ds:
            datasets["val"] = ray_val_ds

        # 5. 启动 Ray Trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets=datasets,
        )

        results = trainer.fit()

        # 6. 保存最终结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # 获取最佳 checkpoint 并保存
            if results.best_checkpoints:
                best_checkpoint = results.best_checkpoints[0][0]
                # Ray checkpoint 处理逻辑...
                # 这里简化：只打印路径，实际应复制文件
                logger.info(f"Best checkpoint saved at {best_checkpoint.path}")

            # 7. 保存 PipelineState 和 Metadata (RayBackend)
            if orchestrator:
                # 保存 PipelineState
                if orchestrator.pipeline_state:
                    orchestrator.save_pipeline_state(output_dir / "pipeline_state.json")

                # 保存 Metadata
                from ..metadata import save_model_metadata

                feature_graph = orchestrator.feature_graph
                feature_spec = feature_graph.output_spec() if feature_graph else {}
                feature_state = feature_graph.export_state() if feature_graph else {}

                # 获取模型配置和输入维度
                model_config = orchestrator.config.get("model", {})
                input_dim = None
                input_key = "event"
                if orchestrator.pipeline_state:
                    input_dim = orchestrator.pipeline_state.input_dim
                    input_key = orchestrator.pipeline_state.input_key

                save_model_metadata(
                    output_dir / "metadata.json",
                    feature_spec=feature_spec,
                    task_type=train_config.get("task_type", "classification"),
                    model_config=model_config,
                    input_dim=input_dim,
                    input_key=input_key,
                    feature_state=feature_state,
                    experiment_name=kwargs.get("experiment_name", "default"),
                )
                logger.info(f"Saved pipeline state and metadata to {output_dir}")

        return {"results": results}


def train_task(
    pipeline_config_path: str,
    experiment_name: str | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    output_dir: str | None = None,
    use_ray: bool = False,
    num_workers: int = 1,
    gpu_per_worker: int = 0,
) -> dict[str, Any]:
    """
    训练任务入口
    """
    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    # 2. 设置数据
    logger.info("Setting up data system...")
    # 自动拟合特征（使用部分数据）
    train_dataset, val_dataset = orchestrator.setup_data(fit_features=True)

    # 3. 获取训练配置
    train_config = orchestrator.get_train_config()

    # 覆盖配置（如果提供）
    if num_epochs is not None:
        train_config["num_epochs"] = num_epochs
    if batch_size is not None:
        train_config["batch_size"] = batch_size
    if learning_rate is not None:
        train_config["learning_rate"] = learning_rate

    # 4. 设置模型
    logger.info("Setting up model...")
    # 此时 FeatureGraph 已拟合，可以准确推断维度
    model = orchestrator.setup_model()

    # 5. 运行训练
    logger.info(f"Starting training with backend: {'Ray' if use_ray else 'Local'}")

    # 选择 Backend
    if use_ray:
        backend = RayBackend(num_workers=num_workers, gpu_per_worker=gpu_per_worker)
    else:
        backend = LocalBackend()

    result = backend.run(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_config=train_config,
        output_dir=output_dir,
        orchestrator=orchestrator,  # 传递 orchestrator 以便保存状态
    )

    # 6. 保存 PipelineState 和 Metadata
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 PipelineState
        orchestrator.save_pipeline_state(output_path / "pipeline_state.json")

        # 保存 Metadata (包含 FeatureGraph 状态)
        from ..metadata import save_model_metadata

        feature_graph = orchestrator.get_feature_graph()
        feature_spec = feature_graph.output_spec() if feature_graph else {}
        feature_state = feature_graph.export_state() if feature_graph else {}
        task_type = train_config.get("task_type", "classification")

        save_model_metadata(
            output_path / "metadata.json",
            feature_spec=feature_spec,
            task_type=task_type,
            model_config=orchestrator.get_model_config(),
            input_dim=orchestrator.get_pipeline_state().input_dim,
            input_key=orchestrator.get_pipeline_state().input_key,
            feature_state=feature_state,
            experiment_name=experiment_name,
        )
        logger.info(f"Saved pipeline state and metadata to {output_dir}")

    return result
