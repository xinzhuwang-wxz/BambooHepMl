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
            train_dataset,
            batch_size=train_config.get("batch_size", 32),
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,  # 简化起见，可配置
        )

        # 验证集
        val_loader = None
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=train_config.get("batch_size", 32),
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )

        # 6. 初始化 Trainer
        # 从 train_config 中提取 paradigm 配置
        learning_paradigm = train_config.get("learning_paradigm", "supervised")
        paradigm_config = train_config.get("paradigm_config", {})
        task_type = train_config.get("task_type", "classification")

        # 创建 MLflow callback（如果有实验/运行名称）
        callbacks = []
        mlflow_experiment = kwargs.get("mlflow_experiment_name")
        mlflow_run = kwargs.get("mlflow_run_name")
        if mlflow_experiment or mlflow_run:
            try:
                from ..engine.callbacks import MLflowCallback

                mlflow_callback = MLflowCallback(
                    experiment_name=mlflow_experiment,
                    run_name=mlflow_run,
                )
                callbacks.append(mlflow_callback)
                logger.info(f"MLflow tracking enabled: experiment='{mlflow_experiment}', " f"run='{mlflow_run}'")
            except Exception as e:
                logger.warning(f"Failed to create MLflowCallback: {e}")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=None,  # 使用默认 Adam
            task_type=task_type,
            learning_paradigm=learning_paradigm,
            paradigm_config=paradigm_config,
            callbacks=callbacks,
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

        # 保存模型（使用最佳模型，如果存在）
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 优先使用 best_model.pt（如果存在），否则使用 final_model.pt
            best_model_path = output_path / "best_model.pt"
            final_model_path = output_path / "final_model.pt"

            if best_model_path.exists():
                # 复制 best_model.pt 为 model.pt（推荐使用）
                import shutil

                shutil.copy(best_model_path, output_path / "model.pt")
                logger.info(f"Model saved to {output_path / 'model.pt'} (copied from best_model.pt)")
            elif final_model_path.exists():
                # 如果没有 best_model.pt，使用 final_model.pt
                import shutil

                shutil.copy(final_model_path, output_path / "model.pt")
                logger.info(f"Model saved to {output_path / 'model.pt'} (copied from final_model.pt)")
            else:
                # 如果都没有，保存当前模型（最后一个 epoch）
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

                # 获取模型配置（已包含 event_input_dim/object_input_dim/embed_dim）
                model_config = orchestrator.config.get("model", {})

                save_model_metadata(
                    output_path / "metadata.json",
                    feature_spec=feature_spec,
                    task_type=train_config.get("task_type", "classification"),
                    model_config=model_config,
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
            # setup_model 已经自动推断并设置了 event_input_dim/object_input_dim
            model_params = model_config.get("params", {}).copy()
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

                # 获取模型配置（已包含 event_input_dim/object_input_dim/embed_dim）
                model_config = orchestrator.config.get("model", {})

                save_model_metadata(
                    output_dir / "metadata.json",
                    feature_spec=feature_spec,
                    task_type=train_config.get("task_type", "classification"),
                    model_config=model_config,
                    feature_state=feature_state,
                    experiment_name=kwargs.get("experiment_name", "default"),
                )
                logger.info(f"Saved pipeline state and metadata to {output_dir}")

        return {"results": results}


def _resolve_data_config_path(task_type: str, pipeline_config_path: str) -> str:
    """
    Resolve the data config YAML path for a given task_type.

    Looks for configs/data_edm4hep_{task_type}.yaml relative to the
    pipeline config directory.  Returns an absolute path string.

    Args:
        task_type: 'classification' or 'regression'
        pipeline_config_path: path to the pipeline YAML (used as anchor)

    Returns:
        Absolute path to the data config YAML

    Raises:
        FileNotFoundError: if the resolved file does not exist
    """
    config_dir = Path(pipeline_config_path).resolve().parent
    data_config = config_dir / f"data_edm4hep_{task_type}.yaml"
    if not data_config.exists():
        raise FileNotFoundError(f"Data config for task_type='{task_type}' not found: {data_config}")
    return str(data_config)


def _resolve_model_name(task_type: str, model_type: str) -> str | None:
    """
    Derive the concrete model name from task_type + model_type.

    Returns None (with a warning) for unsupported combinations.
    """
    mapping = {
        ("classification", "torch"): "mlp_classifier",
        ("regression", "torch"): "mlp_regressor",
        ("classification", "xgboost"): "xgb_classifier",
        ("regression", "xgboost"): "xgb_regressor",
    }
    name = mapping.get((task_type, model_type))
    if name is None:
        logger.warning(
            f"No built-in model mapping for task_type='{task_type}', " f"model_type='{model_type}'. Falling back to config-driven model selection."
        )
    return name


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
    task_type: str | None = None,
    model_type: str | None = None,
    run_index: int = 1,
) -> dict[str, Any]:
    """
    训练任务入口

    Args:
        pipeline_config_path: pipeline.yaml 路径
        experiment_name: MLflow 实验名称
        num_epochs / batch_size / learning_rate: 训练超参覆盖
        output_dir: 输出目录（若 task_type/model_type 指定，自动加子目录）
        use_ray: 是否使用 Ray 分布式训练
        num_workers / gpu_per_worker: Ray 配置
        task_type: 'classification' 或 'regression'（多实验模式）
        model_type: 'torch' 或 'xgboost'（多实验模式）
        run_index: 当前重复运行序号（用于 seed 偏移和日志）
    """
    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    # --- Multi-experiment overrides ---
    if task_type is not None:
        # Inject the task-specific data config into the orchestrator config
        data_config_path = _resolve_data_config_path(task_type, pipeline_config_path)
        orchestrator.config.setdefault("data", {})["config_path"] = data_config_path
        logger.info(f"[multi-run] task_type='{task_type}' → data config: {data_config_path}")

    if model_type is not None and task_type is not None:
        model_name = _resolve_model_name(task_type, model_type)
        if model_name is not None:
            orchestrator.config.setdefault("model", {})["name"] = model_name
            logger.info(f"[multi-run] model_type='{model_type}' → model name: {model_name}")

    if task_type is not None:
        # Ensure training config has the correct task_type
        training_section = orchestrator.config.get("training", orchestrator.config.get("train", {}))
        training_section["task_type"] = task_type
        # Write it back under the canonical key
        orchestrator.config["training"] = training_section

    # Resolve output subdirectory for multi-experiment runs
    effective_output_dir = output_dir
    if task_type is not None and model_type is not None:
        sub = f"{task_type}_{model_type}/run_{run_index}"
        if effective_output_dir:
            effective_output_dir = str(Path(effective_output_dir) / sub)
        else:
            base = orchestrator.config.get("output", {}).get("base_dir", "outputs")
            effective_output_dir = str(Path(base) / sub)
        logger.info(f"[multi-run] output directory: {effective_output_dir}")

    # Auto-generate MLflow experiment/run names for multi-experiment mode
    # Convention: experiment = {task_type}_{model_type}, run = {model_type}_{timestamp}
    effective_experiment_name = experiment_name
    if effective_experiment_name is None and task_type is not None and model_type is not None:
        effective_experiment_name = f"{task_type}_{model_type}"
        logger.info(f"[multi-run] experiment name: {effective_experiment_name}")

    from datetime import datetime

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if task_type is not None and model_type is not None:
        # Use the concrete model name (e.g. mlp_classifier) for readability,
        # falling back to model_type (e.g. torch) if unresolved
        resolved_name = _resolve_model_name(task_type, model_type)
        run_model_label = resolved_name or model_type
        mlflow_run_name = f"{run_model_label}_{run_timestamp}"
    else:
        mlflow_run_name = f"run_{run_timestamp}"

    # Seed variation per run
    seed = orchestrator.config.get("training", {}).get("seed", 42)
    run_seed = seed + (run_index - 1)
    set_seeds(run_seed)
    logger.info(f"[multi-run] run_index={run_index}, seed={run_seed}")

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

    # Ensure task_type is set in train_config for downstream consumers
    if task_type is not None:
        train_config["task_type"] = task_type

    # --- XGBoost branch: separate code path ---
    if model_type == "xgboost":
        from .xgboost_train import train_xgboost_task

        resolved_task = task_type or train_config.get("task_type", "classification")
        result = train_xgboost_task(
            orchestrator=orchestrator,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            task_type=resolved_task,
            effective_experiment_name=effective_experiment_name,
            mlflow_run_name=mlflow_run_name,
            output_dir=effective_output_dir,
            num_epochs=train_config.get("num_epochs"),
            seed=run_seed,
        )
        return result

    # 4. 设置模型（PyTorch path）
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
        output_dir=effective_output_dir,
        orchestrator=orchestrator,  # 传递 orchestrator 以便保存状态
        mlflow_experiment_name=effective_experiment_name,
        mlflow_run_name=mlflow_run_name,
    )

    # 6. 保存 PipelineState 和 Metadata
    if effective_output_dir:
        output_path = Path(effective_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 PipelineState
        orchestrator.save_pipeline_state(output_path / "pipeline_state.json")

        # 保存 Metadata (包含 FeatureGraph 状态)
        from ..metadata import save_model_metadata

        feature_graph = orchestrator.get_feature_graph()
        feature_spec = feature_graph.output_spec() if feature_graph else {}
        feature_state = feature_graph.export_state() if feature_graph else {}
        resolved_task_type = train_config.get("task_type", "classification")

        save_model_metadata(
            output_path / "metadata.json",
            feature_spec=feature_spec,
            task_type=resolved_task_type,
            model_config=orchestrator.get_model_config(),
            feature_state=feature_state,
            experiment_name=effective_experiment_name,
        )
        logger.info(f"Saved pipeline state and metadata to {effective_output_dir}")

    return result
