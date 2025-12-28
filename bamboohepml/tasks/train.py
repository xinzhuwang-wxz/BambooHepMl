"""
训练任务

借鉴 Made-With-ML 的训练流程，支持：
- Ray 分布式训练
- MLflow 实验跟踪
- TensorBoard 日志
- Checkpoint 管理
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import EFS_DIR, MLFLOW_TRACKING_URI, logger
from ..models import get_model
from ..pipeline import PipelineOrchestrator

# Note: FeatureGraph and ExpressionEngine are available but not used in this implementation
# They can be used for feature engineering if needed
# from ..data.features import FeatureGraph, ExpressionEngine
# Note: Trainer, Evaluator, and Callbacks are available but not used in this implementation
# They can be used for non-Ray training paths if needed
# from ..engine import Trainer, Evaluator, EarlyStoppingCallback, LoggingCallback
from ..utils import collate_fn, set_seeds

# Ray imports (optional)
try:
    import ray
    import ray.train as train
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from ray.data import Dataset
    from ray.train import Checkpoint, CheckpointConfig, DataConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from torch.nn.parallel.distributed import DistributedDataParallel

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def _convert_dataset_to_ray(dataset, max_samples: Optional[int] = None):
    """将 HEPDataset 转换为 Ray Dataset（简化版）。

    注意：这是一个简化实现，对于大数据集可能效率不高。
    实际使用时应该考虑使用 Ray 的原生数据加载方式。
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install 'ray[default]' 'ray[train]'")

    import ray.data

    # 收集数据样本（限制数量以避免内存问题）
    samples = []
    logger.info(f"Converting dataset to Ray format (max_samples={max_samples})...")
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        # 转换为可序列化的格式
        sample_dict = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample_dict[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                sample_dict[key] = value
            else:
                # 尝试转换为 numpy
                try:
                    sample_dict[key] = np.array(value)
                except Exception:
                    sample_dict[key] = value
        samples.append(sample_dict)

        if (i + 1) % 1000 == 0:
            logger.info(f"Converted {i + 1} samples...")

    logger.info(f"Total samples converted: {len(samples)}")

    # 创建 Ray Dataset
    return ray.data.from_items(samples)


def train_step_ray(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_key: str,
    task_type: str,
) -> float:
    """Ray 训练步骤。"""
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()
        inputs = batch[input_key]
        labels = batch.get("_label_", None)

        outputs = model({"features": inputs})

        if task_type == "classification" and labels is not None:
            loss_val = loss_fn(outputs, labels)
        elif task_type == "regression" and labels is not None:
            loss_val = loss_fn(outputs.squeeze(), labels.float())
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        loss_val.backward()
        optimizer.step()
        loss += (loss_val.detach().item() - loss) / (i + 1)
    return loss


def eval_step_ray(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    loss_fn: nn.Module,
    input_key: str,
    task_type: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Ray 评估步骤。"""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            inputs = batch[input_key]
            labels = batch.get("_label_", None)

            outputs = model({"features": inputs})

            if task_type == "classification" and labels is not None:
                loss_val = loss_fn(outputs, labels).item()
                y_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            elif task_type == "regression" and labels is not None:
                loss_val = loss_fn(outputs.squeeze(), labels.float()).item()
                y_preds.extend(outputs.squeeze().cpu().numpy())
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            loss += (loss_val - loss) / (i + 1)
            if labels is not None:
                y_trues.extend(labels.cpu().numpy())

    return loss, np.vstack(y_trues) if y_trues else np.array([]), np.vstack(y_preds) if y_preds else np.array([])


def train_loop_per_worker(config: dict) -> None:
    """每个 worker 执行的训练循环（Ray）。"""
    # 从配置获取参数
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    task_type = config["task_type"]
    model_name = config["model_name"]
    model_params = config["model_params"]
    input_key = config["input_key"]

    # 设置随机种子
    set_seeds()

    # 获取数据集分片
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # 创建模型
    model = get_model(model_name, **model_params)
    model = train.torch.prepare_model(model)

    # 训练组件
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器（可选）
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler_type = config.get("scheduler_type", "step")
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("scheduler_step_size", 10),
                gamma=config.get("scheduler_gamma", 0.1),
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=config.get("lr_factor", 0.1), patience=config.get("lr_patience", 5)
            )

    # 训练循环
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers

    for epoch in range(num_epochs):
        # 训练
        train_loss = train_step_ray(train_ds, batch_size_per_worker, model, loss_fn, optimizer, input_key, task_type)

        # 验证
        val_loss, _, _ = eval_step_ray(val_ds, batch_size_per_worker, model, loss_fn, input_key, task_type)

        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Checkpoint
        with tempfile.TemporaryDirectory() as dp:
            if isinstance(model, DistributedDataParallel):
                torch.save(model.module.state_dict(), f"{dp}/model.pt")
            else:
                torch.save(model.state_dict(), f"{dp}/model.pt")

            metrics = dict(
                epoch=epoch,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                val_loss=val_loss,
            )
            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


def train_task(
    pipeline_config_path: str,
    experiment_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    output_dir: Optional[str] = None,
    use_ray: bool = False,
    num_workers: int = 1,
    gpu_per_worker: int = 0,
) -> Dict[str, Any]:
    """
    训练任务主函数。

    Args:
        pipeline_config_path: pipeline.yaml 路径
        experiment_name: 实验名称（用于 MLflow）
        num_epochs: 训练轮数（覆盖配置）
        batch_size: 批次大小（覆盖配置）
        learning_rate: 学习率（覆盖配置）
        output_dir: 输出目录
        use_ray: 是否使用 Ray 分布式训练
        num_workers: Ray worker 数量
        gpu_per_worker: 每个 worker 的 GPU 数量

    Returns:
        训练结果字典
    """
    logger.info("=" * 80)
    logger.info("Training Task")
    logger.info("=" * 80)

    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    # 2. 设置数据
    logger.info("Setting up data system...")
    train_dataset = orchestrator.setup_data()

    # 3. 获取训练配置
    train_config = orchestrator.get_train_config()

    # 覆盖配置（如果提供）
    if num_epochs is not None:
        train_config["num_epochs"] = num_epochs
    if batch_size is not None:
        train_config["batch_size"] = batch_size
    if learning_rate is not None:
        train_config["learning_rate"] = learning_rate

    num_epochs = train_config.get("num_epochs", 10)
    batch_size = train_config.get("batch_size", 32)
    learning_rate = train_config.get("learning_rate", 1e-3)
    task_type = train_config.get("task_type", "classification")

    # 4. 设置模型
    logger.info("Setting up model...")
    # 从数据中推断输入维度
    sample = next(iter(train_dataset))
    input_key = None
    for key in sample.keys():
        if key.startswith("_") and key != "_label_":
            input_key = key
            break

    if input_key is None:
        raise ValueError("Could not find input key in dataset")

    input_value = sample[input_key]
    if isinstance(input_value, torch.Tensor):
        if len(input_value.shape) == 1:
            input_dim = input_value.shape[0]
        elif len(input_value.shape) == 2:
            input_dim = input_value.shape[1]
        else:
            input_dim = int(torch.prod(torch.tensor(input_value.shape)))
    else:
        raise ValueError(f"Unexpected input type: {type(input_value)}")

    model_config = orchestrator.get_model_config()
    model_name = model_config.get("name")
    model_params = model_config.get("params", {})
    model_params["input_dim"] = input_dim

    model = orchestrator.setup_model(input_dim=input_dim)

    # 5. Ray 分布式训练分支
    if use_ray:
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Install with: pip install 'ray[default]' 'ray[train]'")

        logger.info("Using Ray distributed training...")

        # 初始化 Ray（如果未初始化）
        if not ray.is_initialized():
            ray.init()

        # 转换数据集为 Ray Dataset
        logger.info("Converting datasets to Ray format...")
        ray_train_ds = _convert_dataset_to_ray(train_dataset)

        # 创建验证集（简化：使用训练集的一部分）
        # 实际应该从 orchestrator 获取验证集
        val_dataset = orchestrator.setup_data()
        val_dataset.for_training = False
        val_dataset.shuffle = False
        ray_val_ds = _convert_dataset_to_ray(val_dataset)

        # 准备训练循环配置
        train_loop_config = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "task_type": task_type,
            "model_name": model_name,
            "model_params": model_params,
            "input_key": input_key,
            **train_config,
        }

        # Scaling config
        scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=bool(gpu_per_worker),
            resources_per_worker={"CPU": 1, "GPU": gpu_per_worker},
        )

        # Checkpoint config
        checkpoint_config = CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        )

        # MLflow callback
        callbacks_list = []
        if experiment_name and MLFLOW_TRACKING_URI:
            try:
                mlflow_callback = MLflowLoggerCallback(
                    tracking_uri=MLFLOW_TRACKING_URI,
                    experiment_name=experiment_name,
                    save_artifact=True,
                )
                callbacks_list.append(mlflow_callback)
            except Exception as e:
                logger.warning(f"Failed to create MLflow callback: {e}")

        # Run config
        run_config = RunConfig(
            callbacks=callbacks_list,
            checkpoint_config=checkpoint_config,
            storage_path=str(EFS_DIR),
            local_dir=str(EFS_DIR),
        )

        # Dataset config
        options = ray.data.ExecutionOptions(preserve_order=True)
        dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

        # Trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets={"train": ray_train_ds, "val": ray_val_ds},
            dataset_config=dataset_config,
        )

        # Train
        results = trainer.fit()

        # 保存结果
        if output_dir is None:
            output_dir = Path(orchestrator.config.get("output_dir", "./outputs"))
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存最佳模型
        best_checkpoint = results.best_checkpoints[0][0] if results.best_checkpoints else None
        model_path = None
        if best_checkpoint:
            import shutil

            model_path = output_dir / "model.pt"
            shutil.copy2(f"{best_checkpoint.path}/model.pt", str(model_path))
            logger.info(f"Best model saved to {model_path}")

        # 返回结果
        return {
            "experiment_name": experiment_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "history": results.metrics_dataframe.to_dict() if hasattr(results, "metrics_dataframe") else {},
            "model_path": str(model_path) if model_path else None,
            "output_dir": str(output_dir),
            "ray_results": results,
        }

    # 6. 普通训练分支（原有代码）
    logger.info("Using standard training (non-Ray)...")

    # 6.1 设置实验跟踪（MLflow + TensorBoard）
    from ..experiment import ExperimentTracker

    experiment_tracker = ExperimentTracker(
        experiment_name=experiment_name,
        use_mlflow=True,
        use_tensorboard=True,
        log_config=True,
        log_artifacts=True,
    )

    # 准备配置字典（用于记录）
    config_dict = {
        "pipeline_config": str(pipeline_config_path),
        "model": {
            "name": model_name,
            "params": model_params,
        },
        "train": train_config,
        "data": {
            "input_dim": input_dim,
            "input_key": input_key,
        },
    }

    # 开始实验 run
    experiment_tracker.start_run(config=config_dict, model=model)

    # 6.2 设置训练组件
    logger.info("Setting up training components...")

    # 损失函数（从任务类型推断）
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 优化器
    optimizer_name = train_config.get("optimizer", "adam")
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # 学习率调度器（可选）
    scheduler = None
    if train_config.get("use_scheduler", False):
        scheduler_type = train_config.get("scheduler_type", "step")
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_config.get("scheduler_step_size", 10),
                gamma=train_config.get("scheduler_gamma", 0.1),
            )

    # 7. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程问题
    )

    # 8. 训练循环
    logger.info("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(train_loader):
            # 准备输入
            inputs = batch[input_key].to(device)
            labels = batch["_label_"].to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model({"features": inputs})

            # 计算损失
            if task_type == "classification":
                loss = loss_fn(outputs, labels)
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
            else:
                loss = loss_fn(outputs.squeeze(), labels.float())

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 记录指标
        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # 记录到实验跟踪器
        metrics = {"train_loss": avg_loss}
        if task_type == "classification":
            acc = epoch_correct / epoch_total
            history["train_acc"].append(acc)
            metrics["train_acc"] = acc
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # 自动记录指标到 MLflow 和 TensorBoard
        experiment_tracker.log_metrics(metrics, step=epoch)

    # 9. 保存模型
    if output_dir is None:
        output_dir = Path(orchestrator.config.get("output_dir", "./outputs"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    # 保存模型（使用 PyTorch 的保存方式）
    torch.save(model.state_dict(), str(model_path))
    logger.info(f"Model saved to {model_path}")

    # 10. 保存训练历史
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # 11. 保存配置文件（用于 artifact）
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # 12. 结束实验 run（自动保存 artifacts）
    experiment_tracker.end_run(
        logs={
            "model_path": str(model_path),
            "config_path": str(config_path),
            "history_path": str(history_path),
        }
    )

    # 13. 返回结果
    results = {
        "experiment_name": experiment_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": history,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
    }

    logger.info("Training completed!")
    return results
