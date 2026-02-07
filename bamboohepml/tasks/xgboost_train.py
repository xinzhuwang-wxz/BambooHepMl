"""
XGBoost training task.

Provides a separate code path for XGBoost models, which use their own
training loop (model.fit()) on numpy arrays rather than PyTorch's Trainer.

Integrates with the same PipelineOrchestrator and MLflow naming conventions
as the PyTorch training path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import logger


def _dataset_to_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert an HEPDataset (IterableDataset) to numpy (X, y) arrays.

    Iterates through the dataset once, collecting the 'event' features
    and '_label_' targets into contiguous numpy arrays.

    Args:
        dataset: An HEPDataset instance.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    X_parts = []
    y_parts = []

    for sample in dataset:
        if "event" in sample:
            x = sample["event"]
            # Convert to numpy if tensor
            if hasattr(x, "numpy"):
                x = x.numpy()
            X_parts.append(x)

        if "_label_" in sample:
            label = sample["_label_"]
            if hasattr(label, "numpy"):
                label = label.numpy()
            elif hasattr(label, "item"):
                label = label.item()
            y_parts.append(label)

    if not X_parts:
        raise ValueError("No 'event' features found in dataset samples")
    if not y_parts:
        raise ValueError("No '_label_' found in dataset samples")

    X = np.stack(X_parts, axis=0) if X_parts[0].ndim >= 1 else np.array(X_parts)
    y = np.array(y_parts)

    return X, y


def train_xgboost_task(
    orchestrator,
    train_dataset,
    val_dataset,
    task_type: str,
    effective_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    output_dir: str | None = None,
    num_epochs: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train an XGBoost model using the orchestrator's data pipeline.

    Args:
        orchestrator: Initialized PipelineOrchestrator (with data already set up).
        train_dataset: Training HEPDataset.
        val_dataset: Validation HEPDataset (can be None).
        task_type: 'classification' or 'regression'.
        effective_experiment_name: MLflow experiment name.
        mlflow_run_name: MLflow run name.
        output_dir: Output directory for model and artifacts.
        num_epochs: Used as n_estimators for xgboost (default 100).
        seed: Random seed.

    Returns:
        Dict with training results and metrics.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required for model_type='xgboost'. " "Install with: pip install xgboost")

    # Convert datasets to numpy
    logger.info("Converting training data to numpy arrays...")
    X_train, y_train = _dataset_to_numpy(train_dataset)
    logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")

    X_val, y_val = None, None
    if val_dataset is not None:
        logger.info("Converting validation data to numpy arrays...")
        X_val, y_val = _dataset_to_numpy(val_dataset)
        logger.info(f"Validation data: X={X_val.shape}, y={y_val.shape}")

    n_estimators = num_epochs if num_epochs is not None else 100

    # Create and train model
    if task_type == "classification":
        num_classes = len(np.unique(y_train))
        logger.info(f"Training XGBClassifier: {num_classes} classes, {n_estimators} estimators")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=seed,
            use_label_encoder=False,
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Evaluate
        y_pred = model.predict(X_val if X_val is not None else X_train)
        y_true = y_val if y_val is not None else y_train

        try:
            from sklearn.metrics import accuracy_score, f1_score

            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="weighted")),
            }
        except ImportError:
            # Fallback: compute accuracy manually
            metrics = {
                "accuracy": float(np.mean(y_true == y_pred)),
            }
    else:
        logger.info(f"Training XGBRegressor: {n_estimators} estimators")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=seed,
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Evaluate
        y_pred = model.predict(X_val if X_val is not None else X_train)
        y_true = y_val if y_val is not None else y_train

        mse = float(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": r2,
        }

    logger.info(f"XGBoost metrics: {metrics}")

    # Save model
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / "xgb_model.json"
        model.save_model(str(model_file))
        logger.info(f"XGBoost model saved to {model_file}")

        # Save pipeline state and metadata
        if orchestrator and orchestrator.pipeline_state:
            orchestrator.save_pipeline_state(output_path / "pipeline_state.json")

    # MLflow logging
    _log_to_mlflow(
        experiment_name=effective_experiment_name,
        run_name=mlflow_run_name,
        metrics=metrics,
        params={
            "model_type": "xgboost",
            "task_type": task_type,
            "n_estimators": n_estimators,
            "max_depth": 5,
            "learning_rate": 0.1,
            "seed": seed,
            "train_samples": X_train.shape[0],
            "features": X_train.shape[1],
        },
        model_path=str(output_path / "xgb_model.json") if output_dir else None,
    )

    return {"metrics": metrics, "model": model}


def _log_to_mlflow(
    experiment_name: str | None,
    run_name: str | None,
    metrics: dict[str, float],
    params: dict[str, Any],
    model_path: str | None = None,
) -> None:
    """Log XGBoost results to MLflow.

    Uses MLflow's Python API directly (not via Callback, since XGBoost
    doesn't use the PyTorch Trainer lifecycle).
    """
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed, skipping MLflow logging")
        return

    try:
        # Try to get tracking URI from config
        try:
            from ..config import MLFLOW_TRACKING_URI

            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        except (ImportError, AttributeError):
            pass

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # Log params (convert all to strings for MLflow)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)

            if model_path and os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")

        logger.info(f"MLflow: logged to experiment='{experiment_name}', run='{run_name}'")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
