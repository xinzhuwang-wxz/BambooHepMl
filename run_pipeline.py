#!/usr/bin/env python
"""BambooHepMl — Unified EDM4hep Pipeline Runner.

Runs one or more experiment types on real EDM4hep ROOT data.
Four independent experiment paths are supported:

    classification + Torch MLP
    classification + XGBoost
    regression     + Torch MLP
    regression     + XGBoost

Usage examples:
    # Run all 4 experiment types (each with 2 runs):
    python run_pipeline.py --all

    # Run a single experiment:
    python run_pipeline.py --task classification --model torch

    # Inference-only mode (requires a trained model):
    python run_pipeline.py --task classification --model torch --predict-only \\
        --model-path outputs/edm4hep/classification_torch/models/best_model.pt

    # Override device:
    python run_pipeline.py --task regression --model xgboost --device cpu

Experiment naming convention:
    experiment name = {task_type}_{model_type}_edm4hep
    run name        = {model_type}_{YYYYMMdd_HHmmss}

Output structure:
    outputs/edm4hep/
    ├── classification_torch/
    │   ├── models/
    │   ├── plots/
    │   ├── predictions/
    │   └── configs/
    ├── classification_xgboost/
    ├── regression_torch/
    └── regression_xgboost/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from bamboohepml.data.config import DataConfig
from bamboohepml.data.dataset import HEPDataset
from bamboohepml.data.features.expression import ExpressionEngine
from bamboohepml.data.features.feature_graph import FeatureGraph
from bamboohepml.data.sources.base import DataSourceConfig
from bamboohepml.data.sources.root_source import ROOTDataSource
from bamboohepml.engine.trainer import Trainer
from bamboohepml.models.common.mlp import MLPClassifier, MLPRegressor
from bamboohepml.utils import collate_fn, set_seeds


# ─── Constants ───────────────────────────────────────────────────────────
FEATURES_CONFIG = "configs/features_edm4hep.yaml"
CLF_DATA_CONFIG = "configs/data_edm4hep_classification.yaml"
REG_DATA_CONFIG = "configs/data_edm4hep_regression.yaml"
PIPELINE_CONFIG = "configs/pipeline_edm4hep.yaml"
DATA_DIR = "testdata"
NUM_CLASSES = 3
CLASS_NAMES = ["3 GeV", "5 GeV", "7 GeV"]


# ═════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════

def resolve_device(requested: str | None = None) -> torch.device:
    """Return the best available device or honour an explicit request.

    Args:
        requested: One of ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None``
            (auto-detect).

    Returns:
        A ``torch.device`` instance.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def timestamp() -> str:
    """Return a compact timestamp string for run naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_pipeline_config(path: str = PIPELINE_CONFIG) -> dict:
    """Load and return the unified pipeline YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def discover_root_files(data_dir: str = DATA_DIR) -> list[str]:
    """Find ROOT files using glob.  Supports directory or wildcard paths.

    Args:
        data_dir: A directory path, a glob pattern, or a comma-separated
            list of paths.

    Returns:
        Sorted list of absolute file paths.
    """
    all_files: list[str] = []

    for pattern in data_dir.split(","):
        pattern = pattern.strip()
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*.root")
        resolved = sorted(glob.glob(pattern))
        all_files.extend(resolved)

    if not all_files:
        raise FileNotFoundError(
            f"No ROOT files found for pattern(s): {data_dir}"
        )
    return sorted(set(all_files))


# ═════════════════════════════════════════════════════════════════════════
# Data pipeline (shared across all experiment types)
# ═════════════════════════════════════════════════════════════════════════

def build_feature_graph(
    features_config: str = FEATURES_CONFIG,
) -> FeatureGraph:
    """Create a FeatureGraph from the features YAML config.

    Args:
        features_config: Path to the features YAML.

    Returns:
        An un-fitted FeatureGraph.
    """
    engine = ExpressionEngine()
    fg = FeatureGraph.from_yaml(features_config, expression_engine=engine)
    print(f"  FeatureGraph: {len(fg.nodes)} features loaded from {features_config}")
    return fg


def create_data_sources(
    root_files: list[str],
    data_conf: dict,
    train_range: tuple = (0.0, 0.7),
    val_range: tuple = (0.7, 0.85),
    test_range: tuple = (0.85, 1.0),
) -> tuple[ROOTDataSource, ROOTDataSource, ROOTDataSource]:
    """Create train / val / test data sources.

    Args:
        root_files: Resolved ROOT file paths.
        data_conf: Parsed data YAML dict (for treename, file_magic, etc.).
        train_range: Fractional event range for training split.
        val_range: Fractional event range for validation split.
        test_range: Fractional event range for test split.

    Returns:
        Tuple of (train_source, val_source, test_source).
    """
    common = dict(
        treename=data_conf.get("treename", "events"),
        branch_magic=data_conf.get("branch_magic"),
        file_magic=data_conf.get("file_magic"),
    )
    train = ROOTDataSource(DataSourceConfig(
        file_paths=root_files, load_range=train_range, **common
    ))
    val = ROOTDataSource(DataSourceConfig(
        file_paths=root_files, load_range=val_range, **common
    ))
    test = ROOTDataSource(DataSourceConfig(
        file_paths=root_files, load_range=test_range, **common
    ))
    return train, val, test


def fit_feature_graph(
    fg: FeatureGraph,
    train_source: ROOTDataSource,
) -> None:
    """Fit the FeatureGraph normalizers on training data (in-place).

    Args:
        fg: The FeatureGraph to fit.
        train_source: Training data source.
    """
    source_branches = list(fg.get_source_branches())
    print(f"  Fitting normalizers on training data ({len(source_branches)} source branches)...")
    fit_data = train_source.load_branches(source_branches)
    fg.fit(fit_data)
    print(f"  Fitted on {len(fit_data)} events.")

    # Quick sanity check.
    batch = fg.build_batch(fit_data[:10])
    ev = batch["event"]
    print(f"  Sanity check — event tensor shape: {ev.shape}, "
          f"range: [{ev.min():.3f}, {ev.max():.3f}]")


def build_dataloaders(
    data_config: DataConfig,
    fg: FeatureGraph,
    train_source: ROOTDataSource,
    val_source: ROOTDataSource,
    test_source: ROOTDataSource,
    batch_size: int = 128,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build PyTorch DataLoaders for train / val / test.

    Args:
        data_config: The DataConfig instance (labels / selection only).
        fg: Fitted FeatureGraph.
        train_source, val_source, test_source: Data sources.
        batch_size: Batch size.
        shuffle_train: Whether to shuffle training data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    loaders = []
    for src, training, shuffle in [
        (train_source, True, shuffle_train),
        (val_source, True, False),
        (test_source, False, False),
    ]:
        ds = HEPDataset(
            data_source=src,
            data_config=data_config,
            feature_graph=fg,
            for_training=training,
            shuffle=shuffle,
            reweight=False,
        )
        loaders.append(
            DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        )
    return tuple(loaders)


# ═════════════════════════════════════════════════════════════════════════
# Torch training helpers
# ═════════════════════════════════════════════════════════════════════════

def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    num_epochs: int,
    lr: float,
    save_dir: str,
) -> tuple[dict, dict]:
    """Train a Torch model and evaluate on the test set.

    Args:
        model: The PyTorch model.
        train_loader, val_loader, test_loader: Data loaders.
        task_type: ``"classification"`` or ``"regression"``.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        save_dir: Directory for saving model checkpoints.

    Returns:
        Tuple of (training_result, test_metrics).
    """
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        task_type=task_type,
    )

    t0 = time.time()
    result = trainer.fit(
        num_epochs=num_epochs,
        save_dir=os.path.join(save_dir, "models"),
        save_best=True,
        monitor="loss",
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s "
          f"(best epoch {result['best_epoch']}, "
          f"best val loss {result['best_value']:.4f})")

    test_metrics = trainer.test()
    print("  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    return result, test_metrics


# ═════════════════════════════════════════════════════════════════════════
# XGBoost helpers
# ═════════════════════════════════════════════════════════════════════════

def build_numpy_data(
    fg: FeatureGraph,
    root_files: list[str],
    data_conf: dict,
    task_type: str,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Build numpy feature matrices using the shared FeatureGraph.

    Args:
        fg: Fitted FeatureGraph.
        root_files: ROOT file paths.
        data_conf: Data config dict.
        task_type: ``"classification"`` or ``"regression"``.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        Tuple of (X, y).
    """
    range_map = {"train": (0.0, 0.7), "val": (0.7, 0.85), "test": (0.85, 1.0)}
    lr = range_map[split]

    common = dict(
        treename=data_conf.get("treename", "events"),
        branch_magic=data_conf.get("branch_magic"),
        file_magic=data_conf.get("file_magic"),
    )

    source = ROOTDataSource(DataSourceConfig(
        file_paths=root_files, load_range=lr, **common
    ))

    # Determine what branches to load.
    source_branches = list(fg.get_source_branches())
    if task_type == "classification":
        label_branches = ["is_3GeV", "is_5GeV", "is_7GeV"]
    else:
        label_branches = ["_IncidentEnergy"]

    raw_data = source.load_branches(source_branches + label_branches)
    batch = fg.build_batch(raw_data)
    X = batch["event"].numpy()

    if task_type == "classification":
        labels_stack = np.stack(
            [ak.to_numpy(raw_data[lb]) for lb in label_branches], axis=1
        )
        y = np.argmax(labels_stack, axis=1)
    else:
        y = ak.to_numpy(raw_data["_IncidentEnergy"]).astype(np.float32)

    return X, y


def train_xgboost(
    fg: FeatureGraph,
    root_files: list[str],
    data_conf: dict,
    task_type: str,
    seed: int = 42,
    save_dir: str = "outputs",
) -> tuple[Any, dict]:
    """Train an XGBoost model and return (model, test_metrics).

    Args:
        fg: Fitted FeatureGraph.
        root_files: ROOT file paths.
        data_conf: Data config dict.
        task_type: ``"classification"`` or ``"regression"``.
        seed: Random seed.
        save_dir: Output directory.

    Returns:
        Tuple of (xgb_model, test_metrics_dict).
    """
    import xgboost as xgb

    # Build numpy datasets.
    datasets = {}
    for split in ["train", "val", "test"]:
        X, y = build_numpy_data(fg, root_files, data_conf, task_type, split)
        datasets[split] = (X, y)
        dist = np.bincount(y.astype(int), minlength=NUM_CLASSES).tolist() if task_type == "classification" else f"mean={y.mean():.2f}"
        print(f"  {split}: X={X.shape}, y dist={dist}")

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, y_test = datasets["test"]

    if task_type == "classification":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=seed,
            use_label_encoder=False,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=seed,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_test - y_pred))),
            "r2": r2,
        }

    # Save model.
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "xgb_model.json"))

    print("  XGBoost test metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════

def plot_classification(
    model: nn.Module,
    test_loader: DataLoader,
    history: dict,
    save_dir: str,
) -> None:
    """Generate and save classification evaluation plots."""
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            out = model(bd)
            probs = torch.softmax(out, dim=1)
            all_preds.append(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.append(batch["_label_"].numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Training history.
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training History")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Confusion matrix.
    from sklearn.metrics import confusion_matrix
    ax = axes[0, 1]
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)

    # ROC curves.
    from sklearn.metrics import auc, roc_curve
    ax = axes[1, 0]
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Score distributions.
    ax = axes[1, 1]
    for i, name in enumerate(CLASS_NAMES):
        s = probs[:, i]
        ax.hist(s[labels == i], bins=30, alpha=0.5, density=True,
                label=f"{name} (sig)", histtype="stepfilled")
        ax.hist(s[labels != i], bins=30, alpha=0.3, density=True,
                label=f"{name} (bkg)", histtype="step", linestyle="--")
    ax.set(xlabel="Score", ylabel="Density", title="Score Distributions")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "classification_evaluation.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved classification plots to {path}")


def plot_regression(
    model: nn.Module,
    test_loader: DataLoader,
    history: dict,
    save_dir: str,
) -> None:
    """Generate and save regression evaluation plots."""
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            out = model(bd)
            all_preds.append(out.squeeze().cpu().numpy())
            all_targets.append(batch["_label_"].numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    residuals = preds - targets

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="Loss (MSE)", title="Training History")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(targets, preds, alpha=0.3, s=10)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, "r--", label="y=x")
    ax.set(xlabel="True [GeV]", ylabel="Predicted [GeV]", title="Pred vs True")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set(xlabel="Residual [GeV]", ylabel="Count",
           title=f"Residuals (bias={np.mean(residuals):.3f}, "
                 f"res={np.std(residuals):.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    unique_e = sorted(np.unique(targets))
    res_vals = [np.std(preds[targets == e] - e) for e in unique_e]
    ax.bar(range(len(unique_e)), res_vals,
           tick_label=[f"{e:.0f}" for e in unique_e],
           alpha=0.7, color="steelblue", edgecolor="black")
    ax.set(xlabel="True Energy [GeV]", ylabel="Resolution [GeV]",
           title="Resolution vs Energy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "regression_evaluation.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved regression plots to {path}")


def plot_xgboost_importance(
    model: Any,
    fg: FeatureGraph,
    save_dir: str,
    task_type: str,
) -> None:
    """Plot XGBoost feature importance."""
    importances = model.feature_importances_
    feature_names = list(fg.nodes.keys())
    sorted_idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="steelblue")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"XGBoost Feature Importance — {task_type} (Top 15)")
    ax.invert_yaxis()
    plt.tight_layout()
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "xgboost_feature_importance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved feature importance plot to {path}")


# ═════════════════════════════════════════════════════════════════════════
# ROOT output
# ═════════════════════════════════════════════════════════════════════════

def write_root_predictions(
    fg: FeatureGraph,
    root_files: list[str],
    data_conf: dict,
    clf_model: nn.Module | None,
    reg_model: nn.Module | None,
    save_dir: str,
    device: torch.device,
) -> None:
    """Write predictions to a ROOT TTree file.

    Args:
        fg: Fitted FeatureGraph.
        root_files: ROOT file paths.
        data_conf: Data config dict (for file_magic etc.).
        clf_model: Classification model (optional).
        reg_model: Regression model (optional).
        save_dir: Output directory.
        device: Torch device.
    """
    import uproot

    common = dict(
        treename=data_conf.get("treename", "events"),
        branch_magic=data_conf.get("branch_magic"),
        file_magic=data_conf.get("file_magic"),
    )
    test_source = ROOTDataSource(DataSourceConfig(
        file_paths=root_files, load_range=(0.85, 1.0), **common
    ))

    source_branches = list(fg.get_source_branches())
    extra = ["is_3GeV", "is_5GeV", "is_7GeV", "_IncidentEnergy"]
    raw = test_source.load_branches(source_branches + extra)
    batch = fg.build_batch(raw)
    event_tensor = batch["event"]

    output_data: dict[str, np.ndarray] = {
        "event_id": np.arange(len(event_tensor), dtype=np.int32),
    }

    if clf_model is not None:
        clf_model.eval()
        with torch.no_grad():
            out = clf_model({"event": event_tensor.to(device)})
            probs = torch.softmax(out, dim=1).cpu().numpy()
            labels = torch.argmax(out, dim=1).cpu().numpy()
        output_data["clf_label"] = labels.astype(np.int32)
        for i, cname in enumerate(CLASS_NAMES):
            output_data[f"score_{cname.replace(' ', '_')}"] = probs[:, i].astype(np.float32)

    if reg_model is not None:
        reg_model.eval()
        with torch.no_grad():
            pred_e = reg_model({"event": event_tensor.to(device)}).squeeze().cpu().numpy()
        true_e = ak.to_numpy(raw["_IncidentEnergy"]).astype(np.float32)
        output_data["true_IncidentEnergy"] = true_e
        output_data["pred_IncidentEnergy"] = pred_e.astype(np.float32)
        output_data["residual"] = (pred_e - true_e).astype(np.float32)

    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, "predictions.root")
    with uproot.recreate(out_path) as f:
        f["pred"] = output_data

    print(f"  ROOT predictions written to {out_path} "
          f"({len(event_tensor)} events, {len(output_data)} branches)")


# ═════════════════════════════════════════════════════════════════════════
# MLflow logging
# ═════════════════════════════════════════════════════════════════════════

def log_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts_dir: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Log an experiment run to MLflow.

    Args:
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        params: Parameter dict to log.
        metrics: Metric dict to log.
        artifacts_dir: Directory of artifacts to log (optional).
        tracking_uri: MLflow tracking URI (None = local default).
    """
    try:
        import mlflow
    except ImportError:
        print("  WARNING: mlflow not installed, skipping tracking.")
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts_dir and os.path.isdir(artifacts_dir):
            for fname in os.listdir(artifacts_dir):
                fpath = os.path.join(artifacts_dir, fname)
                if os.path.isfile(fpath):
                    mlflow.log_artifact(fpath)
        print(f"  MLflow: logged to experiment='{experiment_name}', "
              f"run='{run_name}'")


# ═════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═════════════════════════════════════════════════════════════════════════

def run_experiment(
    task_type: str,
    model_type: str,
    fg: FeatureGraph,
    root_files: list[str],
    pipeline_cfg: dict,
    device: torch.device,
    run_index: int = 1,
) -> dict:
    """Run a single experiment (train + evaluate + plot + log).

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        model_type: ``"torch"`` or ``"xgboost"``.
        fg: Fitted FeatureGraph.
        root_files: ROOT file paths.
        pipeline_cfg: Pipeline config dict.
        device: Torch device.
        run_index: Run number (for naming).

    Returns:
        Dict of test metrics.
    """
    # Naming.
    experiment_name = f"{task_type}_{model_type}_edm4hep"
    run_name = f"{model_type}_{timestamp()}_run{run_index}"
    save_dir = os.path.join(
        pipeline_cfg.get("output", {}).get("base_dir", "outputs/edm4hep"),
        f"{task_type}_{model_type}",
    )
    for sub in ["models", "plots", "predictions", "configs"]:
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {experiment_name}  |  RUN: {run_name}")
    print(f"{'=' * 70}")

    # Select data config.
    data_config_path = (
        CLF_DATA_CONFIG if task_type == "classification" else REG_DATA_CONFIG
    )
    with open(data_config_path) as f:
        data_conf = yaml.safe_load(f)
    data_config = DataConfig.load(data_config_path)

    # Training settings.
    train_cfg = pipeline_cfg.get("training", {})
    seed = train_cfg.get("seed", 42) + run_index  # Vary seed per run.
    set_seeds(seed)
    num_features = len(fg.nodes)
    batch_size = train_cfg.get("batch_size", 128)
    num_epochs = train_cfg.get("num_epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)

    # Create data sources.
    data_cfg = pipeline_cfg.get("data", {})
    train_range = tuple(data_cfg.get("train_range", [0.0, 0.7]))
    val_range = tuple(data_cfg.get("val_range", [0.7, 0.85]))
    test_range = tuple(data_cfg.get("test_range", [0.85, 1.0]))
    train_src, val_src, test_src = create_data_sources(
        root_files, data_conf, train_range, val_range, test_range
    )

    # ─── Train ──────────────────────────────────────────────────────
    metrics: dict = {}

    if model_type == "torch":
        # Build data loaders.
        train_loader, val_loader, test_loader = build_dataloaders(
            data_config, fg, train_src, val_src, test_src,
            batch_size=batch_size,
        )

        # Create model.
        model_cfg = pipeline_cfg.get("model", {})
        if task_type == "classification":
            model = MLPClassifier(
                event_input_dim=num_features,
                hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
                num_classes=NUM_CLASSES,
                dropout=model_cfg.get("dropout", 0.1),
                activation=model_cfg.get("activation", "relu"),
                batch_norm=model_cfg.get("batch_norm", True),
                embed_dim=model_cfg.get("embed_dim", 64),
            )
        else:
            model = MLPRegressor(
                event_input_dim=num_features,
                hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
                num_outputs=1,
                dropout=model_cfg.get("dropout", 0.1),
                activation=model_cfg.get("activation", "relu"),
                batch_norm=model_cfg.get("batch_norm", True),
                embed_dim=model_cfg.get("embed_dim", 64),
            )

        print(f"  Model: {model.__class__.__name__} "
              f"(params={sum(p.numel() for p in model.parameters()):,})")

        result, metrics = train_torch_model(
            model, train_loader, val_loader, test_loader,
            task_type, num_epochs, lr, save_dir,
        )

        # Plots.
        if task_type == "classification":
            plot_classification(model, test_loader, result["history"], save_dir)
        else:
            plot_regression(model, test_loader, result["history"], save_dir)

    elif model_type == "xgboost":
        xgb_model, metrics = train_xgboost(
            fg, root_files, data_conf, task_type, seed=seed, save_dir=save_dir,
        )
        plot_xgboost_importance(xgb_model, fg, save_dir, task_type)

    # ─── MLflow ─────────────────────────────────────────────────────
    mlflow_cfg = pipeline_cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled", True):
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri is None:
            base_dir = pipeline_cfg.get("output", {}).get("base_dir", "outputs/edm4hep")
            tracking_uri = f"file://{os.path.abspath(os.path.join(base_dir, 'mlruns'))}"
        log_to_mlflow(
            experiment_name=experiment_name,
            run_name=run_name,
            params={
                "task_type": task_type,
                "model_type": model_type,
                "seed": seed,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_features": num_features,
                "device": str(device),
            },
            metrics=metrics,
            artifacts_dir=os.path.join(save_dir, "plots"),
            tracking_uri=tracking_uri,
        )

    # Save feature graph state.
    fg_state_path = os.path.join(save_dir, "configs", "feature_graph_state.yaml")
    with open(fg_state_path, "w") as f:
        yaml.dump(fg.export_state(), f, default_flow_style=False)
    print(f"  FeatureGraph state saved to {fg_state_path}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════
# Inference-only mode
# ═════════════════════════════════════════════════════════════════════════

def run_inference(
    task_type: str,
    model_path: str,
    fg: FeatureGraph,
    root_files: list[str],
    pipeline_cfg: dict,
    device: torch.device,
) -> None:
    """Run inference with a pre-trained model.

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        model_path: Path to the saved model state dict (.pt).
        fg: Fitted FeatureGraph.
        root_files: ROOT file paths.
        pipeline_cfg: Pipeline config dict.
        device: Torch device.
    """
    print(f"\n{'=' * 70}")
    print(f"INFERENCE: {task_type} using {model_path}")
    print(f"{'=' * 70}")

    num_features = len(fg.nodes)
    model_cfg = pipeline_cfg.get("model", {})

    if task_type == "classification":
        model = MLPClassifier(
            event_input_dim=num_features,
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            num_classes=NUM_CLASSES,
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            batch_norm=model_cfg.get("batch_norm", True),
            embed_dim=model_cfg.get("embed_dim", 64),
        )
    else:
        model = MLPRegressor(
            event_input_dim=num_features,
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            num_outputs=1,
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
            batch_norm=model_cfg.get("batch_norm", True),
            embed_dim=model_cfg.get("embed_dim", 64),
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    data_config_path = (
        CLF_DATA_CONFIG if task_type == "classification" else REG_DATA_CONFIG
    )
    with open(data_config_path) as f:
        data_conf = yaml.safe_load(f)

    save_dir = os.path.join(
        pipeline_cfg.get("output", {}).get("base_dir", "outputs/edm4hep"),
        f"{task_type}_inference",
    )

    clf_model = model if task_type == "classification" else None
    reg_model = model if task_type == "regression" else None

    write_root_predictions(
        fg, root_files, data_conf, clf_model, reg_model, save_dir, device
    )
    print("  Inference complete.")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="BambooHepMl EDM4hep Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task", choices=["classification", "regression"],
        help="Task type (required unless --all is used).",
    )
    p.add_argument(
        "--model", choices=["torch", "xgboost"],
        help="Model type (required unless --all is used).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run all 4 experiment types (clf×torch, clf×xgb, reg×torch, reg×xgb), "
             "each with --runs runs.",
    )
    p.add_argument(
        "--runs", type=int, default=2,
        help="Number of runs per experiment (default: 2).",
    )
    p.add_argument(
        "--device", choices=["cuda", "mps", "cpu"],
        help="Force a specific device (default: auto-detect).",
    )
    p.add_argument(
        "--data-dir", default=DATA_DIR,
        help="ROOT file directory or glob pattern (default: testdata).",
    )
    p.add_argument(
        "--predict-only", action="store_true",
        help="Inference-only mode (requires --model-path).",
    )
    p.add_argument(
        "--model-path",
        help="Path to a trained model .pt file (for --predict-only).",
    )
    p.add_argument(
        "--config", default=PIPELINE_CONFIG,
        help="Pipeline config YAML path.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the pipeline runner."""
    args = parse_args()

    print("=" * 70)
    print("BambooHepMl — EDM4hep Pipeline Runner")
    print("=" * 70)
    t_start = time.time()

    # Device.
    device = resolve_device(args.device)
    print(f"  Device: {device}")

    # Pipeline config.
    pipeline_cfg = load_pipeline_config(args.config)

    # Discover ROOT files (supports glob, directory, or comma-separated).
    root_files = discover_root_files(args.data_dir)
    print(f"  ROOT files: {len(root_files)} found")
    for rf in root_files:
        print(f"    {os.path.basename(rf)}")

    # Build and fit FeatureGraph (shared across all experiments).
    fg = build_feature_graph(
        pipeline_cfg.get("data", {}).get("features_config", FEATURES_CONFIG)
    )

    # Fit normalizers on training split.
    clf_data_conf_path = CLF_DATA_CONFIG
    with open(clf_data_conf_path) as f:
        clf_data_conf = yaml.safe_load(f)
    train_src, _, _ = create_data_sources(
        root_files, clf_data_conf,
        train_range=tuple(pipeline_cfg.get("data", {}).get("train_range", [0.0, 0.7])),
    )
    fit_feature_graph(fg, train_src)

    # ─── Inference-only mode ────────────────────────────────────────
    if args.predict_only:
        if not args.task or not args.model_path:
            print("ERROR: --predict-only requires --task and --model-path")
            sys.exit(1)
        run_inference(args.task, args.model_path, fg, root_files, pipeline_cfg, device)
        print(f"\nDone in {time.time() - t_start:.1f}s")
        return

    # ─── Training mode ──────────────────────────────────────────────
    if args.all:
        experiments = [
            ("classification", "torch"),
            ("classification", "xgboost"),
            ("regression", "torch"),
            ("regression", "xgboost"),
        ]
    else:
        if not args.task or not args.model:
            print("ERROR: specify --task and --model, or use --all")
            sys.exit(1)
        experiments = [(args.task, args.model)]

    all_metrics: dict[str, dict] = {}
    for task_type, model_type in experiments:
        for run_idx in range(1, args.runs + 1):
            key = f"{task_type}_{model_type}_run{run_idx}"
            try:
                metrics = run_experiment(
                    task_type, model_type, fg, root_files,
                    pipeline_cfg, device, run_index=run_idx,
                )
                all_metrics[key] = metrics
            except Exception as e:
                print(f"\n  ERROR in {key}: {e}")
                import traceback
                traceback.print_exc()
                all_metrics[key] = {"error": str(e)}

    # ─── ROOT predictions (using last torch models if available) ────
    output_cfg = pipeline_cfg.get("output", {})
    if output_cfg.get("export_root", True):
        # Try to load best models for ROOT export.
        base_dir = output_cfg.get("base_dir", "outputs/edm4hep")
        clf_pt = os.path.join(base_dir, "classification_torch", "models", "best_model.pt")
        reg_pt = os.path.join(base_dir, "regression_torch", "models", "best_model.pt")

        clf_model_for_root = None
        reg_model_for_root = None
        model_cfg = pipeline_cfg.get("model", {})
        num_features = len(fg.nodes)

        if os.path.exists(clf_pt):
            m = MLPClassifier(
                event_input_dim=num_features,
                hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
                num_classes=NUM_CLASSES,
                dropout=model_cfg.get("dropout", 0.1),
                activation=model_cfg.get("activation", "relu"),
                batch_norm=model_cfg.get("batch_norm", True),
                embed_dim=model_cfg.get("embed_dim", 64),
            )
            m.load_state_dict(torch.load(clf_pt, map_location=device))
            m.to(device)
            clf_model_for_root = m

        if os.path.exists(reg_pt):
            m = MLPRegressor(
                event_input_dim=num_features,
                hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
                num_outputs=1,
                dropout=model_cfg.get("dropout", 0.1),
                activation=model_cfg.get("activation", "relu"),
                batch_norm=model_cfg.get("batch_norm", True),
                embed_dim=model_cfg.get("embed_dim", 64),
            )
            m.load_state_dict(torch.load(reg_pt, map_location=device))
            m.to(device)
            reg_model_for_root = m

        if clf_model_for_root or reg_model_for_root:
            with open(CLF_DATA_CONFIG) as f:
                data_conf_for_root = yaml.safe_load(f)
            write_root_predictions(
                fg, root_files, data_conf_for_root,
                clf_model_for_root, reg_model_for_root,
                base_dir, device,
            )

    # ─── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Device: {device}")
    for key, m in all_metrics.items():
        summary = ", ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float))
        print(f"  {key}: {summary}")
    print()


if __name__ == "__main__":
    main()
