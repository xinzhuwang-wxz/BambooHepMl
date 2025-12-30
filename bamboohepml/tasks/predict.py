"""
预测任务

支持：
- 批量预测
- 概率预测
- 结果保存（JSON、ROOT、Parquet）
- 包含标签和观察变量
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import logger
from ..data.fileio import write_root
from ..engine import Predictor
from ..models import get_model
from ..pipeline import PipelineOrchestrator


def predict_task(
    pipeline_config_path: str,
    model_path: str,
    output_path: str | None = None,
    batch_size: int = 32,
    return_probabilities: bool = False,
) -> list[dict[str, Any]]:
    """
    预测任务主函数。

    Args:
        pipeline_config_path: pipeline.yaml 路径
        model_path: 模型文件路径
        output_path: 输出文件路径（可选）
        batch_size: 批次大小
        return_probabilities: 是否返回概率

    Returns:
        预测结果列表
    """
    logger.info("=" * 80)
    logger.info("Prediction Task")
    logger.info("=" * 80)

    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    # 2. 设置数据（用于预测）
    logger.info("Setting up data system...")
    dataset = orchestrator.setup_data()
    dataset.for_training = False  # 预测模式
    dataset.shuffle = False

    # 3. 加载模型
    logger.info(f"Loading model from {model_path}...")

    # 从 pipeline 配置获取模型配置
    model_config = orchestrator.get_model_config()
    model_name = model_config.get("name")
    model_params = model_config.get("params", {})

    # 从数据中推断输入维度
    sample = next(iter(dataset))
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

    model_params["input_dim"] = input_dim
    model = get_model(model_name, **model_params)

    # 加载权重
    # 支持两种格式：
    # 1. 直接保存的 state_dict（用于推理）
    # 2. checkpoint 格式（包含 model_state_dict, optimizer_state_dict 等）
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # checkpoint 格式
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 直接 state_dict 格式
        model.load_state_dict(checkpoint)

    # 使用 Predictor（会自动处理设备）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictor = Predictor(model, device=device)

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 5. 执行预测并收集数据
    logger.info("Running predictions...")

    # 收集预测结果、标签和观察变量
    all_predictions = []
    all_probabilities = []
    all_labels = {}
    all_observers = {}

    # 获取数据配置以确定标签和观察变量名称
    data_config = orchestrator.data_config
    label_names = data_config.label_names if data_config else []
    z_variables = data_config.z_variables if data_config else []

    with torch.no_grad():
        for batch in dataloader:
            # 将 batch 移动到设备（直接传递完整 batch 给 model）
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(predictor.device)
                else:
                    batch_on_device[k] = v

            # 前向传播（直接传递完整 batch）
            outputs = predictor.model(batch_on_device)

            # 获取预测
            if outputs.shape[1] > 1:  # 分类任务
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                if return_probabilities:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probabilities.append(probs)
            else:  # 回归任务
                predictions = outputs.squeeze().cpu().numpy()

            all_predictions.append(predictions)

            # 收集标签（如果存在）
            if "_label_" in batch:
                label_data = batch["_label_"].cpu().numpy()
                label_key = label_names[0] if label_names else "_label_"
                if label_key not in all_labels:
                    all_labels[label_key] = []
                all_labels[label_key].append(label_data)

            # 收集观察变量
            for obs_name in z_variables:
                if obs_name in batch:
                    obs_data = batch[obs_name]
                    # 处理不同类型的观察变量
                    if isinstance(obs_data, torch.Tensor):
                        obs_data = obs_data.cpu().numpy()
                    elif isinstance(obs_data, ak.Array):
                        obs_data = ak.to_numpy(obs_data)

                    if obs_name not in all_observers:
                        all_observers[obs_name] = []
                    all_observers[obs_name].append(obs_data)

    # 合并结果
    all_predictions = np.concatenate(all_predictions)
    if all_probabilities:
        all_probabilities = np.concatenate(all_probabilities)

    for key in all_labels:
        all_labels[key] = np.concatenate(all_labels[key])

    for key in all_observers:
        all_observers[key] = (
            np.concatenate(all_observers[key]) if isinstance(all_observers[key][0], np.ndarray) else ak.concatenate(all_observers[key])
        )

    # 6. 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ext = output_path.suffix.lower()

        if ext == ".root":
            # 保存为 ROOT 文件（类似 weaver）
            output_dict = {}

            # 添加预测结果
            # 判断任务类型：如果输出维度 > 1，通常是分类任务；否则是回归任务
            if len(all_predictions.shape) == 1 and (all_probabilities is not None and len(all_probabilities) > 0 and all_probabilities.shape[1] > 1):
                # 分类任务：添加每个类别的分数
                if data_config and data_config.label_type == "simple" and data_config.label_value:
                    # 多分类任务：为每个类别添加分数
                    for idx, label_name in enumerate(data_config.label_value):
                        # 添加分数（概率或 logits）
                        if all_probabilities is not None and len(all_probabilities) > 0:
                            output_dict[f"score_{label_name}"] = all_probabilities[:, idx]
                        else:
                            # 如果没有概率，使用预测结果生成 one-hot
                            output_dict[f"pred_{label_name}"] = all_predictions == idx

                    # 添加预测类别索引
                    output_dict["prediction"] = all_predictions

                    # 如果有标签，还原 one-hot 编码的标签
                    if len(all_labels) > 0:
                        label_key = list(all_labels.keys())[0]
                        for idx, label_name in enumerate(data_config.label_value):
                            output_dict[label_name] = all_labels[label_key] == idx
                else:
                    # 通用分类任务（类别名称未知）
                    output_dict["prediction"] = all_predictions
                    if all_probabilities is not None and len(all_probabilities) > 0:
                        for idx in range(all_probabilities.shape[1]):
                            output_dict[f"score_class_{idx}"] = all_probabilities[:, idx]
            else:
                # 回归任务：输出单个连续值
                output_dict["prediction"] = all_predictions

            # 添加标签
            output_dict.update(all_labels)

            # 添加观察变量
            for key, value in all_observers.items():
                output_dict[key] = value

            # 写入 ROOT 文件
            write_root(str(output_path), ak.Array(output_dict), treename="Events")
            logger.info(f"Predictions saved to ROOT file: {output_path}")

        elif ext == ".parquet":
            # 保存为 Parquet 文件
            output_dict = {"prediction": all_predictions}
            if all_probabilities is not None and len(all_probabilities) > 0:
                for idx in range(all_probabilities.shape[1]):
                    output_dict[f"score_class_{idx}"] = all_probabilities[:, idx]
            output_dict.update(all_labels)
            output_dict.update(all_observers)

            ak.to_parquet(ak.Array(output_dict), str(output_path), compression="LZ4", compression_level=4)
            logger.info(f"Predictions saved to Parquet file: {output_path}")

        else:
            # 默认保存为 JSON（向后兼容）
            results = []
            for i in range(len(all_predictions)):
                result = {"prediction": int(all_predictions[i]) if all_predictions.dtype == np.int64 else float(all_predictions[i])}
                if all_probabilities is not None and len(all_probabilities) > 0:
                    result["probabilities"] = all_probabilities[i].tolist()
                # 添加标签（如果有）
                for key, value in all_labels.items():
                    result[key] = int(value[i]) if value.dtype == np.int64 else float(value[i])
                # 添加观察变量（如果有）
                for key, value in all_observers.items():
                    if isinstance(value, ak.Array):
                        result[key] = ak.to_list(value[i])
                    else:
                        result[key] = float(value[i]) if value.ndim == 1 else value[i].tolist()
                results.append(result)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Predictions saved to JSON file: {output_path}")

    logger.info(f"Prediction completed! Total samples: {len(all_predictions)}")

    # 返回结果（向后兼容）
    results = []
    for i in range(len(all_predictions)):
        result = {"prediction": int(all_predictions[i]) if all_predictions.dtype == np.int64 else float(all_predictions[i])}
        if all_probabilities is not None and len(all_probabilities) > 0:
            result["probabilities"] = all_probabilities[i].tolist()
        results.append(result)

    return results
