"""
检查任务

支持：
- 数据检查（统计信息、分布）
- 特征检查（依赖关系、计算顺序）
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import logger
from ..pipeline import PipelineOrchestrator


def inspect_task(
    pipeline_config_path: str,
    output_path: Optional[str] = None,
    num_samples: int = 1000,
    inspect_data: bool = True,
    inspect_features: bool = True,
) -> Dict[str, Any]:
    """
    检查任务主函数。

    Args:
        pipeline_config_path: pipeline.yaml 路径
        output_path: 输出文件路径（可选）
        num_samples: 检查的样本数量
        inspect_data: 是否检查数据
        inspect_features: 是否检查特征

    Returns:
        检查结果字典
    """
    logger.info("=" * 80)
    logger.info("Inspect Task")
    logger.info("=" * 80)

    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    results = {}

    # 2. 检查数据
    if inspect_data:
        logger.info("Inspecting data...")
        dataset = orchestrator.setup_data()
        dataset.for_training = False
        dataset.shuffle = False

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        # 收集统计信息
        data_stats = {
            "num_samples": 0,
            "sample_keys": [],
            "input_shapes": {},
            "label_stats": {},
        }

        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            if batch_idx == 0:
                data_stats["sample_keys"] = list(batch.keys())

            # 检查输入形状
            for key, value in batch.items():
                if key.startswith("_") and key != "_label_":
                    if key not in data_stats["input_shapes"]:
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            data_stats["input_shapes"][key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                            }

            # 检查标签统计
            if "_label_" in batch:
                labels = batch["_label_"]
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.numpy()
                else:
                    labels_np = np.array(labels)

                if "_label_" not in data_stats["label_stats"]:
                    data_stats["label_stats"]["_label_"] = {
                        "min": float(labels_np.min()),
                        "max": float(labels_np.max()),
                        "mean": float(labels_np.mean()),
                        "std": float(labels_np.std()),
                        "unique_values": np.unique(labels_np).tolist(),
                    }

            sample_count += len(batch.get("_label_", [1]))

        data_stats["num_samples"] = sample_count
        results["data"] = data_stats

        logger.info(f"Data inspection completed: {sample_count} samples")
        logger.info(f"  Sample keys: {data_stats['sample_keys']}")
        logger.info(f"  Input shapes: {data_stats['input_shapes']}")

    # 3. 检查特征
    if inspect_features:
        logger.info("Inspecting features...")
        feature_graph = orchestrator.get_feature_graph()

        if feature_graph is not None:
            feature_stats = {
                "num_features": len(feature_graph.nodes),
                "execution_order": feature_graph.get_execution_order(),
                "dependencies": {},
            }

            # 获取每个特征的依赖关系
            for node_name, node in feature_graph.nodes.items():
                feature_stats["dependencies"][node_name] = {
                    "deps": list(node.dependencies),
                    "expression": node.expression,
                }

            results["features"] = feature_stats

            logger.info(f"Feature inspection completed: {feature_stats['num_features']} features")
            logger.info(f"  Execution order: {feature_stats['execution_order'][:5]}...")
        else:
            logger.info("No feature graph found, skipping feature inspection")
            results["features"] = None

    # 4. 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Inspection results saved to {output_path}")

    logger.info("Inspection completed!")
    return results
