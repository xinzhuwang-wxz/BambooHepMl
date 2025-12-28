"""
预测任务

支持：
- 批量预测
- 概率预测
- 结果保存
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..config import logger
from ..engine import Predictor
from ..models import get_model
from ..pipeline import PipelineOrchestrator


def predict_task(
    pipeline_config_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
    return_probabilities: bool = False,
) -> List[Dict[str, Any]]:
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
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 使用 Predictor
    predictor = Predictor(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 5. 执行预测（使用 Predictor）
    logger.info("Running predictions...")
    results = predictor.predict(dataloader, return_probabilities=return_probabilities)

    # 7. 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")

    logger.info(f"Prediction completed! Total samples: {len(results)}")
    return results
