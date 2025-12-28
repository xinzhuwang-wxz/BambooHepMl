"""
导出任务

支持：
- ONNX 格式导出
- 模型验证
"""

from pathlib import Path
from typing import Any

import onnx
import torch

from ..config import logger
from ..models import get_model
from ..pipeline import PipelineOrchestrator


def export_task(
    pipeline_config_path: str,
    model_path: str,
    output_path: str,
    input_shape: tuple | None = None,
    opset_version: int = 11,
) -> dict[str, Any]:
    """
    导出任务主函数（ONNX 格式）。

    Args:
        pipeline_config_path: pipeline.yaml 路径
        model_path: 模型文件路径
        output_path: 输出 ONNX 文件路径
        input_shape: 输入形状（如果为 None，将从数据中推断）
        opset_version: ONNX opset 版本

    Returns:
        导出结果字典
    """
    logger.info("=" * 80)
    logger.info("Export Task (ONNX)")
    logger.info("=" * 80)

    # 1. 初始化 Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config_path)

    # 2. 加载模型
    logger.info(f"Loading model from {model_path}...")

    # 从 pipeline 配置获取模型配置
    model_config = orchestrator.get_model_config()
    model_name = model_config.get("name")
    model_params = model_config.get("params", {})

    # 从数据中推断输入维度（如果 input_shape 未提供）
    if input_shape is None:
        dataset = orchestrator.setup_data()
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
            input_shape = tuple(input_value.shape)
        else:
            raise ValueError(f"Unexpected input type: {type(input_value)}")

    input_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
    model_params["input_dim"] = input_dim
    model = get_model(model_name, **model_params)

    # 加载权重
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, *input_shape[1:]) if len(input_shape) > 1 else torch.randn(1, input_shape[0])

    # 4. 导出 ONNX
    logger.info(f"Exporting to ONNX (opset_version={opset_version})...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建包装类，将 tensor 输入转换为字典
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features):
            return self.model({"features": features})

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes={
            "features": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"ONNX model exported to {output_path}")

    # 5. 验证 ONNX 模型
    logger.info("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed!")

    # 6. 测试 ONNX 模型（可选）
    try:
        logger.info("Testing ONNX model...")
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not available, skipping ONNX model test")
        else:
            ort_session = ort.InferenceSession(str(output_path))

            # 运行推理
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)

            logger.info(f"ONNX model test passed! Output shape: {ort_outs[0].shape}")
    except Exception as e:
        logger.warning(f"ONNX model test failed: {e}")

    # 7. 返回结果
    results = {
        "model_path": model_path,
        "onnx_path": str(output_path),
        "input_shape": input_shape,
        "opset_version": opset_version,
    }

    logger.info("Export completed!")
    return results
