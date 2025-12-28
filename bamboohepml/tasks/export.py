"""
导出任务

支持：
- ONNX 格式导出
- 模型验证

导出不依赖 Dataset 和 Pipeline，只依赖模型和 metadata。
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from ..config import logger
from ..models import get_model
from ..pipeline.orchestrator import PipelineOrchestrator
from ..utils import load_model_metadata, save_dict


def _build_dummy_input_from_spec(feature_spec: dict[str, Any], input_key: str, batch_size: int = 1) -> torch.Tensor:
    """
    从 feature_spec 构建虚拟输入。

    Args:
        feature_spec: 特征规范（来自 metadata）
        input_key: 输入键名（"event" 或 "object"）
        batch_size: 批次大小

    Returns:
        torch.Tensor: 虚拟输入张量
    """
    if input_key == "event":
        if "event" not in feature_spec:
            raise ValueError("input_key is 'event' but 'event' not found in feature_spec")
        dim = feature_spec["event"]["dim"]
        return torch.randn(batch_size, dim, dtype=torch.float32)
    elif input_key == "object":
        if "object" not in feature_spec:
            raise ValueError("input_key is 'object' but 'object' not found in feature_spec")
        dim = feature_spec["object"]["dim"]
        max_length = feature_spec["object"]["max_length"]
        return torch.randn(batch_size, max_length, dim, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown input_key: {input_key}. Must be 'event' or 'object'.")


def export_task(
    model_path: str,
    output_path: str,
    metadata_path: str | None = None,
    input_shape: tuple | None = None,
    opset_version: int = 11,
    pipeline_config_path: str | None = None,  # 向后兼容，已废弃
) -> dict[str, Any]:
    """
    导出任务主函数（ONNX 格式）。

    Args:
        model_path: 模型文件路径
        output_path: 输出 ONNX 文件路径
        metadata_path: 元数据文件路径（默认为 model_path 同目录下的 metadata.json）
        input_shape: 输入形状（如果为 None，将从 metadata 推断）
        opset_version: ONNX opset 版本
        pipeline_config_path: pipeline.yaml 路径（已废弃，向后兼容用）

    Returns:
        导出结果字典
    """
    logger.info("=" * 80)
    logger.info("Export Task (ONNX)")
    logger.info("=" * 80)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. 加载 metadata（优先从 metadata_path，否则从 model_path 同目录）
    if metadata_path is None:
        metadata_path = model_path.parent / "metadata.json"
    else:
        metadata_path = Path(metadata_path)

    # 初始化变量
    feature_spec = None
    input_key = None

    if not metadata_path.exists():
        if pipeline_config_path:
            logger.warning(
                f"Metadata file not found at {metadata_path}. "
                "Falling back to pipeline_config_path (deprecated). "
                "Please ensure metadata.json is saved during training."
            )
            # 向后兼容：从 pipeline 配置推断（已废弃）
            from ..pipeline import PipelineOrchestrator

            orchestrator = PipelineOrchestrator(pipeline_config_path)
            model_config = orchestrator.get_model_config()
            model_name = model_config.get("name")
            model_params = model_config.get("params", {})

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
        else:
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path} and pipeline_config_path not provided. "
                "Please provide metadata_path or ensure metadata.json exists in the same directory as model.pt"
            )
    else:
        # 从 metadata 加载
        metadata = load_model_metadata(metadata_path)
        model_config = metadata["model_config"]
        feature_spec = metadata["feature_spec"]
        input_dim = metadata["input_dim"]
        input_key = metadata["input_key"]
        model_name = model_config.get("name")
        model_params = model_config.get("params", {}).copy()
        model_params["input_dim"] = input_dim

        # 从 feature_spec 推断 input_shape（如果未提供）
        if input_shape is None:
            dummy_input = _build_dummy_input_from_spec(feature_spec, input_key, batch_size=1)
            input_shape = tuple(dummy_input.shape[1:])  # 去掉 batch 维度

    # 2. 加载模型
    logger.info(f"Loading model from {model_path}...")
    model = get_model(model_name, **model_params)

    # 加载权重
    state_dict = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 创建示例输入
    if input_shape is None:
        if feature_spec is not None and input_key is not None:
            # 从 feature_spec 构建
            dummy_input = _build_dummy_input_from_spec(feature_spec, input_key, batch_size=1)
        else:
            raise ValueError("Cannot create dummy input: both input_shape and (feature_spec, input_key) are None")
    else:
        # 使用提供的 input_shape
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
