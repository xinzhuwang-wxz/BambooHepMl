"""
导出任务

支持：
- ONNX 格式导出
- 模型验证

导出不依赖 Dataset 和 Pipeline，只依赖模型和 metadata。
"""

from pathlib import Path
from typing import Any

import torch

from ..config import logger
from ..metadata import load_model_metadata
from ..models import get_model


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

            # setup_model 会自动从 FeatureGraph 推断维度
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
        model_name = model_config.get("name")
        model_params = model_config.get("params", {}).copy()
        # model_params 已包含 event_input_dim/object_input_dim/embed_dim

    # 2. 加载模型
    logger.info(f"Loading model from {model_path}...")
    model = get_model(model_name, **model_params)

    # 加载权重
    state_dict = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 创建示例输入（支持多输入：event, object, mask）
    if feature_spec is not None:
        # 从 feature_spec 构建 dummy inputs（支持多输入）
        dummy_inputs = []
        input_names = []
        dynamic_axes = {}

        if "event" in feature_spec:
            dummy_event = torch.randn(1, feature_spec["event"]["dim"], dtype=torch.float32)
            dummy_inputs.append(dummy_event)
            input_names.append("event")
            dynamic_axes["event"] = {0: "batch_size"}

        if "object" in feature_spec:
            dim = feature_spec["object"]["dim"]
            max_length = feature_spec["object"]["max_length"]
            dummy_object = torch.randn(1, max_length, dim, dtype=torch.float32)
            dummy_inputs.append(dummy_object)
            input_names.append("object")
            dynamic_axes["object"] = {0: "batch_size"}

            # 如果有 mask，也添加
            if "mask" in feature_spec:
                dummy_mask = torch.ones(1, max_length, dtype=torch.bool)
                dummy_inputs.append(dummy_mask)
                input_names.append("mask")
                dynamic_axes["mask"] = {0: "batch_size"}

        # 确保至少有一个输入
        if len(dummy_inputs) == 0:
            raise ValueError("Cannot create dummy input: feature_spec must contain at least 'event' or 'object' features")
    else:
        raise ValueError("feature_spec is required for ONNX export")

    # 4. 导出 ONNX
    logger.info(f"Exporting to ONNX (opset_version={opset_version})...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建包装类，将 tensor 输入转换为字典
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, input_names):
            super().__init__()
            self.model = model
            self.input_names = input_names

        def forward(self, *args):
            # 构建 batch 字典（统一使用新架构）
            batch = {}
            for i, name in enumerate(self.input_names):
                batch[name] = args[i]
            return self.model(batch)

    wrapped_model = ModelWrapper(model, input_names)
    wrapped_model.eval()

    # 如果只有一个输入，解包；否则传递元组
    if len(dummy_inputs) == 1:
        export_input = dummy_inputs[0]
    else:
        export_input = tuple(dummy_inputs)

    torch.onnx.export(
        wrapped_model,
        export_input,
        str(output_path),
        input_names=input_names,
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"ONNX model exported to {output_path}")

    # 5. 验证 ONNX 模型
    logger.info("Validating ONNX model...")
    # Lazy import onnx to avoid torch._dynamo import errors during training
    import onnx

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

            # 运行推理（支持多输入）
            ort_inputs = {}
            for i, input_name in enumerate(input_names):
                if len(dummy_inputs) == 1:
                    input_value = dummy_inputs[0]
                else:
                    input_value = dummy_inputs[i]
                ort_input_name = ort_session.get_inputs()[i].name
                # 转换 bool mask 为 int8（ONNX 通常需要）
                if input_name == "mask" and input_value.dtype == torch.bool:
                    ort_inputs[ort_input_name] = input_value.detach().cpu().numpy().astype("int8")
                else:
                    ort_inputs[ort_input_name] = input_value.detach().cpu().numpy()
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
