"""
ONNX 推理测试
"""

import sys

import pytest

# Skip ONNX tests on Python 3.8 due to onnxscript compatibility issues
ONNX_AVAILABLE = sys.version_info >= (3, 9)

if ONNX_AVAILABLE:
    try:
        import onnx
        import torch

        from bamboohepml.serve import ONNXPredictor

        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
def test_onnx_export_and_predict(temp_dir, sample_model, sample_features):
    """测试 ONNX 导出和推理。"""
    # 导出 ONNX
    onnx_path = temp_dir / "test_model.onnx"
    dummy_input = {"features": torch.randn(1, 10)}

    # Use model directly, not a wrapper function
    sample_model.eval()
    torch.onnx.export(
        sample_model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["output"],
        opset_version=11,
    )

    # 验证 ONNX 模型
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 创建 ONNX 预测器
    predictor = ONNXPredictor(str(onnx_path))

    # 预测
    results = predictor.predict(sample_features)
    assert len(results) == len(sample_features)
    assert "prediction" in results[0]


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
def test_onnx_batch_predict(temp_dir, sample_model, sample_features):
    """测试 ONNX 批量预测。"""
    # 导出 ONNX
    onnx_path = temp_dir / "test_model.onnx"
    dummy_input = {"features": torch.randn(1, 10)}

    # Use model directly, not a wrapper function
    sample_model.eval()
    torch.onnx.export(
        sample_model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["output"],
        opset_version=11,
    )

    # 创建 ONNX 预测器
    predictor = ONNXPredictor(str(onnx_path))

    # 批量预测
    features_list = sample_features.tolist()
    results = predictor.predict_batch(features_list)
    assert len(results) == len(features_list)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
def test_onnx_model_info(temp_dir, sample_model):
    """测试 ONNX 模型信息。"""
    # 导出 ONNX
    onnx_path = temp_dir / "test_model.onnx"
    dummy_input = {"features": torch.randn(1, 10)}

    # Use model directly, not a wrapper function
    sample_model.eval()
    torch.onnx.export(
        sample_model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["output"],
        opset_version=11,
    )

    # 创建 ONNX 预测器
    predictor = ONNXPredictor(str(onnx_path))

    # 获取模型信息
    info = predictor.get_model_info()
    assert "onnx_path" in info
    assert "input_name" in info
    assert "output_name" in info
