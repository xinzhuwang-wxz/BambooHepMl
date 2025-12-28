"""
ONNX 推理测试（简化版）

只测试核心功能：导出和推理。
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


@pytest.fixture
def onnx_model_path(temp_dir, sample_model):
    """创建并导出 ONNX 模型。"""
    onnx_path = temp_dir / "test_model.onnx"
    dummy_input = {"features": torch.randn(1, 10)}

    sample_model.eval()
    torch.onnx.export(
        sample_model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["output"],
        opset_version=11,
    )

    # 验证模型
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    return str(onnx_path)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
def test_onnx_predict(onnx_model_path, sample_features):
    """测试 ONNX 推理（简化版：只测试核心功能）。"""
    predictor = ONNXPredictor(onnx_model_path)

    # 预测
    results = predictor.predict(sample_features)
    assert len(results) == len(sample_features)
    assert "prediction" in results[0]

    # 测试批量预测
    features_list = sample_features.tolist()
    batch_results = predictor.predict_batch(features_list)
    assert len(batch_results) == len(features_list)

    # 测试模型信息
    info = predictor.get_model_info()
    assert "onnx_path" in info
    assert "input_name" in info
    assert "output_name" in info
