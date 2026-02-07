"""
一致性测试

测试 FeatureGraph 与 Model 配置的一致性：
- feature dim vs model input_dim
- 配置验证
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from bamboohepml.data.features.expression import ExpressionEngine
from bamboohepml.data.features.feature_graph import FeatureGraph
from bamboohepml.models import get_model
from bamboohepml.pipeline.state import PipelineState


def test_feature_model_consistency():
    """测试 FeatureGraph 与 Model 配置的一致性。"""
    print("=" * 60)
    print("测试: Feature vs Model 一致性")
    print("=" * 60)

    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "data" / "features" / "test_features_config.yaml"

    # 创建 FeatureGraph
    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)
    spec = graph.output_spec()

    # 获取输入维度
    if "event" in spec:
        input_dim = spec["event"]["dim"]
        input_key = "event"
    elif "object" in spec:
        input_dim = spec["object"]["dim"] * spec["object"]["max_length"]
        input_key = "object"
    else:
        raise ValueError("No features found in spec")

    print(f"FeatureGraph input_dim: {input_dim}, input_key: {input_key}")

    # 创建模型（使用正确的 input_dim）
    model = get_model(
        "mlp_classifier",
        task_type="classification",
        event_input_dim=input_dim,
        hidden_dims=[64, 32],
        num_classes=2,
    )

    print(f"Model created with event_input_dim: {input_dim}")

    # 测试前向传播
    if input_key == "event":
        dummy_input = torch.randn(2, input_dim)
    else:
        max_length = spec["object"]["max_length"]
        object_dim = spec["object"]["dim"]
        dummy_input = torch.randn(2, max_length, object_dim)

    output = model({"event": dummy_input})
    print(f"Model output shape: {output.shape}")

    assert output.shape[0] == 2, "batch size mismatch"
    print("✓ Feature vs Model 一致性测试通过\n")


def test_pipeline_state_validation():
    """测试 PipelineState 的配置验证。"""
    print("=" * 60)
    print("测试: PipelineState 验证")
    print("=" * 60)

    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "data" / "features" / "test_features_config.yaml"

    # 创建 FeatureGraph
    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)
    spec = graph.output_spec()

    # 获取输入维度
    if "event" in spec:
        input_dim = spec["event"]["dim"]
        input_key = "event"
    else:
        input_dim = spec["object"]["dim"] * spec["object"]["max_length"]
        input_key = "object"

    # 创建正确的 PipelineState
    model_config = {
        "name": "mlp_classifier",
        "params": {
            "event_input_dim": input_dim,
            "hidden_dims": [64, 32],
            "num_classes": 2,
        },
    }

    state = PipelineState.from_configs(
        feature_graph=graph,
        model_config=model_config,
        task_type="classification",
    )

    # 验证状态
    is_valid, errors = state.validate()
    assert is_valid, f"PipelineState validation failed: {errors}"
    print("✓ PipelineState 验证通过\n")

    # 测试不一致的配置
    wrong_model_config = {
        "name": "mlp_classifier",
        "params": {
            "event_input_dim": input_dim + 10,  # 错误的维度
            "hidden_dims": [64, 32],
            "num_classes": 2,
        },
    }

    wrong_state = PipelineState.from_configs(
        feature_graph=graph,
        model_config=wrong_model_config,
        task_type="classification",
    )

    is_valid, errors = wrong_state.validate()
    assert not is_valid, "PipelineState should fail validation with wrong input_dim"
    assert len(errors) > 0, "Should have validation errors"
    print(f"✓ 检测到配置不一致: {errors[0]}\n")


if __name__ == "__main__":
    test_feature_model_consistency()
    test_pipeline_state_validation()
    print("=" * 60)
    print("所有一致性测试通过！")
    print("=" * 60)
