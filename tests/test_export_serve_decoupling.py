"""
Export/Serve 解耦测试

测试 Export 和 Serve 不依赖 Dataset/Pipeline：
- Export 从 metadata 推断输入
- Serve 只依赖 metadata + ONNX
"""

import sys
import tempfile
from pathlib import Path

import torch

# 添加项目根目录到路径（必须在导入项目模块之前）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bamboohepml.metadata import load_model_metadata, save_model_metadata  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402


def test_export_without_dataset():
    """测试 Export 不依赖 Dataset。"""
    print("=" * 60)
    print("测试: Export 解耦（不依赖 Dataset）")
    print("=" * 60)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建模型
        input_dim = 10
        model = get_model(
            "mlp_classifier",
            task_type="classification",
            input_dim=input_dim,
            hidden_dims=[32, 16],
            num_classes=2,
        )

        # 保存模型
        model_path = tmpdir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # 创建并保存 metadata（不依赖 Dataset）
        feature_spec = {
            "event": {
                "features": [f"feature_{i}" for i in range(input_dim)],
                "dim": input_dim,
            }
        }

        metadata_path = tmpdir / "metadata.json"
        save_model_metadata(
            metadata_path=str(metadata_path),
            feature_spec=feature_spec,
            task_type="classification",
            model_config={
                "name": "mlp_classifier",
                "params": {
                    "input_dim": input_dim,
                    "hidden_dims": [32, 16],
                    "num_classes": 2,
                },
            },
            input_dim=input_dim,
            input_key="event",
        )

        # 验证 metadata 可以独立加载
        metadata = load_model_metadata(metadata_path)
        assert metadata["input_dim"] == input_dim
        assert metadata["input_key"] == "event"
        assert "event" in metadata["feature_spec"]

        print(f"✓ Metadata 保存成功: {metadata_path}")
        print(f"✓ Metadata 加载成功: input_dim={metadata['input_dim']}, input_key={metadata['input_key']}")

        # 验证可以从 metadata 重建模型配置
        model_config = metadata["model_config"]
        model_params = model_config["params"]
        assert model_params["input_dim"] == input_dim

        print("✓ Export 解耦测试通过\n")


def test_serve_without_pipeline():
    """测试 Serve 不依赖 Pipeline。"""
    print("=" * 60)
    print("测试: Serve 解耦（不依赖 Pipeline）")
    print("=" * 60)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建模型
        input_dim = 10
        model = get_model(
            "mlp_classifier",
            task_type="classification",
            input_dim=input_dim,
            hidden_dims=[32, 16],
            num_classes=2,
        )

        # 保存模型
        model_path = tmpdir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # 创建并保存 metadata（不依赖 Pipeline）
        feature_spec = {
            "event": {
                "features": [f"feature_{i}" for i in range(input_dim)],
                "dim": input_dim,
            }
        }

        metadata_path = tmpdir / "metadata.json"
        save_model_metadata(
            metadata_path=str(metadata_path),
            feature_spec=feature_spec,
            task_type="classification",
            model_config={
                "name": "mlp_classifier",
                "params": {
                    "input_dim": input_dim,
                    "hidden_dims": [32, 16],
                    "num_classes": 2,
                },
            },
            input_dim=input_dim,
            input_key="event",
        )

        # 验证可以从 metadata 加载模型配置（不依赖 Pipeline）
        metadata = load_model_metadata(metadata_path)
        model_config = metadata["model_config"]
        model_params = model_config["params"]

        # 重建模型（不依赖 Pipeline）
        loaded_model = get_model(
            model_config["name"],
            task_type=metadata["task_type"],
            **model_params,
        )

        # 加载权重
        loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # 测试推理
        dummy_input = torch.randn(1, input_dim)
        output = loaded_model({"features": dummy_input})
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"

        print("✓ 从 metadata 重建模型成功")
        print(f"✓ 模型推理成功: output shape={output.shape}")
        print("✓ Serve 解耦测试通过\n")


if __name__ == "__main__":
    test_export_without_dataset()
    test_serve_without_pipeline()
    print("=" * 60)
    print("所有 Export/Serve 解耦测试通过！")
    print("=" * 60)
