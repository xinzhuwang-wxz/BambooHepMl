"""
完整 ML Pipeline 集成测试（包含 FeatureGraph 和 PipelineOrchestrator）

测试完整的流程：data → feature (FeatureGraph) → model → train → eval → export → serve

重点验证：
1. PipelineOrchestrator 的完整工作流程
2. FeatureGraph 的特征处理和拟合
3. 不同任务类型（classification, regression）的支持
4. Metadata-driven 架构的端到端一致性
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import awkward as ak
import numpy as np
import torch
from torch.utils.data import DataLoader

# Mock Ray for testing (Ray is optional dependency)
sys.modules["ray"] = MagicMock()
sys.modules["ray.data"] = MagicMock()
sys.modules["ray.train"] = MagicMock()
sys.modules["ray.train.torch"] = MagicMock()
sys.modules["ray.train.torch"].TorchTrainer = MagicMock()

from bamboohepml.engine import Predictor  # noqa: E402
from bamboohepml.metadata import load_model_metadata  # noqa: E402
from bamboohepml.tasks import export_task, train_task  # noqa: E402


def _create_mock_data_source(num_events: int = 100):
    """创建模拟数据源，返回包含 met 和 Jet 的 awkward array"""
    # 创建 met（event-level）
    met = np.abs(np.random.randn(num_events) * 50)

    # 创建 Jet（object-level，每个事件有不同数量的 jet）
    jet_pt_list = []
    jet_eta_list = []
    jet_phi_list = []
    jet_mass_list = []

    for _ in range(num_events):
        n_jets = np.random.randint(5, 15)
        jet_pt_list.append(np.abs(np.random.randn(n_jets) * 50))
        jet_eta_list.append(np.random.randn(n_jets) * 2)
        jet_phi_list.append(np.random.randn(n_jets) * np.pi)
        jet_mass_list.append(np.abs(np.random.randn(n_jets) * 10))

    jet_pt = ak.Array(jet_pt_list)
    jet_eta = ak.Array(jet_eta_list)
    jet_phi = ak.Array(jet_phi_list)
    jet_mass = ak.Array(jet_mass_list)

    Jet = ak.zip(
        {
            "pt": jet_pt,
            "eta": jet_eta,
            "phi": jet_phi,
            "mass": jet_mass,
        }
    )

    # 创建标签（分类任务：0 或 1）
    label = np.random.randint(0, 2, num_events)

    # 组装成 table
    table = ak.Array(
        {
            "met": met,
            "Jet": Jet,
            "label": label,
        }
    )

    return table


def _create_pipeline_config(tmpdir: Path, features_config_path: str, data_config_path: str, task_type: str = "classification"):
    """创建 pipeline.yaml 配置文件"""
    pipeline_config = {
        "data": {
            "config_path": str(data_config_path),
            "source_path": str(tmpdir / "dummy.root"),  # 需要 source_path，但会被 mock
            "treename": "tree",
        },
        "features": {
            "config_path": str(features_config_path),
        },
        "model": {
            "name": "mlp_classifier" if task_type == "classification" else "mlp_regressor",
            "params": {
                "hidden_dims": [32, 16],
                "num_classes": 2 if task_type == "classification" else None,
            },
        },
        "train": {
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "task_type": task_type,
            "learning_paradigm": "supervised",
        },
    }

    pipeline_path = tmpdir / "pipeline.yaml"
    import yaml

    with open(pipeline_path, "w") as f:
        yaml.dump(pipeline_config, f)

    return pipeline_path


def _create_features_config(tmpdir: Path):
    """创建 features.yaml 配置文件"""
    features_config = {
        "features": {
            "event_level": [
                {
                    "name": "met",
                    "source": "met",
                    "type": "event",
                    "dtype": "float32",
                },
                {
                    "name": "met_log",
                    "source": "met",
                    "type": "event",
                    "dtype": "float32",
                    "expr": "log1p(met)",
                    "normalize": {"method": "auto"},
                    "clip": {"min": 0.0, "max": 10.0},
                },
                {
                    "name": "nJets",
                    "source": "Jet",
                    "type": "event",
                    "dtype": "int32",
                    "expr": "len(Jet)",
                },
                {
                    "name": "ht",
                    "source": "Jet",
                    "type": "event",
                    "dtype": "float32",
                    "expr": "sum(Jet.pt)",
                    "normalize": {"method": "auto"},
                },
            ],
        },
    }

    features_path = tmpdir / "features.yaml"
    import yaml

    with open(features_path, "w") as f:
        yaml.dump(features_config, f)

    return features_path


def _create_data_config(tmpdir: Path, task_type: str = "classification"):
    """创建 data.yaml 配置文件"""
    data_config = {
        "train_load_branches": ["met", "Jet", "label"],
        "test_load_branches": ["met", "Jet"],
    }

    # 对于分类任务，使用 simple 类型（通过 argmax 计算）
    # 对于回归任务，使用 complex 类型，直接映射字段名
    if task_type == "classification":
        data_config["labels"] = {
            "type": "simple",
            "value": ["label"],  # 标签字段名列表，会通过 argmax 计算
        }
    else:  # regression
        # 对于回归任务，使用 complex 类型，直接映射字段名
        # complex 类型会直接使用字段值，不通过 argmax
        data_config["labels"] = {
            "type": "complex",
            "value": {"_label_": "label"},  # 直接使用 label 字段作为 _label_
        }

    data_path = tmpdir / "data.yaml"
    import yaml

    with open(data_path, "w") as f:
        yaml.dump(data_config, f)

    return data_path


def test_complete_pipeline_classification():
    """
    测试完整的分类任务 pipeline：data → feature → model → train → eval → export → serve

    使用 PipelineOrchestrator 和 train_task 进行端到端测试
    """
    print("\n" + "=" * 80)
    print("测试完整 Pipeline（分类任务）")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ========== 1. 创建配置文件 ==========
        print("\n1. [CONFIG] 创建配置文件...")
        features_config_path = _create_features_config(tmpdir)
        data_config_path = _create_data_config(tmpdir, task_type="classification")
        pipeline_config_path = _create_pipeline_config(tmpdir, features_config_path, data_config_path, task_type="classification")
        print("   ✓ 配置文件创建成功")

        # ========== 2. 创建模拟数据源 ==========
        print("\n2. [DATA] 创建模拟数据源...")
        # 创建 mock DataSource
        mock_data_source = MagicMock()
        mock_table = _create_mock_data_source(num_events=200)
        mock_data_source.load_branches.return_value = mock_table
        mock_data_source.get_available_branches.return_value = ["met", "Jet", "label"]
        mock_data_source.get_file_paths.return_value = [str(tmpdir / "dummy.root")]

        # ========== 3. 训练模型 ==========
        print("\n3. [TRAIN] 使用 train_task 训练模型...")
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Mock DataSourceFactory.create
        with patch("bamboohepml.pipeline.orchestrator.DataSourceFactory.create", return_value=mock_data_source):
            train_task(
                pipeline_config_path=str(pipeline_config_path),
                experiment_name="test_classification",
                num_epochs=2,
                batch_size=16,
                output_dir=str(output_dir),
                use_ray=False,
            )
            print("   ✓ 训练完成")

        # ========== 4. 验证输出 ==========
        print("\n4. [VERIFY] 验证训练输出...")
        model_path = output_dir / "model.pt"
        metadata_path = output_dir / "metadata.json"
        pipeline_state_path = output_dir / "pipeline_state.json"

        assert model_path.exists(), "模型文件不存在"
        assert metadata_path.exists(), "Metadata 文件不存在"
        assert pipeline_state_path.exists(), "PipelineState 文件不存在"
        print("   ✓ 所有输出文件存在")

        # ========== 5. 验证 Metadata ==========
        print("\n5. [VERIFY] 验证 Metadata...")
        metadata = load_model_metadata(metadata_path)
        assert metadata["task_type"] == "classification"
        assert "feature_spec" in metadata
        assert "event" in metadata["feature_spec"]
        assert metadata["input_key"] == "event"
        print("   ✓ Metadata 验证成功")

        # ========== 6. 导出 ONNX ==========
        print("\n6. [EXPORT] 导出 ONNX 模型...")
        onnx_path = tmpdir / "model.onnx"
        export_task(
            model_path=str(model_path),
            output_path=str(onnx_path),
            metadata_path=str(metadata_path),
        )
        assert onnx_path.exists()
        print("   ✓ ONNX 导出成功")

        # ========== 7. 推理测试 ==========
        print("\n7. [SERVE] 测试模型推理...")
        from bamboohepml.models import get_model

        # 重新加载模型
        model = get_model(
            "mlp_classifier",
            input_dim=metadata["input_dim"],
            hidden_dims=[32, 16],
            num_classes=2,
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # 创建预测器
        predictor = Predictor(model, device=torch.device("cpu"), input_key="event")

        # 创建测试数据（模拟 FeatureGraph 的输出格式）
        test_batch = {
            "event": torch.randn(10, metadata["input_dim"]),
        }
        test_dataset = torch.utils.data.TensorDataset(test_batch["event"])

        def test_collate_fn(batch):
            X = torch.stack(batch)
            return {"event": X}

        test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=test_collate_fn)
        results = predictor.predict(test_loader)

        assert len(results) == 10
        assert "prediction" in results[0]
        print("   ✓ 推理成功")

    print("\n" + "=" * 80)
    print("完整 Pipeline（分类任务）测试通过！")
    print("=" * 80)


def test_complete_pipeline_regression():
    """
    测试完整的回归任务 pipeline：data → feature → model → train → eval → export → serve
    """
    print("\n" + "=" * 80)
    print("测试完整 Pipeline（回归任务）")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ========== 1. 创建配置文件 ==========
        print("\n1. [CONFIG] 创建配置文件...")
        features_config_path = _create_features_config(tmpdir)
        data_config_path = _create_data_config(tmpdir, task_type="regression")
        pipeline_config_path = _create_pipeline_config(tmpdir, features_config_path, data_config_path, task_type="regression")
        print("   ✓ 配置文件创建成功")

        # ========== 2. 创建模拟数据源（回归任务：标签为连续值）==========
        print("\n2. [DATA] 创建模拟数据源（回归任务）...")
        mock_data_source = MagicMock()
        # 创建回归任务的标签（连续值）
        num_events = 200
        mock_table = _create_mock_data_source(num_events=num_events)
        # 修改标签为连续值
        mock_table = ak.with_field(mock_table, np.abs(np.random.randn(num_events) * 10), "label")
        mock_data_source.load_branches.return_value = mock_table
        mock_data_source.get_available_branches.return_value = ["met", "Jet", "label"]
        mock_data_source.get_file_paths.return_value = [str(tmpdir / "dummy.root")]

        # ========== 3. 训练模型 ==========
        print("\n3. [TRAIN] 使用 train_task 训练模型...")
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Mock DataSourceFactory.create
        with patch("bamboohepml.pipeline.orchestrator.DataSourceFactory.create", return_value=mock_data_source):
            train_task(
                pipeline_config_path=str(pipeline_config_path),
                experiment_name="test_regression",
                num_epochs=2,
                batch_size=16,
                output_dir=str(output_dir),
                use_ray=False,
            )
            print("   ✓ 训练完成")

        # ========== 4. 验证输出 ==========
        print("\n4. [VERIFY] 验证训练输出...")
        model_path = output_dir / "model.pt"
        metadata_path = output_dir / "metadata.json"

        assert model_path.exists()
        assert metadata_path.exists()
        print("   ✓ 所有输出文件存在")

        # ========== 5. 验证 Metadata ==========
        print("\n5. [VERIFY] 验证 Metadata...")
        metadata = load_model_metadata(metadata_path)
        assert metadata["task_type"] == "regression"
        assert metadata["input_key"] == "event"
        print("   ✓ Metadata 验证成功")

        # ========== 6. 导出 ONNX ==========
        print("\n6. [EXPORT] 导出 ONNX 模型...")
        onnx_path = tmpdir / "model.onnx"
        export_task(
            model_path=str(model_path),
            output_path=str(onnx_path),
            metadata_path=str(metadata_path),
        )
        assert onnx_path.exists()
        print("   ✓ ONNX 导出成功")

    print("\n" + "=" * 80)
    print("完整 Pipeline（回归任务）测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_complete_pipeline_classification()
    test_complete_pipeline_regression()
    print("\n所有测试通过！")
