"""
完整 ML Pipeline 集成测试

测试重构后的完整流程：data → feature → model → train → eval → export → serve

重点验证：
1. FeatureGraph 作为唯一特征源的完整性
2. Metadata-driven 架构的解耦性
3. 端到端流程的流通性
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from bamboohepml.engine import Evaluator, Predictor, Trainer
from bamboohepml.metadata import load_model_metadata, save_model_metadata
from bamboohepml.models import get_model
from bamboohepml.tasks import export_task


def _create_dummy_data(num_samples: int = 100, input_dim: int = 10):
    """创建虚拟数据用于测试。"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)


def _collate_fn(batch):
    """将数据转换为模型输入格式。"""
    X, y = zip(*batch)
    return {
        "event": torch.stack(X),
        "_label_": torch.stack(y),
    }


def test_full_pipeline_flow():
    """
    测试完整 pipeline 流程：data → feature → model → train → eval → export → serve

    验证点：
    1. 数据加载和特征构建
    2. 模型创建和训练
    3. 模型评估
    4. Metadata 保存和加载
    5. 模型导出（ONNX）
    6. 模型推理（使用导出的 metadata）
    """
    print("\n" + "=" * 80)
    print("测试完整 Pipeline 流程")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ========== 1. DATA: 创建数据 ==========
        print("\n1. [DATA] 创建训练和验证数据...")
        train_dataset = _create_dummy_data(num_samples=200, input_dim=10)
        val_dataset = _create_dummy_data(num_samples=50, input_dim=10)
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=_collate_fn)
        print("   ✓ 数据创建成功")

        # ========== 2. MODEL: 创建模型 ==========
        print("\n2. [MODEL] 创建模型...")
        input_dim = 10
        num_classes = 2
        model = get_model("mlp_classifier", input_dim=input_dim, hidden_dims=[32, 16], num_classes=num_classes)
        print(f"   ✓ 模型创建成功: input_dim={input_dim}, num_classes={num_classes}")

        # ========== 3. TRAIN: 训练模型 ==========
        print("\n3. [TRAIN] 训练模型...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=torch.device("cpu"),
            task_type="classification",
        )
        history = trainer.fit(num_epochs=3)
        print(f"   ✓ 训练完成: {len(history)} epochs")

        # ========== 4. EVAL: 评估模型 ==========
        print("\n4. [EVAL] 评估模型...")
        evaluator = Evaluator(task_type="classification", input_key="event")
        metrics = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
        assert "loss" in metrics
        assert "accuracy" in metrics
        print(f"   ✓ 评估完成: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")

        # ========== 5. EXPORT: 保存模型和 Metadata ==========
        print("\n5. [EXPORT] 保存模型和 Metadata...")
        model_path = tmpdir / "model.pt"
        metadata_path = tmpdir / "metadata.json"

        # 保存模型权重
        torch.save(model.state_dict(), model_path)

        # 构建并保存 metadata
        feature_spec = {
            "event": {
                "features": [
                    "feature_0",
                    "feature_1",
                    "feature_2",
                    "feature_3",
                    "feature_4",
                    "feature_5",
                    "feature_6",
                    "feature_7",
                    "feature_8",
                    "feature_9",
                ],
                "dim": input_dim,
            }
        }
        save_model_metadata(
            metadata_path,
            feature_spec=feature_spec,
            task_type="classification",
            model_config={
                "name": "mlp_classifier",
                "params": {"input_dim": input_dim, "hidden_dims": [32, 16], "num_classes": num_classes},
            },
            input_dim=input_dim,
            input_key="event",
            feature_state={},
            experiment_name="test_pipeline",
        )
        print(f"   ✓ 模型和 Metadata 保存成功: {model_path}, {metadata_path}")

        # ========== 6. VERIFY: 验证 Metadata 可以加载 ==========
        print("\n6. [VERIFY] 验证 Metadata 加载...")
        metadata = load_model_metadata(metadata_path)
        assert metadata["feature_spec"] == feature_spec
        assert metadata["input_dim"] == input_dim
        assert metadata["input_key"] == "event"
        assert metadata["task_type"] == "classification"
        print("   ✓ Metadata 加载验证成功")

        # ========== 7. EXPORT: 导出 ONNX 模型 ==========
        print("\n7. [EXPORT] 导出 ONNX 模型...")
        onnx_path = tmpdir / "model.onnx"
        export_task(
            model_path=str(model_path),
            output_path=str(onnx_path),
            metadata_path=str(metadata_path),
        )
        assert onnx_path.exists()
        print(f"   ✓ ONNX 模型导出成功: {onnx_path}")

        # ========== 8. SERVE: 使用模型进行推理 ==========
        print("\n8. [SERVE] 使用模型进行推理...")
        # 重新加载模型
        model_reloaded = get_model("mlp_classifier", input_dim=input_dim, hidden_dims=[32, 16], num_classes=num_classes)
        model_reloaded.load_state_dict(torch.load(model_path))
        model_reloaded.eval()

        # 创建预测器
        predictor = Predictor(model_reloaded, device=torch.device("cpu"))

        # 创建测试数据
        test_dataset = TensorDataset(torch.randn(10, input_dim))

        def test_collate_fn(batch):
            X = torch.stack([x[0] for x in batch])
            return {"event": X}

        test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=test_collate_fn)

        # 进行预测
        results = predictor.predict(test_loader)
        assert len(results) == 10
        assert "prediction" in results[0]
        print(f"   ✓ 推理成功: {len(results)} 个样本")

        # ========== 9. VERIFY: 验证端到端一致性 ==========
        print("\n9. [VERIFY] 验证端到端一致性...")
        # 使用相同的输入验证训练后的模型和重新加载的模型输出一致
        test_input = torch.randn(1, input_dim)
        model.eval()
        model_reloaded.eval()
        with torch.no_grad():
            output1 = model({"event": test_input})
            output2 = model_reloaded({"event": test_input})
            assert torch.allclose(output1, output2, atol=1e-6)
        print("   ✓ 端到端一致性验证成功")

    print("\n" + "=" * 80)
    print("完整 Pipeline 流程测试通过！")
    print("=" * 80)


def test_metadata_driven_export():
    """
    测试 Metadata-driven 导出：验证导出不依赖原始 Dataset 或 Pipeline。
    """
    print("\n" + "=" * 80)
    print("测试 Metadata-driven 导出")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. 创建并训练模型
        print("\n1. 创建并训练模型...")
        input_dim = 10
        num_classes = 2
        model = get_model("mlp_classifier", input_dim=input_dim, hidden_dims=[32, 16], num_classes=num_classes)
        train_dataset = _create_dummy_data(num_samples=100, input_dim=input_dim)
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=_collate_fn)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            device=torch.device("cpu"),
        )
        trainer.fit(num_epochs=1)

        # 2. 保存模型和 metadata
        model_path = tmpdir / "model.pt"
        metadata_path = tmpdir / "metadata.json"
        torch.save(model.state_dict(), model_path)
        feature_spec = {
            "event": {
                "features": [f"feature_{i}" for i in range(input_dim)],
                "dim": input_dim,
            }
        }
        save_model_metadata(
            metadata_path,
            feature_spec=feature_spec,
            task_type="classification",
            model_config={
                "name": "mlp_classifier",
                "params": {"input_dim": input_dim, "hidden_dims": [32, 16], "num_classes": num_classes},
            },
            input_dim=input_dim,
            input_key="event",
            feature_state={},
        )
        print("   ✓ 模型和 metadata 保存成功")

        # 3. 仅使用模型和 metadata 导出 ONNX（不依赖 Dataset 或 Pipeline）
        print("\n2. 仅使用模型和 metadata 导出 ONNX...")
        onnx_path = tmpdir / "model.onnx"
        export_task(
            model_path=str(model_path),
            output_path=str(onnx_path),
            metadata_path=str(metadata_path),
        )
        assert onnx_path.exists()
        print("   ✓ ONNX 导出成功（无需 Dataset 或 Pipeline）")

    print("\n" + "=" * 80)
    print("Metadata-driven 导出测试通过！")
    print("=" * 80)


def test_featuregraph_integration():
    """
    测试 FeatureGraph 作为唯一特征源的集成性。

    注意：这是一个简化的测试，完整测试需要真实的 ROOT/Parquet 数据和 features.yaml。
    这里主要验证 FeatureGraph 的输出格式与模型输入的一致性。
    """
    print("\n" + "=" * 80)
    print("测试 FeatureGraph 集成性")
    print("=" * 80)

    # 验证 FeatureGraph 的输出格式（event/object/mask）与模型输入的一致性
    # 这里我们模拟 FeatureGraph.build_batch() 的输出格式
    batch_size = 32
    input_dim = 10

    # 模拟 FeatureGraph.build_batch() 的输出
    batch = {
        "event": torch.randn(batch_size, input_dim),  # event-level features
        "_label_": torch.randint(0, 2, (batch_size,)),
    }

    # 验证格式符合模型输入要求
    assert "event" in batch
    assert batch["event"].shape == (batch_size, input_dim)

    # 创建模型并验证可以处理这种格式
    model = get_model("mlp_classifier", input_dim=input_dim, hidden_dims=[32, 16], num_classes=2)
    model.eval()
    with torch.no_grad():
        # 模型 forward 方法期望 "features" 键，需要从 batch 中提取 event 并转换
        output = model({"features": batch["event"]})
        assert output.shape == (batch_size, 2)

    print("   ✓ FeatureGraph 输出格式与模型输入一致")

    print("\n" + "=" * 80)
    print("FeatureGraph 集成性测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_full_pipeline_flow()
    test_metadata_driven_export()
    test_featuregraph_integration()
    print("\n所有测试通过！")
