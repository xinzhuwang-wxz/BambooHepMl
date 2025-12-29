"""
完整 ML Pipeline 集成测试

测试完整的流程：data → model → train → eval → export → serve
"""

import sys
import tempfile
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

# Mock onnx module before importing torch to avoid PyTorch _dynamo import errors
if "onnx" not in sys.modules:
    onnx_mock = ModuleType("onnx")
    # Create a proper ModuleSpec to satisfy PyTorch _dynamo's import checks
    onnx_mock.__spec__ = ModuleSpec("onnx", None)
    sys.modules["onnx"] = onnx_mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from bamboohepml.engine import Evaluator, Predictor, Trainer
from bamboohepml.models import get_model


def test_module_availability():
    """测试所有核心模块的可用性。"""
    print("\n" + "=" * 80)
    print("测试模块可用性")
    print("=" * 80)

    # 1. Data 模块
    print("\n1. 测试 Data 模块...")
    from bamboohepml.data import DataConfig, DataSourceFactory, HEPDataset

    assert DataConfig is not None
    assert DataSourceFactory is not None
    assert HEPDataset is not None
    print("   ✓ Data 模块可用")

    # 2. Models 模块
    print("\n2. 测试 Models 模块...")
    from bamboohepml.models import BaseModel, get_model

    assert BaseModel is not None
    assert get_model is not None
    print("   ✓ Models 模块可用")

    # 3. Engine 模块
    print("\n3. 测试 Engine 模块...")
    from bamboohepml.engine import Evaluator, Predictor, Trainer

    assert Trainer is not None
    assert Evaluator is not None
    assert Predictor is not None
    print("   ✓ Engine 模块可用")

    # 4. Tasks 模块
    print("\n4. 测试 Tasks 模块...")
    from bamboohepml.tasks import export_task, predict_task, train_task

    assert train_task is not None
    assert predict_task is not None
    assert export_task is not None
    print("   ✓ Tasks 模块可用")

    # 5. Pipeline 模块
    print("\n5. 测试 Pipeline 模块...")
    from bamboohepml.pipeline import PipelineOrchestrator

    assert PipelineOrchestrator is not None
    print("   ✓ Pipeline 模块可用")

    # 6. Serve 模块
    print("\n6. 测试 Serve 模块...")
    from bamboohepml.serve import ONNXPredictor

    assert ONNXPredictor is not None
    print("   ✓ Serve 模块可用")

    print("\n" + "=" * 80)
    print("所有模块可用性测试通过！")
    print("=" * 80)


def test_data_to_model_pipeline():
    """测试 data → model 流程。"""
    print("\n" + "=" * 80)
    print("测试 Data → Model 流程")
    print("=" * 80)

    # 1. 创建模型
    print("\n1. 创建模型...")
    model = get_model("mlp_classifier", input_dim=10, hidden_dims=[32, 16], num_classes=2)
    assert model is not None
    print("   ✓ 模型创建成功")

    # 2. 创建虚拟数据
    print("\n2. 创建虚拟数据...")
    X = torch.randn(32, 10)

    # 转换为模型输入格式
    batch = {"features": X}
    print("   ✓ 数据创建成功")

    # 3. 测试前向传播
    print("\n3. 测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(batch)
        assert output.shape == (32, 2)
    print("   ✓ 前向传播成功")

    # 4. 测试预测
    print("\n4. 测试模型预测...")
    predictions = model.predict(batch)
    assert predictions.shape == (32,)
    print("   ✓ 预测成功")

    print("\n" + "=" * 80)
    print("Data → Model 流程测试通过！")
    print("=" * 80)


def test_train_eval_pipeline():
    """测试 train → eval 流程。"""
    print("\n" + "=" * 80)
    print("测试 Train → Eval 流程")
    print("=" * 80)

    # 1. 创建模型
    print("\n1. 创建模型...")
    model = get_model("mlp_classifier", input_dim=10, hidden_dims=[32, 16], num_classes=2)
    print("   ✓ 模型创建成功")

    # 2. 创建数据
    print("\n2. 创建训练和验证数据...")
    train_dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    val_dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

    def collate_fn(batch):
        X, y = zip(*batch)
        return {
            "_features": torch.stack(X),
            "_label_": torch.stack(y),
        }

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    print("   ✓ 数据加载器创建成功")

    # 3. 训练
    print("\n3. 训练模型...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
    )

    trainer.fit(num_epochs=2)
    print("   ✓ 训练完成")

    # 4. 评估
    print("\n4. 评估模型...")
    evaluator = Evaluator(task_type="classification")
    metrics = evaluator.evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )
    assert "loss" in metrics
    assert "accuracy" in metrics
    print(f"   ✓ 评估完成: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("Train → Eval 流程测试通过！")
    print("=" * 80)


def test_export_predict_pipeline():
    """测试 export → predict 流程。"""
    print("\n" + "=" * 80)
    print("测试 Export → Predict 流程")
    print("=" * 80)

    # 1. 创建并训练模型
    print("\n1. 创建并训练模型...")
    model = get_model("mlp_classifier", input_dim=10, hidden_dims=[32, 16], num_classes=2)

    # 简单训练
    train_dataset = TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,)))

    def collate_fn(batch):
        X, y = zip(*batch)
        return {
            "_features": torch.stack(X),
            "_label_": torch.stack(y),
        }

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
    )
    trainer.fit(num_epochs=1)
    print("   ✓ 模型训练完成")

    # 2. 保存模型
    print("\n2. 保存模型...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"   ✓ 模型保存到 {model_path}")

        # 3. 测试预测
        print("\n3. 测试预测...")
        predictor = Predictor(model, device=torch.device("cpu"))

        # 创建测试数据加载器
        test_dataset = TensorDataset(torch.randn(10, 10))

        def test_collate_fn(batch):
            X = torch.stack([x[0] for x in batch])
            return {"_features": X}

        test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=test_collate_fn)
        results = predictor.predict(test_loader)

        assert len(results) == 10
        assert "prediction" in results[0]
        print(f"   ✓ 预测成功: {len(results)} 个样本")

        print("\n" + "=" * 80)
        print("Export → Predict 流程测试通过！")
        print("=" * 80)


def test_full_pipeline_integration():
    """测试完整的 ML Pipeline：data → model → train → eval → export → serve。"""
    print("\n" + "=" * 80)
    print("测试完整 ML Pipeline")
    print("=" * 80)

    # 1. Data: 创建数据
    print("\n1. [DATA] 创建数据...")
    train_dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    val_dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

    def collate_fn(batch):
        X, y = zip(*batch)
        return {
            "_features": torch.stack(X),
            "_label_": torch.stack(y),
        }

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    print("   ✓ 数据创建成功")

    # 2. Model: 创建模型
    print("\n2. [MODEL] 创建模型...")
    model = get_model("mlp_classifier", input_dim=10, hidden_dims=[32, 16], num_classes=2)
    print("   ✓ 模型创建成功")

    # 3. Train: 训练模型
    print("\n3. [TRAIN] 训练模型...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
    )
    trainer.fit(num_epochs=2)
    print("   ✓ 训练完成")

    # 4. Eval: 评估模型
    print("\n4. [EVAL] 评估模型...")
    evaluator = Evaluator(task_type="classification")
    metrics = evaluator.evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )
    print(f"   ✓ 评估完成: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")

    # 5. Export: 保存模型
    print("\n5. [EXPORT] 保存模型...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"   ✓ 模型保存到 {model_path}")

        # 6. Serve: 使用模型进行预测
        print("\n6. [SERVE] 使用模型进行预测...")
        predictor = Predictor(model, device=torch.device("cpu"))

        # 创建测试数据加载器
        test_dataset = TensorDataset(torch.randn(5, 10))

        def test_collate_fn(batch):
            X = torch.stack([x[0] for x in batch])
            return {"_features": X}

        test_loader = DataLoader(test_dataset, batch_size=5, collate_fn=test_collate_fn)
        results = predictor.predict(test_loader)

        assert len(results) == 5
        assert "prediction" in results[0]
        print(f"   ✓ 预测成功: {len(results)} 个样本")

    print("\n" + "=" * 80)
    print("完整 ML Pipeline 测试通过！")
    print("=" * 80)
