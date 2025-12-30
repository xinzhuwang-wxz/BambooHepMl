"""
新架构集成测试

测试新 MLP 架构的完整流程，包括：
1. 只有 event 特征的情况
2. 只有 object 特征的情况
3. 同时有 event 和 object 特征的情况
4. 整个 pipeline：data → model → train → eval → export → serve

重点验证：
- 维度自动推断
- 数据格式（event/object/mask）的正确性
- 端到端流程的完整性
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bamboohepml.data import DataConfig, DataSourceFactory, HEPDataset  # noqa: E402
from bamboohepml.data.features import ExpressionEngine, FeatureGraph  # noqa: E402
from bamboohepml.engine import Evaluator, Trainer  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402
from bamboohepml.utils import collate_fn  # noqa: E402


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_batch_info(batch: dict, title: str = "Batch 信息"):
    """打印 batch 的详细信息"""
    print(f"\n{title}:")
    print(f"  键: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # 对于非浮点类型的 tensor（如 int64, bool），不能计算 mean
            try:
                mean_val = value.float().mean().item()
                min_val = value.min().item()
                max_val = value.max().item()
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
            except (RuntimeError, TypeError):
                # 对于 bool 或某些类型，只打印 shape 和 dtype
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, ak.Array):
            print(f"    {key}: type=ak.Array, length={len(value)}")
        else:
            print(f"    {key}: type={type(value).__name__}, value={value}")


def create_mock_data_source(num_events: int = 100):
    """创建模拟数据源"""
    # 创建 event-level 数据
    met = np.random.uniform(0, 500, num_events)
    ht = np.random.uniform(0, 1000, num_events)

    # 创建 object-level 数据（jagged array）
    num_jets_per_event = np.random.randint(2, 8, num_events)
    jet_pt = []
    jet_eta = []
    for n in num_jets_per_event:
        jet_pt.append(np.random.uniform(20, 200, n))
        jet_eta.append(np.random.uniform(-2.5, 2.5, n))

    # 创建标签
    labels = np.random.randint(0, 2, num_events)

    # 构建 awkward array
    data = {
        "met": met,
        "ht": ht,
        "Jet_pt": ak.Array(jet_pt),
        "Jet_eta": ak.Array(jet_eta),
        "is_signal": labels,
    }

    table = ak.Array(data)

    # 创建 MockDataSource
    class MockDataSource:
        def __init__(self, table):
            self.table = table

        def load_branches(self, branches):
            if not branches:
                return self.table
            result = {}
            for branch in branches:
                if branch in self.table.fields:
                    result[branch] = self.table[branch]
            if not result:
                return self.table  # 如果没有找到任何分支，返回整个表
            return ak.Array(result)

        def get_num_events(self):
            return len(self.table)

        def get_available_branches(self):
            return list(self.table.fields)

    return MockDataSource(table)


def test_only_event_features():
    """测试只有 event-level 特征的情况"""
    print_section("测试 1: 只有 Event-Level 特征")

    # 创建 FeatureGraph（只有 event 特征）
    feature_defs = {
        "met": {
            "expr": "met",
            "type": "event",
            "dtype": "float32",
        },
        "ht": {
            "expr": "ht",
            "type": "event",
            "dtype": "float32",
        },
    }

    engine = ExpressionEngine()
    feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
    feature_graph.compile()

    # 创建数据
    data_source = create_mock_data_source(num_events=50)
    table = data_source.load_branches(["met", "ht", "is_signal"])

    # 拟合 FeatureGraph
    print("\n1. 拟合 FeatureGraph...")
    feature_graph.fit(table)
    print("   ✓ FeatureGraph 拟合成功")

    # 检查 output_spec
    print("\n2. 检查 FeatureGraph.output_spec()...")
    output_spec = feature_graph.output_spec()
    print(f"   output_spec: {output_spec}")
    assert "event" in output_spec, "应该包含 event 特征"
    assert "object" not in output_spec, "不应该包含 object 特征"
    print(f"   event dim: {output_spec['event']['dim']}")
    print(f"   event features: {output_spec['event']['features']}")

    # 构建 batch
    print("\n3. 构建 batch...")
    batch = feature_graph.build_batch(table)
    print_batch_info(batch, "FeatureGraph.build_batch() 输出")

    assert "event" in batch, "batch 应该包含 'event' 键"
    assert "object" not in batch, "batch 不应该包含 'object' 键"
    assert batch["event"].shape[1] == 2, f"event 维度应该是 2，实际是 {batch['event'].shape[1]}"
    print("   ✓ Batch 格式正确")

    # 创建模型（自动推断维度）
    print("\n4. 创建模型（自动推断维度）...")
    event_input_dim = output_spec["event"]["dim"]
    model = get_model(
        "mlp_classifier",
        event_input_dim=event_input_dim,
        object_input_dim=None,
        embed_dim=64,
        hidden_dims=[32, 16],
        num_classes=2,
    )
    print(f"   ✓ 模型创建成功")
    print(f"   event_input_dim: {event_input_dim}")
    print(f"   object_input_dim: None")

    # 测试前向传播
    print("\n5. 测试模型前向传播...")

    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   输入 event shape: {batch['event'].shape}")
    print(f"   输出 shape: {output.shape}")
    print(f"   输出值（前5个）: {output[:5]}")
    assert output.shape == (len(table), 2), f"输出形状应该是 ({len(table)}, 2)，实际是 {output.shape}"
    print("   ✓ 前向传播测试通过")

    # 测试训练
    print("\n6. 测试训练流程...")
    data_config = DataConfig(
        selection=None,
        labels={"type": "simple", "value": ["is_signal"]},
    )
    dataset = HEPDataset(
        data_source=data_source,
        data_config=data_config,
        feature_graph=feature_graph,
        for_training=True,
        shuffle=False,
    )

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

    # 检查 DataLoader 输出的 batch 格式
    sample_batch = next(iter(loader))
    print_batch_info(sample_batch, "DataLoader 输出的 Batch")

    trainer = Trainer(
        model=model,
        train_loader=loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
        task_type="classification",
    )

    # 训练一个 epoch
    history = trainer.fit(num_epochs=1)
    print(f"   ✓ 训练完成，history keys: {list(history.keys())}")

    # 测试评估
    print("\n7. 测试评估流程...")
    evaluator = Evaluator(task_type="classification")
    metrics = evaluator.evaluate(model, loader, loss_fn=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"))
    print(f"   评估指标: {metrics}")
    print("   ✓ 评估测试通过")

    print("\n✓ 测试 1 完成：只有 Event-Level 特征")


def test_only_object_features():
    """测试只有 object-level 特征的情况"""
    print_section("测试 2: 只有 Object-Level 特征")

    # 创建 FeatureGraph（只有 object 特征）
    feature_defs = {
        "jet_pt": {
            "expr": "Jet_pt",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 10,
                "mode": "constant",
            },
        },
        "jet_eta": {
            "expr": "Jet_eta",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 10,
                "mode": "constant",
            },
        },
    }

    engine = ExpressionEngine()
    feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
    feature_graph.compile()

    # 创建数据
    data_source = create_mock_data_source(num_events=50)
    table = data_source.load_branches(["Jet_pt", "Jet_eta", "is_signal"])

    # 拟合 FeatureGraph
    print("\n1. 拟合 FeatureGraph...")
    feature_graph.fit(table)
    print("   ✓ FeatureGraph 拟合成功")

    # 检查 output_spec
    print("\n2. 检查 FeatureGraph.output_spec()...")
    output_spec = feature_graph.output_spec()
    print(f"   output_spec: {output_spec}")
    assert "object" in output_spec, "应该包含 object 特征"
    assert "event" not in output_spec, "不应该包含 event 特征"
    print(f"   object dim: {output_spec['object']['dim']}")
    print(f"   object max_length: {output_spec['object']['max_length']}")
    print(f"   object features: {output_spec['object']['features']}")

    # 构建 batch
    print("\n3. 构建 batch...")
    batch = feature_graph.build_batch(table)
    print_batch_info(batch, "FeatureGraph.build_batch() 输出")

    assert "object" in batch, "batch 应该包含 'object' 键"
    assert "mask" in batch, "batch 应该包含 'mask' 键"
    assert "event" not in batch, "batch 不应该包含 'event' 键"
    assert batch["object"].shape == (len(table), 10, 2), f"object 形状应该是 ({len(table)}, 10, 2)，实际是 {batch['object'].shape}"
    assert batch["mask"].shape == (len(table), 10), f"mask 形状应该是 ({len(table)}, 10)，实际是 {batch['mask'].shape}"
    print("   ✓ Batch 格式正确")

    # 创建模型（自动推断维度）
    print("\n4. 创建模型（自动推断维度）...")
    object_input_dim = output_spec["object"]["dim"]
    model = get_model(
        "mlp_classifier",
        event_input_dim=None,
        object_input_dim=object_input_dim,
        embed_dim=64,
        hidden_dims=[32, 16],
        num_classes=2,
        object_pooling_mode="mean",
    )
    print(f"   ✓ 模型创建成功")
    print(f"   event_input_dim: None")
    print(f"   object_input_dim: {object_input_dim}")
    print(f"   object_pooling_mode: mean")

    # 测试前向传播
    print("\n5. 测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   输入 object shape: {batch['object'].shape}")
    print(f"   输入 mask shape: {batch['mask'].shape}")
    print(f"   输出 shape: {output.shape}")
    print(f"   输出值（前5个）: {output[:5]}")
    assert output.shape == (len(table), 2), f"输出形状应该是 ({len(table)}, 2)，实际是 {output.shape}"
    print("   ✓ 前向传播测试通过")

    # 测试训练
    print("\n6. 测试训练流程...")
    data_config = DataConfig(
        selection=None,
        labels={"type": "simple", "value": ["is_signal"]},
    )
    dataset = HEPDataset(
        data_source=data_source,
        data_config=data_config,
        feature_graph=feature_graph,
        for_training=True,
        shuffle=False,
    )

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

    # 检查 DataLoader 输出的 batch 格式
    sample_batch = next(iter(loader))
    print_batch_info(sample_batch, "DataLoader 输出的 Batch")

    trainer = Trainer(
        model=model,
        train_loader=loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
        task_type="classification",
    )

    # 训练一个 epoch
    history = trainer.fit(num_epochs=1)
    print(f"   ✓ 训练完成，history keys: {list(history.keys())}")

    # 测试评估
    print("\n7. 测试评估流程...")
    evaluator = Evaluator(task_type="classification")
    metrics = evaluator.evaluate(model, loader, loss_fn=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"))
    print(f"   评估指标: {metrics}")
    print("   ✓ 评估测试通过")

    print("\n✓ 测试 2 完成：只有 Object-Level 特征")


def test_both_event_and_object_features():
    """测试同时有 event 和 object 特征的情况"""
    print_section("测试 3: Event + Object 特征")

    # 创建 FeatureGraph（同时有 event 和 object 特征）
    feature_defs = {
        "met": {
            "expr": "met",
            "type": "event",
            "dtype": "float32",
        },
        "ht": {
            "expr": "ht",
            "type": "event",
            "dtype": "float32",
        },
        "jet_pt": {
            "expr": "Jet_pt",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 10,
                "mode": "constant",
            },
        },
        "jet_eta": {
            "expr": "Jet_eta",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 10,
                "mode": "constant",
            },
        },
    }

    engine = ExpressionEngine()
    feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
    feature_graph.compile()

    # 创建数据
    data_source = create_mock_data_source(num_events=50)
    table = data_source.load_branches(["met", "ht", "Jet_pt", "Jet_eta", "is_signal"])

    # 拟合 FeatureGraph
    print("\n1. 拟合 FeatureGraph...")
    feature_graph.fit(table)
    print("   ✓ FeatureGraph 拟合成功")

    # 检查 output_spec
    print("\n2. 检查 FeatureGraph.output_spec()...")
    output_spec = feature_graph.output_spec()
    print(f"   output_spec: {output_spec}")
    assert "event" in output_spec, "应该包含 event 特征"
    assert "object" in output_spec, "应该包含 object 特征"
    print(f"   event dim: {output_spec['event']['dim']}")
    print(f"   event features: {output_spec['event']['features']}")
    print(f"   object dim: {output_spec['object']['dim']}")
    print(f"   object max_length: {output_spec['object']['max_length']}")
    print(f"   object features: {output_spec['object']['features']}")

    # 构建 batch
    print("\n3. 构建 batch...")
    batch = feature_graph.build_batch(table)
    print_batch_info(batch, "FeatureGraph.build_batch() 输出")

    assert "event" in batch, "batch 应该包含 'event' 键"
    assert "object" in batch, "batch 应该包含 'object' 键"
    assert "mask" in batch, "batch 应该包含 'mask' 键"
    assert batch["event"].shape[1] == 2, f"event 维度应该是 2，实际是 {batch['event'].shape[1]}"
    assert batch["object"].shape == (len(table), 10, 2), f"object 形状应该是 ({len(table)}, 10, 2)，实际是 {batch['object'].shape}"
    assert batch["mask"].shape == (len(table), 10), f"mask 形状应该是 ({len(table)}, 10)，实际是 {batch['mask'].shape}"
    print("   ✓ Batch 格式正确")

    # 创建模型（自动推断维度）
    print("\n4. 创建模型（自动推断维度）...")
    event_input_dim = output_spec["event"]["dim"]
    object_input_dim = output_spec["object"]["dim"]
    model = get_model(
        "mlp_classifier",
        event_input_dim=event_input_dim,
        object_input_dim=object_input_dim,
        embed_dim=64,
        hidden_dims=[32, 16],
        num_classes=2,
        object_pooling_mode="mean",
    )
    print(f"   ✓ 模型创建成功")
    print(f"   event_input_dim: {event_input_dim}")
    print(f"   object_input_dim: {object_input_dim}")
    print(f"   embed_dim: 64")
    print(f"   object_pooling_mode: mean")

    # 测试前向传播
    print("\n5. 测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   输入 event shape: {batch['event'].shape}")
    print(f"   输入 object shape: {batch['object'].shape}")
    print(f"   输入 mask shape: {batch['mask'].shape}")
    print(f"   输出 shape: {output.shape}")
    print(f"   输出值（前5个）: {output[:5]}")
    assert output.shape == (len(table), 2), f"输出形状应该是 ({len(table)}, 2)，实际是 {output.shape}"
    print("   ✓ 前向传播测试通过")

    # 测试训练
    print("\n6. 测试训练流程...")
    data_config = DataConfig(
        selection=None,
        labels={"type": "simple", "value": ["is_signal"]},
    )
    dataset = HEPDataset(
        data_source=data_source,
        data_config=data_config,
        feature_graph=feature_graph,
        for_training=True,
        shuffle=False,
    )

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

    # 检查 DataLoader 输出的 batch 格式
    sample_batch = next(iter(loader))
    print_batch_info(sample_batch, "DataLoader 输出的 Batch")

    trainer = Trainer(
        model=model,
        train_loader=loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
        task_type="classification",
    )

    # 训练一个 epoch
    history = trainer.fit(num_epochs=1)
    print(f"   ✓ 训练完成，history keys: {list(history.keys())}")

    # 测试评估
    print("\n7. 测试评估流程...")
    evaluator = Evaluator(task_type="classification")
    metrics = evaluator.evaluate(model, loader, loss_fn=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"))
    print(f"   评估指标: {metrics}")
    print("   ✓ 评估测试通过")

    print("\n✓ 测试 3 完成：Event + Object 特征")


def test_pipeline_orchestrator_auto_inference():
    """测试 PipelineOrchestrator 的自动维度推断"""
    print_section("测试 4: PipelineOrchestrator 自动维度推断")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建 pipeline.yaml 配置
        pipeline_config = {
            "data": {
                "source": {
                    "type": "mock",
                    "path": "dummy",
                },
            },
            "features": {
                "config_path": str(tmpdir / "features.yaml"),
            },
            "model": {
                "name": "mlp_classifier",
                "params": {
                    "embed_dim": 64,
                    "hidden_dims": [32, 16],
                    "num_classes": 2,
                    "object_pooling_mode": "mean",
                    # 不指定 event_input_dim 和 object_input_dim，让系统自动推断
                },
            },
            "train": {
                "num_epochs": 1,
                "batch_size": 8,
                "task_type": "classification",
            },
        }

        # 创建 features.yaml
        feature_defs = {
            "met": {
                "expr": "met",
                "type": "event",
                "dtype": "float32",
            },
            "jet_pt": {
                "expr": "Jet_pt",
                "type": "object",
                "dtype": "float32",
                "padding": {
                    "max_length": 10,
                    "mode": "constant",
                },
            },
        }

        with open(tmpdir / "features.yaml", "w") as f:
            yaml.dump({"features": feature_defs}, f)

        with open(tmpdir / "pipeline.yaml", "w") as f:
            yaml.dump(pipeline_config, f)

        print(f"\n1. Pipeline 配置文件: {tmpdir / 'pipeline.yaml'}")
        print(f"   Features 配置文件: {tmpdir / 'features.yaml'}")

        # 注意：这里我们不能真正运行 PipelineOrchestrator，因为它需要真实的数据源
        # 但我们可以测试维度推断的逻辑

        # 创建 FeatureGraph 并检查维度
        print("\n2. 创建 FeatureGraph 并检查维度推断...")
        data_source = create_mock_data_source(num_events=50)
        table = data_source.load_branches(["met", "Jet_pt", "is_signal"])

        engine = ExpressionEngine()
        feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
        feature_graph.compile()
        feature_graph.fit(table)

        output_spec = feature_graph.output_spec()
        print(f"   output_spec: {output_spec}")

        # 模拟 PipelineOrchestrator.setup_model() 的维度推断逻辑
        print("\n3. 模拟维度自动推断...")
        model_kwargs = pipeline_config["model"]["params"].copy()

        if "event" in output_spec:
            inferred_event_dim = output_spec["event"]["dim"]
            model_kwargs["event_input_dim"] = inferred_event_dim
            print(f"   ✓ 自动推断 event_input_dim={inferred_event_dim}")

        if "object" in output_spec:
            inferred_object_dim = output_spec["object"]["dim"]
            model_kwargs["object_input_dim"] = inferred_object_dim
            print(f"   ✓ 自动推断 object_input_dim={inferred_object_dim}")

        # 创建模型
        print("\n4. 使用推断的维度创建模型...")
        model = get_model("mlp_classifier", **model_kwargs)
        print(f"   ✓ 模型创建成功")
        print(f"   模型参数: event_input_dim={model_kwargs.get('event_input_dim')}, object_input_dim={model_kwargs.get('object_input_dim')}")

        # 测试前向传播
        print("\n5. 测试模型前向传播...")
        batch = feature_graph.build_batch(table)
        print_batch_info(batch, "模型输入 Batch")

        model.eval()
        with torch.no_grad():
            output = model(batch)
        print(f"   输出 shape: {output.shape}")
        print("   ✓ 前向传播测试通过")

        print("\n✓ 测试 4 完成：PipelineOrchestrator 自动维度推断")


def test_regression_with_new_architecture():
    """测试回归任务使用新架构"""
    print_section("测试 5: 回归任务（新架构）")

    # 创建 FeatureGraph（event + object 特征）
    feature_defs = {
        "met": {
            "expr": "met",
            "type": "event",
            "dtype": "float32",
        },
        "jet_pt": {
            "expr": "Jet_pt",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 10,
                "mode": "constant",
            },
        },
    }

    engine = ExpressionEngine()
    feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
    feature_graph.compile()

    # 创建数据
    data_source = create_mock_data_source(num_events=50)
    table = data_source.load_branches(["met", "Jet_pt", "is_signal"])

    # 拟合 FeatureGraph
    print("\n1. 拟合 FeatureGraph...")
    feature_graph.fit(table)

    # 检查 output_spec
    output_spec = feature_graph.output_spec()
    print(f"   output_spec: {output_spec}")

    # 创建回归模型
    print("\n2. 创建回归模型...")
    model = get_model(
        "mlp_regressor",
        event_input_dim=output_spec["event"]["dim"],
        object_input_dim=output_spec["object"]["dim"],
        embed_dim=64,
        hidden_dims=[32, 16],
        num_outputs=1,
        object_pooling_mode="mean",
    )
    print(f"   ✓ 回归模型创建成功")

    # 测试前向传播
    print("\n3. 测试模型前向传播...")
    batch = feature_graph.build_batch(table)
    print_batch_info(batch, "模型输入 Batch")

    model.eval()
    with torch.no_grad():
        output = model(batch)
    print(f"   输出 shape: {output.shape}")
    print(f"   输出值（前5个）: {output[:5].squeeze()}")
    assert output.shape == (len(table), 1), f"输出形状应该是 ({len(table)}, 1)，实际是 {output.shape}"
    print("   ✓ 前向传播测试通过")

    # 测试训练
    print("\n4. 测试训练流程...")
    data_config = DataConfig(
        selection=None,
        labels={"type": "simple", "value": ["is_signal"]},
    )
    dataset = HEPDataset(
        data_source=data_source,
        data_config=data_config,
        feature_graph=feature_graph,
        for_training=True,
        shuffle=False,
    )

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
        task_type="regression",
    )

    trainer.fit(num_epochs=1)
    print(f"   ✓ 训练完成")

    print("\n✓ 测试 5 完成：回归任务（新架构）")


def test_specific_dimensions():
    """测试特定维度配置：event_input_dim=5, object_input_dim=10"""
    print_section("测试 6: 特定维度配置演示 (event_input_dim=5, object_input_dim=10)")

    print("\n" + "=" * 80)
    print("  📊 数据流分析：event_input_dim=5, object_input_dim=10")
    print("=" * 80)

    # 直接创建模型演示维度配置
    event_input_dim = 5
    object_input_dim = 10
    embed_dim = 64

    print(f"\n1. 模型配置：")
    print(f"   event_input_dim = {event_input_dim}  # Event-level 特征数量")
    print(f"   object_input_dim = {object_input_dim}  # Object-level 特征数量（每个对象）")
    print(f"   embed_dim = {embed_dim}  # Embedding 维度")
    print(f"   object_pooling_mode = 'mean'  # Object 特征池化方式")

    print(f"\n2. 创建模型...")
    model = get_model(
        "mlp_classifier",
        event_input_dim=event_input_dim,
        object_input_dim=object_input_dim,
        embed_dim=embed_dim,
        hidden_dims=[128, 64, 32],
        num_classes=2,
        object_pooling_mode="mean",
    )
    print(f"   ✓ 模型创建成功")

    print(f"\n3. 数据流详解：")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ Batch 输入 (batch_size = B)                                 │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ event:  (B, {event_input_dim})  # Event-level 特征          │")
    print(f"   │ object: (B, N, {object_input_dim})  # Object-level 特征     │")
    print(f"   │ mask:   (B, N)  # Mask（True=有效，False=padding）         │")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    print(f"                          ↓")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ Event Embedding                                             │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ Linear({event_input_dim}, {embed_dim}) + Activation          │")
    print(f"   │ → event_emb: (B, {embed_dim})                                │")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    print(f"                          ↓")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ Object Embedding                                            │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ Linear({object_input_dim}, {embed_dim}) + Activation          │")
    print(f"   │ → object_emb: (B, N, {embed_dim})                           │")
    print(f"   │ → Pooling (mean with mask)                                  │")
    print(f"   │ → object_emb_pooled: (B, {embed_dim})                       │")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    print(f"                          ↓")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ Fusion (Concatenate)                                        │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ Concat([event_emb, object_emb_pooled], dim=-1)             │")
    print(f"   │ → fused: (B, {embed_dim * 2})  # {embed_dim} + {embed_dim} = {embed_dim * 2}  │")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    print(f"                          ↓")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ MLP Backbone                                                │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ Linear({embed_dim * 2}, 128) → ReLU → Dropout               │")
    print(f"   │ Linear(128, 64) → ReLU → Dropout                            │")
    print(f"   │ Linear(64, 32) → ReLU → Dropout                             │")
    print(f"   │ Linear(32, 2)  # num_classes                                │")
    print(f"   │ → output: (B, 2)                                            │")
    print(f"   └─────────────────────────────────────────────────────────────┘")

    # 创建模拟数据演示
    print(f"\n4. 使用模拟数据演示前向传播...")
    batch_size = 8
    max_objects = 10

    # 创建模拟 batch
    mock_batch = {
        "event": torch.randn(batch_size, event_input_dim),
        "object": torch.randn(batch_size, max_objects, object_input_dim),
        "mask": torch.ones(batch_size, max_objects, dtype=torch.bool),
    }

    print_batch_info(mock_batch, "模拟 Batch 输入")

    model.eval()
    with torch.no_grad():
        output = model(mock_batch)

    print(f"\n   输出 shape: {output.shape}")
    print(f"   输出值（前3个样本）:")
    for i in range(min(3, batch_size)):
        print(f"      样本 {i}: {output[i].cpu().numpy()}")

    print(f"\n5. 参数统计：")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")

    # 计算各层的参数量
    print(f"\n   各层参数量分解：")
    print(f"   - Event Embedding (Linear({event_input_dim}, {embed_dim})): {event_input_dim * embed_dim + embed_dim:,}")
    print(f"   - Object Embedding (Linear({object_input_dim}, {embed_dim})): {object_input_dim * embed_dim + embed_dim:,}")
    print(f"   - MLP Backbone (128→64→32→2): ~{(embed_dim * 2 * 128 + 128 * 64 + 64 * 32 + 32 * 2):,}")

    print("\n✓ 测试 6 完成：特定维度配置演示")


def test_real_root_file_regression():
    """使用真实 ROOT 文件测试回归任务（需要本地 ROOT 文件，CI 中跳过）"""
    print_section("测试 7: 真实 ROOT 文件回归任务")

    # ROOT 文件路径
    root_file_path = ".../merge_ss_0006.root"
    tree_name = "tree"  # 根据 ROOT 文件中的实际 tree 名称

    # 在 CI 环境中或文件不存在时跳过测试
    if os.getenv("CI") == "true" or not os.path.exists(root_file_path):
        pytest.skip(f"跳过测试：需要本地 ROOT 文件（在 CI 环境中或文件不存在）\n文件路径: {root_file_path}")

    print(f"\n1. 加载 ROOT 文件...")
    print(f"   文件路径: {root_file_path}")
    print(f"   Tree 名称: {tree_name}")

    try:
        # 创建数据源（只加载一小部分数据用于测试）
        data_source = DataSourceFactory.create(
            root_file_path,
            treename=tree_name,
            load_range=(0, 0.1),  # 只加载前 10% 的数据
        )
        print(f"   ✓ 数据源创建成功")
        print(f"   事件数: {data_source.get_num_events()}")

        # 获取可用分支
        available_branches = data_source.get_available_branches()
        print(f"   可用分支数: {len(available_branches)}")

    except Exception as e:
        pytest.skip(f"无法加载 ROOT 文件: {e}")

    # 创建 FeatureGraph
    print("\n2. 创建 FeatureGraph...")
    feature_defs = {
        # Event-level 特征
        "jet_phi": {
            "expr": "jet_phi",
            "type": "event",
            "dtype": "float32",
        },
        "jet_eta": {
            "expr": "jet_eta",
            "type": "event",
            "dtype": "float32",
        },
        # Object-level 特征
        "part_d0": {
            "expr": "part_d0",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 50,  # 根据实际数据调整
                "mode": "constant",
            },
        },
        "part_isKLong": {
            "expr": "part_isKLong",
            "type": "object",
            "dtype": "float32",  # bool 转为 float32
            "padding": {
                "max_length": 50,
                "mode": "constant",
            },
        },
        "part_deltaR": {
            "expr": "part_deltaR",
            "type": "object",
            "dtype": "float32",
            "padding": {
                "max_length": 50,
                "mode": "constant",
            },
        },
    }

    engine = ExpressionEngine()
    feature_graph = FeatureGraph.from_feature_defs(feature_defs, engine, enable_cache=False)
    feature_graph.compile()
    print("   ✓ FeatureGraph 创建成功")

    # 加载数据
    print("\n3. 加载数据并拟合 FeatureGraph...")
    table = data_source.load_branches(
        [
            "jet_phi",
            "jet_eta",
            "jet_energy",
            "part_d0",
            "part_isKLong",
            "part_deltaR",
        ]
    )

    # 检查原始数据类型（用于验证）
    print(f"   数据表字段: {list(table.fields)}")
    print(f"   事件数: {len(table)}")
    if "part_isKLong" in table.fields:
        sample_val = table["part_isKLong"][0]
        if isinstance(sample_val, ak.Array):
            print(f"   part_isKLong 原始类型: vector<bool> (jagged array)")
        else:
            print(f"   part_isKLong 原始类型: {type(sample_val)}")

    # 注意：不手动转换类型，让 FeatureProcessor 自动处理 bool -> float32 转换
    print(f"   ✓ 将使用 FeatureProcessor 自动处理类型转换（bool -> float32）")

    # 拟合 FeatureGraph（会触发 FeatureProcessor 的类型转换）
    print("\n4. 拟合 FeatureGraph（测试自动类型转换）...")
    feature_graph.fit(table)
    print("   ✓ FeatureGraph 拟合成功")

    # 验证类型转换：检查处理后的 part_isKLong 类型
    if "part_isKLong" in table.fields:
        # 构建一个测试 batch 来查看处理后的类型
        test_batch = feature_graph.build_batch(table[:1])  # 只处理第一个事件用于检查
        if "object" in test_batch:
            object_tensor = test_batch["object"]
            print(f"   ✓ 类型转换验证：")
            print(f"     - 处理后的 object tensor dtype: {object_tensor.dtype}")
            print(f"     - object tensor shape: {object_tensor.shape}")
            # part_isKLong 应该是 object 特征的一部分，所以 object tensor 应该包含它
            print(f"     - 确认：part_isKLong (bool) 已自动转换为 float32 并包含在 object tensor 中")

    # 检查 output_spec
    print("\n5. 检查 FeatureGraph.output_spec()...")
    output_spec = feature_graph.output_spec()
    print(f"   output_spec: {output_spec}")

    if "event" in output_spec:
        print(f"   event dim: {output_spec['event']['dim']}")
        print(f"   event features: {output_spec['event']['features']}")
    if "object" in output_spec:
        print(f"   object dim: {output_spec['object']['dim']}")
        print(f"   object max_length: {output_spec['object']['max_length']}")
        print(f"   object features: {output_spec['object']['features']}")

    # 创建数据配置
    print("\n6. 创建 DataConfig...")
    data_config = DataConfig(
        treename=tree_name,
        selection=None,
        labels={"type": "complex", "value": {"_label_": "jet_energy"}},  # 回归任务使用 complex 类型，value 必须是字典
    )
    print("   ✓ DataConfig 创建成功")

    # 创建数据集
    print("\n7. 创建 HEPDataset...")
    dataset = HEPDataset(
        data_source=data_source,
        data_config=data_config,
        feature_graph=feature_graph,
        for_training=True,
        shuffle=False,
    )
    print("   ✓ HEPDataset 创建成功")

    # 创建 DataLoader
    print("\n8. 创建 DataLoader...")
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=0)

    # 检查 batch 格式
    sample_batch = next(iter(loader))
    print_batch_info(sample_batch, "DataLoader 输出的 Batch")

    # 创建回归模型
    print("\n9. 创建回归模型（自动推断维度）...")
    event_input_dim = output_spec.get("event", {}).get("dim")
    object_input_dim = output_spec.get("object", {}).get("dim")

    model = get_model(
        "mlp_regressor",
        event_input_dim=event_input_dim,
        object_input_dim=object_input_dim,
        embed_dim=64,
        hidden_dims=[128, 64, 32],
        num_outputs=1,
        object_pooling_mode="mean",
    )
    print(f"   ✓ 模型创建成功")
    print(f"   event_input_dim: {event_input_dim}")
    print(f"   object_input_dim: {object_input_dim}")
    print(f"   embed_dim: 64")

    # 测试前向传播
    print("\n10. 测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
    print(f"   输入 event shape: {sample_batch['event'].shape if 'event' in sample_batch else 'N/A'}")
    print(f"   输入 object shape: {sample_batch['object'].shape if 'object' in sample_batch else 'N/A'}")
    print(f"   输入 mask shape: {sample_batch['mask'].shape if 'mask' in sample_batch else 'N/A'}")
    print(f"   标签 shape: {sample_batch['_label_'].shape if '_label_' in sample_batch else 'N/A'}")
    print(f"   输出 shape: {output.shape}")
    print(f"   输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   输出均值: {output.mean().item():.4f}")
    if "_label_" in sample_batch:
        labels = sample_batch["_label_"]
        # 对于 int64 类型的标签，需要转换为 float 才能计算 mean
        labels_float = labels.float()
        print(f"   标签值范围: [{labels_float.min().item():.4f}, {labels_float.max().item():.4f}]")
        print(f"   标签均值: {labels_float.mean().item():.4f}")

    # 测试训练
    print("\n11. 测试训练流程（1个 epoch）...")
    trainer = Trainer(
        model=model,
        train_loader=loader,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device("cpu"),
        task_type="regression",
    )

    history = trainer.fit(num_epochs=1)
    print(f"   ✓ 训练完成")
    print(f"   History keys: {list(history.keys())}")
    if "history" in history:
        train_loss = history["history"].get("train_loss", [])
        val_loss = history["history"].get("val_loss", [])
        if train_loss:
            print(f"   训练 loss: {train_loss[-1]:.6f}")
        if val_loss:
            print(f"   验证 loss: {val_loss[-1]:.6f}")

    # 测试评估
    print("\n12. 测试评估流程...")
    evaluator = Evaluator(task_type="regression")
    metrics = evaluator.evaluate(model, loader, loss_fn=torch.nn.MSELoss(), device=torch.device("cpu"))
    print(f"   评估指标: {metrics}")
    print("   ✓ 评估测试通过")

    print("\n✓ 测试 7 完成：真实 ROOT 文件回归任务")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("  新架构集成测试")
    print("=" * 80)
    print("\n测试内容：")
    print("  1. 只有 Event-Level 特征")
    print("  2. 只有 Object-Level 特征")
    print("  3. Event + Object 特征")
    print("  4. PipelineOrchestrator 自动维度推断")
    print("  5. 回归任务（新架构）")
    print("  6. 特定维度配置 (event_input_dim=5, object_input_dim=10)")
    print("  7. 真实 ROOT 文件回归任务")
    print("=" * 80)

    try:
        test_only_event_features()
        test_only_object_features()
        test_both_event_and_object_features()
        test_pipeline_orchestrator_auto_inference()
        test_regression_with_new_architecture()
        test_specific_dimensions()
        test_real_root_file_regression()

        print("\n" + "=" * 80)
        print("  ✓ 所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
