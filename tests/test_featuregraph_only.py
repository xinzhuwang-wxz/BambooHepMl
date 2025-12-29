"""
FeatureGraph-only 测试

测试 FeatureGraph 作为"唯一可信的特征事实源"的功能：
- 不依赖 DataConfig.inputs
- 能够独立生成模型输入
- output_spec() 正确性
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import awkward as ak
import numpy as np
import torch

from bamboohepml.data.features.expression import ExpressionEngine
from bamboohepml.data.features.feature_graph import FeatureGraph


def test_featuregraph_output_spec():
    """测试 FeatureGraph.output_spec() 的正确性。"""
    print("=" * 60)
    print("测试 1: FeatureGraph.output_spec()")
    print("=" * 60)

    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "data" / "features" / "test_features_config.yaml"

    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)

    # 获取 output_spec
    spec = graph.output_spec()

    print(f"Output spec: {spec}")

    # 验证 spec 结构
    assert "event" in spec or "object" in spec, "output_spec must contain 'event' or 'object'"

    if "event" in spec:
        assert "dim" in spec["event"], "event spec must contain 'dim'"
        assert spec["event"]["dim"] > 0, "event dim must be positive"
        print(f"✓ Event-level features: dim={spec['event']['dim']}")

    if "object" in spec:
        assert "dim" in spec["object"], "object spec must contain 'dim'"
        assert "max_length" in spec["object"], "object spec must contain 'max_length'"
        assert spec["object"]["dim"] > 0, "object dim must be positive"
        assert spec["object"]["max_length"] > 0, "object max_length must be positive"
        print(f"✓ Object-level features: dim={spec['object']['dim']}, max_length={spec['object']['max_length']}")

    print("✓ output_spec() 测试通过\n")


def test_featuregraph_build_batch():
    """测试 FeatureGraph.build_batch() 独立生成模型输入。"""
    print("=" * 60)
    print("测试 2: FeatureGraph.build_batch()")
    print("=" * 60)

    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "data" / "features" / "test_features_config.yaml"

    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)

    # 创建模拟数据（不依赖 DataConfig）
    num_events = 10
    # 创建 Jet 对象（特征配置需要 Jet.pt, Jet.eta, Jet.phi, Jet.mass）
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
    
    table = ak.Array(
        {
            "met": np.abs(np.random.randn(num_events) * 50),
            "Jet": Jet,
        }
    )

    # 先拟合特征（因为使用了 auto 归一化）
    graph.fit(table)

    # 使用 FeatureGraph.build_batch() 生成模型输入
    batch = graph.build_batch(table)

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")

    # 验证 batch 结构
    assert isinstance(batch, dict), "batch must be a dict"
    assert len(batch) > 0, "batch must not be empty"

    # 验证所有值都是 torch.Tensor
    for key, value in batch.items():
        assert isinstance(value, torch.Tensor), f"batch['{key}'] must be torch.Tensor"
        print(f"✓ {key}: shape={value.shape}, dtype={value.dtype}")

    # 验证与 output_spec 的一致性
    spec = graph.output_spec()
    if "event" in spec:
        assert "event" in batch, "batch must contain 'event' key when spec has 'event'"
        expected_dim = spec["event"]["dim"]
        assert batch["event"].shape[1] == expected_dim, f"event dim mismatch: {batch['event'].shape[1]} != {expected_dim}"

    print("✓ build_batch() 测试通过\n")


def test_featuregraph_independent_of_dataconfig():
    """测试 FeatureGraph 不依赖 DataConfig.inputs。"""
    print("=" * 60)
    print("测试 3: FeatureGraph 独立性")
    print("=" * 60)

    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "data" / "features" / "test_features_config.yaml"

    # 创建 FeatureGraph（不依赖 DataConfig）
    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)

    # 创建原始数据表（需要包含所有特征依赖的字段）
    num_events = 5
    # 创建 Jet 对象（特征配置需要 Jet 对象来计算 nJets, ht 等特征）
    jet_pt_list = []
    jet_eta_list = []
    jet_phi_list = []
    jet_mass_list = []
    
    for _ in range(num_events):
        n_jets = np.random.randint(3, 8)
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
    
    table = ak.Array(
        {
            "met": np.abs(np.random.randn(num_events) * 50),
            "Jet": Jet,
        }
    )

    # 先拟合特征（因为使用了 auto 归一化）
    graph.fit(table)

    # 直接使用 FeatureGraph 生成模型输入
    batch = graph.build_batch(table)

    # 验证能够生成有效的 batch
    assert "event" in batch or "object" in batch, "batch must contain at least one input key"
    assert all(isinstance(v, torch.Tensor) for v in batch.values()), "all batch values must be torch.Tensor"

    print("✓ FeatureGraph 独立性测试通过\n")


if __name__ == "__main__":
    test_featuregraph_output_spec()
    test_featuregraph_build_batch()
    test_featuregraph_independent_of_dataconfig()
    print("=" * 60)
    print("所有 FeatureGraph-only 测试通过！")
    print("=" * 60)
