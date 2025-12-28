"""
Dataset 测试

测试：
1. 数据源创建和加载
2. Dataset 基本功能
3. Jagged array 支持
4. Padding + Mask
5. Transformer 格式
6. 与特征系统集成
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import awkward as ak  # noqa: E402

# 导入必须在路径设置之后
import numpy as np  # noqa: E402

from bamboohepml.data import DataConfig, HEPDataset, TransformerDataset  # noqa: E402
from bamboohepml.data.features import ExpressionEngine, FeatureGraph  # noqa: E402


def create_mock_data_source():
    """创建模拟数据源（用于测试）。"""
    # 创建模拟的 ROOT 数据
    n_events = 100

    # Event-level 数据
    met = np.random.uniform(20, 200, n_events).astype(np.float32)

    # Object-level 数据（jagged array）
    jet_pt_list = []
    jet_eta_list = []
    jet_phi_list = []
    jet_mass_list = []

    for i in range(n_events):
        n_jets = np.random.randint(2, 10)
        jet_pt_list.append(np.random.uniform(20, 150, n_jets).astype(np.float32))
        jet_eta_list.append(np.random.uniform(-3, 3, n_jets).astype(np.float32))
        jet_phi_list.append(np.random.uniform(-np.pi, np.pi, n_jets).astype(np.float32))
        jet_mass_list.append(np.random.uniform(5, 20, n_jets).astype(np.float32))

    # 构建 Jet 对象
    Jet = ak.zip(
        {
            "pt": ak.Array(jet_pt_list),
            "eta": ak.Array(jet_eta_list),
            "phi": ak.Array(jet_phi_list),
            "mass": ak.Array(jet_mass_list),
        }
    )

    # 构建数据表
    table = ak.Array(
        {
            "met": met,
            "Jet": Jet,
            "is_signal": np.random.randint(0, 2, n_events).astype(bool),
        }
    )

    return table


class MockDataSource:
    """模拟数据源（用于测试）。"""

    def __init__(self, table):
        self.table = table
        self._file_paths = ["mock://data"]

    def load_branches(self, branches):
        """加载分支。"""
        result = {}
        for branch in branches:
            if branch in self.table.fields:
                result[branch] = self.table[branch]
            elif branch == "Jet" and "Jet" in self.table.fields:
                result["Jet"] = self.table["Jet"]
        return ak.Array(result)

    def get_available_branches(self):
        """获取可用分支。"""
        return list(self.table.fields)

    def get_file_paths(self):
        """获取文件路径。"""
        return self._file_paths

    def get_num_events(self):
        """获取事件数量。"""
        return len(self.table)


def test_data_source():
    """测试数据源。"""
    print("=" * 60)
    print("测试 1: 数据源")
    print("=" * 60)

    # 创建模拟数据
    table = create_mock_data_source()
    source = MockDataSource(table)

    # 测试加载分支
    branches = ["met", "Jet"]
    loaded = source.load_branches(branches)

    print(f"加载了 {len(loaded)} 个事件")
    print(f"可用分支: {source.get_available_branches()}")

    assert "met" in loaded.fields
    assert "Jet" in loaded.fields

    print("✓ 数据源测试通过\n")


def test_jagged_array():
    """测试 Jagged Array 支持。"""
    print("=" * 60)
    print("测试 2: Jagged Array 支持")
    print("=" * 60)

    table = create_mock_data_source()

    # 检查每个事件的 Jet 数量
    jet_counts = ak.num(table["Jet"])
    print(f"Jet 数量范围: {ak.min(jet_counts)} - {ak.max(jet_counts)}")
    print(f"平均 Jet 数量: {ak.mean(jet_counts):.2f}")

    # 检查变长数组
    assert ak.min(jet_counts) >= 2
    assert ak.max(jet_counts) <= 10

    print("✓ Jagged Array 测试通过\n")


def test_padding_mask():
    """测试 Padding + Mask。"""
    print("=" * 60)
    print("测试 3: Padding + Mask")
    print("=" * 60)

    from bamboohepml.data.tools import _pad

    # 创建 jagged array
    jet_pt = ak.Array(
        [
            [30.0, 25.0, 20.0],
            [50.0, 40.0],
            [45.0],
        ]
    )

    # Padding 到长度 5
    padded = _pad(jet_pt, maxlen=5, value=0.0)
    print(f"原始形状: {[len(x) for x in jet_pt]}")
    # Convert to numpy to get shape
    if isinstance(padded, ak.Array):
        padded_np = ak.to_numpy(padded)
        print(f"Padding 后形状: {padded_np.shape}")
    else:
        print(f"Padding 后形状: {padded.shape}")

    # 生成 mask
    # Convert to numpy first if needed
    if isinstance(padded, ak.Array):
        padded_np = ak.to_numpy(padded)
    else:
        padded_np = padded
    mask = (padded_np != 0).any(axis=1)  # 假设 padding 值为 0
    print(f"Mask 形状: {mask.shape}")
    print(f"Mask (事件 0): {mask[0]}")
    print(f"Mask (事件 1): {mask[1]}")
    print(f"Mask (事件 2): {mask[2]}")

    assert padded.shape == (3, 5)
    assert mask.shape == (3, 5)

    print("✓ Padding + Mask 测试通过\n")


def test_dataset_basic():
    """测试 Dataset 基本功能。"""
    print("=" * 60)
    print("测试 4: Dataset 基本功能")
    print("=" * 60)

    # 创建数据配置（简化版）
    data_config_dict = {
        "treename": None,
        "selection": None,
        "new_variables": {},
        "inputs": {
            "pf_points": {
                "length": 10,
                "pad_mode": "constant",
                "vars": [["Jet.pt", None]],
            }
        },
        "labels": {
            "type": "simple",
            "value": ["is_signal"],
        },
        "preprocess": {
            "method": "manual",
            "params": {
                "Jet.pt": {
                    "length": 10,
                    "pad_mode": "constant",
                    "center": None,
                    "scale": 1,
                    "min": -5,
                    "max": 5,
                    "pad_value": 0,
                }
            },
        },
    }

    data_config = DataConfig(**data_config_dict)

    # 创建数据源和数据集
    table = create_mock_data_source()
    source = MockDataSource(table)

    dataset = HEPDataset(
        data_source=source,
        data_config=data_config,
        for_training=False,
        shuffle=False,
    )

    # 测试迭代
    count = 0
    for sample in dataset:
        count += 1
        if count >= 3:
            break
        print(f"样本 {count}:")
        print(f"  标签: {sample.get('_label_', 'N/A')}")
        print(f"  输入键: {list(sample.keys())}")

    print(f"成功迭代了 {count} 个样本")
    print("✓ Dataset 基本功能测试通过\n")


def test_transformer_format():
    """测试 Transformer 格式。"""
    print("=" * 60)
    print("测试 5: Transformer 格式")
    print("=" * 60)

    # 创建数据配置
    data_config_dict = {
        "treename": None,
        "selection": None,
        "new_variables": {},
        "inputs": {
            "pf_points": {
                "length": 10,
                "pad_mode": "constant",
                "vars": [["Jet.pt", None]],
            }
        },
        "labels": {
            "type": "simple",
            "value": ["is_signal"],
        },
        "preprocess": {
            "method": "manual",
            "params": {
                "Jet.pt": {
                    "length": 10,
                    "pad_mode": "constant",
                    "center": None,
                    "scale": 1,
                    "min": -5,
                    "max": 5,
                    "pad_value": 0,
                }
            },
        },
    }

    data_config = DataConfig(**data_config_dict)

    # 创建数据源和 Transformer 数据集
    table = create_mock_data_source()
    source = MockDataSource(table)

    transformer_dataset = TransformerDataset(
        data_source=source,
        data_config=data_config,
        for_training=False,
        shuffle=False,
    )

    # 测试 Transformer 格式
    count = 0
    for batch in transformer_dataset:
        count += 1
        if count >= 3:
            break

        print(f"批次 {count}:")
        if "x" in batch:
            print(f"  x 形状: {batch['x'].shape}")
        if "mask" in batch:
            print(f"  mask 形状: {batch['mask'].shape}")
            print(f"  mask (前3个): {batch['mask'][:3]}")
        if "label" in batch or "_label_" in batch:
            label_key = "label" if "label" in batch else "_label_"
            print(f"  标签: {batch[label_key]}")

    print(f"成功迭代了 {count} 个批次")
    print("✓ Transformer 格式测试通过\n")


def test_feature_system_integration():
    """测试与特征系统集成。"""
    print("=" * 60)
    print("测试 6: 与特征系统集成")
    print("=" * 60)

    # 创建表达式引擎和特征图
    engine = ExpressionEngine()

    features = {
        "ht": {
            "expr": "sum(Jet.pt)",
            "type": "event",
            "dtype": "float32",
        },
        "met": {
            "expr": "met",
            "type": "event",
            "dtype": "float32",
        },
    }

    graph = FeatureGraph.from_feature_defs(features, engine)

    # 创建数据配置
    data_config_dict = {
        "treename": None,
        "selection": None,
        "new_variables": {},
        "inputs": {},
        "labels": {
            "type": "simple",
            "value": ["is_signal"],
        },
        "preprocess": {"method": "manual", "params": {}},
    }

    data_config = DataConfig(**data_config_dict)

    # 创建数据集（集成特征系统）
    table = create_mock_data_source()
    source = MockDataSource(table)

    HEPDataset(
        data_source=source,
        data_config=data_config,
        feature_graph=graph,
        expression_engine=engine,
        for_training=False,
    )

    # 测试特征计算
    print("特征图执行顺序:", graph.get_execution_order())
    print("缓存的特征:", list(graph._cache.keys()))

    print("✓ 特征系统集成测试通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dataset 测试套件")
    print("=" * 60 + "\n")

    try:
        test_data_source()
        test_jagged_array()
        test_padding_mask()
        test_dataset_basic()
        test_transformer_format()
        test_feature_system_integration()

        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
