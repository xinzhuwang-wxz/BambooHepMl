"""
全方位数据读取测试

测试：
1. 读取 ROOT 文件
2. 不同类型特征（double, vector<float>, vector<bool>）
3. Padding（固定长度、动态长度）
4. 选择条件
5. 新变量构建
6. 预处理配置
7. 与 DataConfig 集成
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
# 确保项目根目录在 PYTHONPATH 中
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import tempfile  # noqa: E402

import awkward as ak  # noqa: E402

# 导入必须在路径设置之后
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bamboohepml.data import DataConfig, DataSourceFactory, HEPDataset  # noqa: E402

# 测试文件路径
TEST_ROOT_FILE = "/Users/physicsboy/Desktop/cepc_hss_scripts/Analysis/sample/tagging/input/TDR/train/Higgs/ss/merge_ss_0006.root"
TREE_NAME = "tree"


def test_basic_reading():
    """测试 1: 基本数据读取"""
    print("=" * 80)
    print("测试 1: 基本数据读取")
    print("=" * 80)

    # 创建数据源
    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
    )

    # 检查可用分支
    branches = source.get_available_branches()
    print(f"可用分支数: {len(branches)}")
    print(f"前 10 个分支: {branches[:10]}")

    # 检查关键分支是否存在
    required_branches = ["jet_energy", "jet_nparticles", "jet_theta", "part_energy", "part_isElectron"]
    missing = [b for b in required_branches if b not in branches]
    if missing:
        print(f"⚠️  缺少分支: {missing}")
        print(f"   可用的相似分支: {[b for b in branches if any(m in b.lower() for m in ['jet', 'part'])]}")
    else:
        print("✓ 所有必需分支都存在")

    # 加载数据
    load_branches = [b for b in required_branches if b in branches]
    if not load_branches:
        # 尝试加载所有包含 jet 或 part 的分支
        load_branches = [b for b in branches if "jet" in b.lower() or "part" in b.lower()][:20]

    print(f"\n加载分支: {load_branches}")
    data = source.load_branches(load_branches)

    print("\n数据形状:")
    print(f"  事件数: {len(data)}")
    for branch in load_branches[:5]:
        if branch in data.fields:
            arr = data[branch]
            if isinstance(arr, ak.Array):
                print(f"  {branch}: {type(arr).__name__}, shape={ak.num(arr, axis=0) if arr.ndim > 1 else len(arr)}")
            else:
                print(f"  {branch}: {type(arr).__name__}, shape={arr.shape if hasattr(arr, 'shape') else 'N/A'}")

    print("\n✓ 基本数据读取测试完成\n")
    return data, load_branches


def test_different_types():
    """测试 2: 不同类型特征"""
    print("=" * 80)
    print("测试 2: 不同类型特征")
    print("=" * 80)

    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
    )

    branches = source.get_available_branches()

    # 查找不同类型的分支
    test_branches = []

    # double 类型（event-level）
    double_branches = [b for b in branches if "jet_energy" in b.lower() or "jet_theta" in b.lower() or "jet_nparticles" in b.lower()]
    if double_branches:
        test_branches.extend(double_branches[:3])

    # vector<float> 类型（object-level）
    vector_float_branches = [b for b in branches if "part_energy" in b.lower() or "part_px" in b.lower() or "part_py" in b.lower()]
    if vector_float_branches:
        test_branches.extend(vector_float_branches[:3])

    # vector<bool> 类型
    vector_bool_branches = [
        b for b in branches if "part_is" in b.lower() and ("electron" in b.lower() or "muon" in b.lower() or "photon" in b.lower())
    ]
    if vector_bool_branches:
        test_branches.extend(vector_bool_branches[:3])

    if not test_branches:
        print("⚠️  未找到测试分支，使用所有可用分支")
        test_branches = branches[:10]

    print(f"测试分支: {test_branches}")
    data = source.load_branches(test_branches)

    print("\n数据类型检查:")
    for branch in test_branches:
        if branch in data.fields:
            arr = data[branch]
            print(f"  {branch}:")
            print(f"    类型: {type(arr).__name__}")
            if isinstance(arr, ak.Array):
                print(f"    维度: {arr.ndim}")
                if arr.ndim == 1:
                    print(f"    长度: {len(arr)}")
                    if len(arr) > 0:
                        print(f"    第一个值: {arr[0]}")
                        print(f"    数据类型: {arr.type}")
                elif arr.ndim == 2:
                    print(f"    事件数: {len(arr)}")
                    if len(arr) > 0:
                        print(f"    第一个事件的长度: {len(arr[0])}")
                        print(f"    第一个事件的值: {arr[0][:3] if len(arr[0]) > 0 else 'empty'}")

    print("\n✓ 不同类型特征测试完成\n")
    return data


def test_padding_fixed_length():
    """测试 3: 固定长度 Padding"""
    print("=" * 80)
    print("测试 3: 固定长度 Padding")
    print("=" * 80)

    from bamboohepml.data.tools import _pad

    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
    )

    branches = source.get_available_branches()

    # 查找 part_energy 或类似的 vector 分支
    vector_branches = [b for b in branches if "part_energy" in b.lower() or "part_px" in b.lower()]
    if not vector_branches:
        print("⚠️  未找到 part_energy，跳过此测试")
        return

    test_branch = vector_branches[0]
    print(f"测试分支: {test_branch}")

    data = source.load_branches([test_branch])
    raw_data = data[test_branch]

    print("\n原始数据:")
    print(f"  事件数: {len(raw_data)}")
    if len(raw_data) > 0:
        lengths = ak.num(raw_data)
        print(f"  长度范围: {ak.min(lengths)} - {ak.max(lengths)}")
        print(f"  平均长度: {ak.mean(lengths):.2f}")
        print(f"  前 5 个事件的长度: {lengths[:5]}")

    # 测试不同长度的 padding
    for pad_length in [32, 64, 128]:
        print(f"\nPadding 到长度 {pad_length}:")
        padded = _pad(raw_data, maxlen=pad_length, value=0.0)

        if isinstance(padded, np.ndarray):
            print(f"  结果形状: {padded.shape}")
            print(f"  数据类型: {padded.dtype}")
            if len(padded) > 0:
                print(f"  第一个事件 (前10个值): {padded[0][:10]}")
                print(f"  第一个事件 (后10个值): {padded[0][-10:]}")
                # 检查 padding
                non_zero = np.count_nonzero(padded[0])
                print(f"  第一个事件非零值数量: {non_zero}/{pad_length}")
                print(f"  Padding 值数量: {pad_length - non_zero}")
        elif isinstance(padded, ak.Array):
            print("  结果类型: ak.Array")
            padded_np = ak.to_numpy(padded)
            print(f"  转换为 numpy 后形状: {padded_np.shape}")
            if len(padded_np) > 0:
                print(f"  第一个事件 (前10个值): {padded_np[0][:10]}")
                print(f"  第一个事件 (后10个值): {padded_np[0][-10:]}")
                non_zero = np.count_nonzero(padded_np[0])
                print(f"  第一个事件非零值数量: {non_zero}/{pad_length}")
        else:
            print(f"  结果类型: {type(padded).__name__}")

    print("\n✓ 固定长度 Padding 测试完成\n")


def test_padding_dynamic_length():
    """测试 4: 动态长度 Padding（到 batch 中最长的）"""
    print("=" * 80)
    print("测试 4: 动态长度 Padding（到 batch 中最长的）")
    print("=" * 80)

    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.1),  # 只加载 10% 的数据用于测试
    )

    branches = source.get_available_branches()
    vector_branches = [b for b in branches if "part_energy" in b.lower() or "part_px" in b.lower()]

    if not vector_branches:
        print("⚠️  未找到 vector 分支，跳过此测试")
        return

    test_branch = vector_branches[0]
    print(f"测试分支: {test_branch}")

    data = source.load_branches([test_branch])
    raw_data = data[test_branch]

    print("\n原始数据:")
    print(f"  事件数: {len(raw_data)}")
    if len(raw_data) > 0:
        lengths = ak.num(raw_data)
        print(f"  长度范围: {ak.min(lengths)} - {ak.max(lengths)}")
        max_length = int(ak.max(lengths))
        print(f"  最大长度: {max_length}")

    # 模拟 batch 处理
    batch_size = 32
    num_batches = min(3, len(raw_data) // batch_size)

    print(f"\n模拟 batch 处理 (batch_size={batch_size}):")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(raw_data))
        batch_data = raw_data[start_idx:end_idx]

        batch_lengths = ak.num(batch_data)
        batch_max_length = int(ak.max(batch_lengths))

        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    事件数: {len(batch_data)}")
        print(f"    长度范围: {ak.min(batch_lengths)} - {ak.max(batch_lengths)}")
        print(f"    Batch 最大长度: {batch_max_length}")

        # Padding 到 batch 最大长度
        from bamboohepml.data.tools import _pad

        padded_batch = _pad(batch_data, maxlen=batch_max_length, value=0.0)

        if isinstance(padded_batch, np.ndarray):
            print(f"    Padding 后形状: {padded_batch.shape}")
            assert padded_batch.shape[1] == batch_max_length, f"Padding 长度不匹配: {padded_batch.shape[1]} != {batch_max_length}"

    print("\n✓ 动态长度 Padding 测试完成\n")


def create_test_config():
    """创建测试配置（类似用户提供的格式）"""
    config = {
        "treename": TREE_NAME,
        "selection": None,
        "new_variables": {
            "part_mask": "ak.ones_like(part_energy)",
            "part_pt": "np.hypot(part_px, part_py)",
            "part_pt_log": "np.log(part_pt + 1e-6)",
            "part_e_log": "np.log(part_energy + 1e-6)",
        },
        "preprocess": {
            "method": "manual",
            "data_fraction": 0.5,
        },
        "inputs": {
            "pf_features": {
                "length": 128,
                "pad_mode": "wrap",
                "vars": [
                    ["part_pt_log", None, 1, -5, 5, 0],
                    ["part_e_log", None, 1, -5, 5, 0],
                    ["part_energy", None, 1, 0, 1000, 0],
                ],
            },
            "pf_mask": {
                "length": 128,
                "pad_mode": "constant",
                "vars": [
                    ["part_mask", None],
                ],
            },
        },
        "labels": {
            "type": "simple",
            "value": ["jet_theta"],  # 简化：只用一个标签
        },
        "observers": [
            "jet_energy",
            "jet_nparticles",
        ],
        "weights": None,
    }
    return config


def test_with_config():
    """测试 5: 使用 DataConfig"""
    print("=" * 80)
    print("测试 5: 使用 DataConfig")
    print("=" * 80)

    # 创建配置
    config_dict = create_test_config()

    # 检查分支是否存在
    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.1),  # 只加载 10% 用于测试
    )

    branches = source.get_available_branches()

    # 调整配置以匹配实际存在的分支
    available_vars = []
    for var_list in config_dict["inputs"]["pf_features"]["vars"]:
        var_name = var_list[0]
        if var_name in branches or any(var_name.startswith(b) for b in branches):
            available_vars.append(var_list)
        else:
            # 尝试找到相似的分支
            similar = [b for b in branches if var_name.split("_")[0] in b.lower()]
            if similar:
                print(f"  使用 {similar[0]} 替代 {var_name}")
                var_list[0] = similar[0]
                available_vars.append(var_list)

    if not available_vars:
        print("⚠️  未找到匹配的变量，使用基础配置")
        config_dict["inputs"]["pf_features"]["vars"] = [
            [b for b in branches if "part" in b.lower() or "jet" in b.lower()][0] if branches else "jet_energy",
            None,
        ]
    else:
        config_dict["inputs"]["pf_features"]["vars"] = available_vars[:3]  # 只使用前3个

    print("配置:")
    print(f"  输入变量: {[v[0] for v in config_dict['inputs']['pf_features']['vars']]}")
    print(f"  Padding 长度: {config_dict['inputs']['pf_features']['length']}")
    print(f"  Padding 模式: {config_dict['inputs']['pf_features']['pad_mode']}")

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # 加载配置
        data_config = DataConfig.load(config_file)

        print("\n加载的配置:")
        print(f"  输入组: {list(data_config.input_dicts.keys())}")
        print(f"  标签: {data_config.label_names}")
        print(f"  观察者: {data_config.observer_names}")

        # 创建数据集
        dataset = HEPDataset(
            data_source=source,
            data_config=data_config,
            for_training=False,
            shuffle=False,
        )

        # 测试迭代
        print("\n测试数据集迭代:")
        count = 0
        for sample in dataset:
            count += 1
            if count >= 3:
                break

            print(f"\n  样本 {count}:")
            for key, value in sample.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    if hasattr(value, "shape"):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: type={type(value).__name__}")
                elif isinstance(value, ak.Array):
                    print(f"    {key}: ak.Array, length={len(value)}")
                else:
                    print(f"    {key}: {type(value).__name__}")

        print(f"\n成功处理了 {count} 个样本")

    finally:
        # 清理临时文件
        if os.path.exists(config_file):
            os.unlink(config_file)

    print("\n✓ DataConfig 测试完成\n")


def test_full_pipeline():
    """测试 6: 完整流程（类似用户配置）"""
    print("=" * 80)
    print("测试 6: 完整流程（类似用户配置）")
    print("=" * 80)

    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.05),  # 只加载 5% 用于测试
    )

    branches = source.get_available_branches()
    print(f"可用分支数: {len(branches)}")

    # 查找实际存在的分支
    def find_branch(pattern):
        matches = [b for b in branches if pattern.lower() in b.lower()]
        return matches[0] if matches else None

    # 构建配置
    part_energy_branch = find_branch("part_energy") or find_branch("part_px")
    part_px_branch = find_branch("part_px")
    part_py_branch = find_branch("part_py")
    part_is_electron_branch = find_branch("part_isElectron") or find_branch("part_is_electron")
    jet_energy_branch = find_branch("jet_energy")
    jet_theta_branch = find_branch("jet_theta")
    jet_nparticles_branch = find_branch("jet_nparticles")

    print("\n找到的分支:")
    print(f"  part_energy: {part_energy_branch}")
    print(f"  part_px: {part_px_branch}")
    print(f"  part_py: {part_py_branch}")
    print(f"  part_isElectron: {part_is_electron_branch}")
    print(f"  jet_energy: {jet_energy_branch}")
    print(f"  jet_theta: {jet_theta_branch}")
    print(f"  jet_nparticles: {jet_nparticles_branch}")

    if not part_energy_branch:
        print("⚠️  未找到必需的分支，跳过完整流程测试")
        return

    # 创建配置
    config_dict = {
        "treename": TREE_NAME,
        "selection": None,
        "new_variables": {},
        "preprocess": {
            "method": "manual",
            "data_fraction": 0.5,
        },
        "inputs": {
            "pf_features": {
                "length": 128,
                "pad_mode": "wrap",
                "vars": [],
            },
            "pf_mask": {
                "length": 128,
                "pad_mode": "constant",
                "vars": [],
            },
        },
        "labels": {
            "type": "simple",
            "value": [],
        },
        "observers": [],
        "weights": None,
    }

    # 添加可用的输入变量
    if part_energy_branch:
        config_dict["inputs"]["pf_features"]["vars"].append([part_energy_branch, None, 1, 0, 1000, 0])

    if part_is_electron_branch:
        config_dict["inputs"]["pf_features"]["vars"].append([part_is_electron_branch, None])

    # 为 pf_mask 添加变量（如果为空会导致错误）
    if part_energy_branch:
        # 添加 part_mask 变量定义
        config_dict["new_variables"]["part_mask"] = "ak.ones_like(part_energy)"
        config_dict["inputs"]["pf_mask"]["vars"].append(["part_mask", None])

    # 添加标签
    if jet_theta_branch:
        config_dict["labels"]["value"].append(jet_theta_branch)
    elif jet_energy_branch:
        config_dict["labels"]["value"].append(jet_energy_branch)

    # 添加观察者
    if jet_energy_branch:
        config_dict["observers"].append(jet_energy_branch)
    if jet_nparticles_branch:
        config_dict["observers"].append(jet_nparticles_branch)

    print("\n配置:")
    print(f"  输入变量数: {len(config_dict['inputs']['pf_features']['vars'])}")
    print(f"  标签数: {len(config_dict['labels']['value'])}")
    print(f"  观察者数: {len(config_dict['observers'])}")

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name

    try:
        # 加载配置并创建数据集
        data_config = DataConfig.load(config_file)
        dataset = HEPDataset(
            data_source=source,
            data_config=data_config,
            for_training=False,
            shuffle=False,
        )

        # 测试 DataLoader
        print("\n测试 DataLoader:")
        loader = DataLoader(dataset, batch_size=8, num_workers=0)

        batch_count = 0
        for batch in loader:
            batch_count += 1
            print(f"\n  Batch {batch_count}:")
            for key, value in batch.items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    if hasattr(value, "shape"):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: type={type(value).__name__}")
                elif isinstance(value, ak.Array):
                    print(f"    {key}: ak.Array, length={len(value)}")
                else:
                    print(f"    {key}: {type(value).__name__}")

            if batch_count >= 2:
                break

        print(f"\n成功处理了 {batch_count} 个批次")

    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)

    print("\n✓ 完整流程测试完成\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("全方位数据读取测试")
    print("=" * 80)
    print(f"\n测试文件: {TEST_ROOT_FILE}")
    print(f"Tree 名称: {TREE_NAME}\n")

    if not os.path.exists(TEST_ROOT_FILE):
        print(f"❌ 错误: 测试文件不存在: {TEST_ROOT_FILE}")
        print("   请检查文件路径")
        return

    try:
        # 测试 1: 基本读取
        data, branches = test_basic_reading()

        # 测试 2: 不同类型
        test_different_types()

        # 测试 3: 固定长度 Padding
        test_padding_fixed_length()

        # 测试 4: 动态长度 Padding
        test_padding_dynamic_length()

        # 测试 5: DataConfig
        test_with_config()

        # 测试 6: 完整流程
        test_full_pipeline()

        print("=" * 80)
        print("✓ 所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
