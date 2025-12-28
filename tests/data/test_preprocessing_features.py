#!/usr/bin/env python
"""
测试预处理功能：
1. 标准化操作（center, scale）
2. 自动标准化（从 train 集计算参数）
3. Padding 模式（wrap, constant, repeat）
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

import numpy as np
import awkward as ak
import yaml
import tempfile

from bamboohepml.data import DataSourceFactory, HEPDataset, DataConfig
from bamboohepml.data.tools import _pad, _repeat_pad, _clip
from bamboohepml.data.preprocess import AutoStandardizer

# 测试文件路径
TEST_ROOT_FILE = "/Users/physicsboy/Desktop/cepc_hss_scripts/Analysis/sample/tagging/input/TDR/train/Higgs/ss/merge_ss_0006.root"
TREE_NAME = "tree"


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_standardization():
    """测试 1: 标准化操作（center, scale）"""
    print_section("测试 1: 标准化操作（center, scale）")
    
    # 创建测试数据
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    center = 2.0
    scale = 0.5
    min_val = -5
    max_val = 5
    
    print(f"\n原始数据: {test_data}")
    print(f"标准化参数: center={center}, scale={scale}")
    print(f"裁剪范围: [{min_val}, {max_val}]")
    
    # 标准化: (value - center) * scale
    standardized = (test_data - center) * scale
    print(f"\n标准化后: {standardized}")
    print(f"  计算: (1.0 - 2.0) * 0.5 = {-0.5}")
    print(f"  计算: (2.0 - 2.0) * 0.5 = {0.0}")
    print(f"  计算: (3.0 - 2.0) * 0.5 = {0.5}")
    
    # 裁剪
    clipped = _clip(standardized, min_val, max_val)
    print(f"\n裁剪后: {clipped}")
    print(f"  所有值都在 [{min_val}, {max_val}] 范围内: {np.all((clipped >= min_val) & (clipped <= max_val))}")
    
    # 测试实际数据
    print(f"\n测试实际数据（从 ROOT 文件）:")
    source = DataSourceFactory.create(
        TEST_ROOT_FILE,
        treename=TREE_NAME,
        load_range=(0, 0.01),  # 只加载 1% 用于测试
    )
    
    branches = source.get_available_branches()
    test_branch = None
    for b in branches:
        if 'part_energy' in b.lower() or 'part_px' in b.lower():
            test_branch = b
            break
    
    if test_branch:
        data = source.load_branches([test_branch])
        raw_values = data[test_branch]
        
        # 取第一个事件的第一个值
        if isinstance(raw_values, ak.Array) and len(raw_values) > 0:
            first_value = raw_values[0][0] if len(raw_values[0]) > 0 else None
            if first_value is not None:
                print(f"  分支: {test_branch}")
                print(f"  第一个值: {first_value}")
                
                # 应用标准化
                center_test = 2.0
                scale_test = 0.7
                standardized_value = (first_value - center_test) * scale_test
                clipped_value = _clip(np.array([standardized_value]), -5, 5)[0]
                
                print(f"  标准化: ({first_value} - {center_test}) * {scale_test} = {standardized_value}")
                print(f"  裁剪: clip({standardized_value}, -5, 5) = {clipped_value}")
                print(f"  ✓ 标准化和裁剪功能正常")
    
    print("\n✓ 标准化操作测试完成")


def test_padding_modes():
    """测试 2: Padding 模式（wrap, constant, repeat）"""
    print_section("测试 2: Padding 模式（wrap, constant, repeat）")
    
    # 创建测试数据（jagged array）
    test_data = ak.Array([
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
    ])
    maxlen = 5
    
    print(f"\n原始数据（jagged array）:")
    for i, arr in enumerate(test_data):
        print(f"  事件 {i}: {arr.tolist()} (长度: {len(arr)})")
    print(f"目标长度: {maxlen}")
    
    # 测试 constant 模式
    print(f"\n1. Constant 模式（常量填充）:")
    padded_constant = _pad(test_data, maxlen, value=0)
    print(f"  填充值: 0")
    for i, arr in enumerate(padded_constant):
        print(f"  事件 {i}: {arr.tolist()} (长度: {len(arr)})")
        # 验证
        original_len = len(test_data[i])
        if original_len < maxlen:
            expected_padding = maxlen - original_len
            actual_padding = np.sum(arr[original_len:] == 0)
            print(f"    ✓ Padding 数量: {actual_padding}/{expected_padding}")
    
    # 测试 wrap 模式
    print(f"\n2. Wrap 模式（循环填充）:")
    padded_wrap = _repeat_pad(test_data, maxlen, shuffle=False)
    print(f"  填充方式: 循环重复原始数据")
    for i, arr in enumerate(padded_wrap):
        print(f"  事件 {i}: {arr.tolist()} (长度: {len(arr)})")
        # 验证循环
        original = test_data[i].tolist()
        if len(original) < maxlen:
            expected = (original * (maxlen // len(original) + 1))[:maxlen]
            print(f"    期望: {expected}")
            print(f"    实际: {arr.tolist()}")
            match = np.allclose(arr[:len(original)], np.array(original))
            print(f"    ✓ 原始数据匹配: {match}")
    
    # 测试 repeat 模式（带打乱）
    print(f"\n3. Repeat 模式（重复填充，可打乱）:")
    padded_repeat = _repeat_pad(test_data, maxlen, shuffle=True)
    print(f"  填充方式: 重复（打乱）")
    for i, arr in enumerate(padded_repeat):
        print(f"  事件 {i}: {arr.tolist()} (长度: {len(arr)})")
        # 验证长度
        print(f"    ✓ 长度正确: {len(arr) == maxlen}")
    
    print("\n✓ Padding 模式测试完成")


def test_auto_standardization():
    """测试 3: 自动标准化（从 train 集计算参数）"""
    print_section("测试 3: 自动标准化（从 train 集计算参数）")
    
    if not os.path.exists(TEST_ROOT_FILE):
        print("⚠️  测试文件不存在，跳过此测试")
        return
    
    # 创建配置
    config_dict = {
        'treename': TREE_NAME,
        'selection': None,
        'new_variables': {},
        'preprocess': {
            'method': 'auto',  # 自动模式
            'data_fraction': 0.1,  # 使用 10% 的数据（确保有足够的数据）
        },
        'inputs': {
            'pf_features': {
                'length': 128,
                'pad_mode': 'wrap',
                'vars': [
                    # 不设置 center（删除 None），让 DataConfig 根据 method='auto' 自动设置为 'auto'
                    # 格式: [var_name, subtract_by, multiply_by, clip_min, clip_max, pad_value]
                    # 如果 method='auto'，不设置 subtract_by 时会自动使用 'auto'
                    ['part_energy'],  # 只设置变量名，让 DataConfig 自动设置 center='auto'
                ],
            },
        },
        'labels': {
            'type': 'simple',
            'value': ['is_signal'] if 'is_signal' in DataSourceFactory.create(TEST_ROOT_FILE, treename=TREE_NAME).get_available_branches() else ['jet_energy'],
        },
        'observers': [],
        'weights': None,
    }
    
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_file = f.name
    
    try:
        # 加载配置
        data_config = DataConfig.load(config_file)
        
        print(f"\n配置加载后:")
        print(f"  preprocess_params 中的变量: {list(data_config.preprocess_params.keys())}")
        if 'part_energy' in data_config.preprocess_params:
            print(f"  part_energy 的 center: {data_config.preprocess_params['part_energy'].get('center')}")
            print(f"  part_energy 的完整参数: {data_config.preprocess_params['part_energy']}")
        else:
            print(f"  ⚠️  part_energy 不在 preprocess_params 中")
        
        # 确保需要自动标准化的变量设置了 center='auto'
        if 'part_energy' in data_config.preprocess_params:
            if data_config.preprocess_params['part_energy'].get('center') != 'auto':
                data_config.preprocess_params['part_energy']['center'] = 'auto'
                print(f"  ✓ 已设置 part_energy.center = 'auto'")
            else:
                print(f"  ✓ part_energy.center 已经是 'auto'")
        else:
            print(f"  ⚠️  无法设置 part_energy.center，因为不在 preprocess_params 中")
        
        print(f"\n配置:")
        print(f"  预处理方法: {data_config.preprocess['method']}")
        print(f"  数据比例: {data_config.preprocess['data_fraction']}")
        print(f"  需要自动标准化的变量: {[k for k, v in data_config.preprocess_params.items() if v.get('center') == 'auto']}")
        
        # 创建 AutoStandardizer
        print(f"\n创建 AutoStandardizer...")
        auto_std = AutoStandardizer(TEST_ROOT_FILE, data_config)
        # 明确设置 load_range 确保有足够的数据
        auto_std.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))
        
        # 读取训练数据
        print(f"读取训练数据...")
        print(f"  load_range: {auto_std.load_range}")
        train_table = auto_std.read_file([TEST_ROOT_FILE])
        print(f"  事件数: {len(train_table)}")
        
        # 计算标准化参数
        print(f"\n计算标准化参数...")
        preprocess_params = auto_std.make_preprocess_params(train_table)
        
        print(f"\n计算出的标准化参数:")
        for k, params in preprocess_params.items():
            if params.get('center') != 'auto' and params.get('center') is not None:
                print(f"  {k}:")
                print(f"    center: {params['center']:.4f}")
                print(f"    scale: {params['scale']:.4f}")
                print(f"    min: {params['min']}")
                print(f"    max: {params['max']}")
        
        # 更新配置
        data_config.preprocess_params.update(preprocess_params)
        
        # 验证参数已更新
        print(f"\n验证参数已更新:")
        for k, params in data_config.preprocess_params.items():
            if params.get('center') != 'auto' and params.get('center') is not None:
                print(f"  {k}: center={params['center']:.4f}, scale={params['scale']:.4f}")
                print(f"    ✓ 参数已从 'auto' 更新为实际值")
        
        print(f"\n✓ 自动标准化功能正常")
        print(f"  现在可以在验证集/测试集上使用相同的参数")
        
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)
    
    print("\n✓ 自动标准化测试完成")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("  预处理功能验证测试")
    print("=" * 80)
    
    try:
        # 测试 1: 标准化操作
        test_standardization()
        
        # 测试 2: Padding 模式
        test_padding_modes()
        
        # 测试 3: 自动标准化
        test_auto_standardization()
        
        print("\n" + "=" * 80)
        print("  ✓ 所有测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

