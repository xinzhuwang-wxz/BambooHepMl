"""
表达式引擎测试

测试表达式引擎的功能：
1. 表达式解析和求值
2. 向量化计算
3. 函数注册
4. 依赖提取
"""
import sys
import os
import numpy as np
import awkward as ak
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bamboohepml.data.features.expression import ExpressionEngine, OperatorRegistry
from bamboohepml.data.features.feature_graph import FeatureGraph
from bamboohepml.data.features.processors import FeatureProcessor


def create_test_data():
    """创建测试数据（模拟 ROOT 文件数据）。"""
    # 模拟 3 个事件
    n_events = 3
    
    # Event-level 数据
    met = np.array([50.0, 100.0, 75.0], dtype=np.float32)
    
    # Object-level 数据（每个事件有不同数量的 Jet）
    jet_pt = ak.Array([
        [30.0, 25.0, 20.0],      # 事件 0：3 个 jet
        [50.0, 40.0, 35.0, 30.0], # 事件 1：4 个 jet
        [45.0, 35.0],            # 事件 2：2 个 jet
    ])
    
    jet_eta = ak.Array([
        [1.2, -0.5, 2.1],
        [0.8, -1.2, 1.5, -0.3],
        [1.0, -0.8],
    ])
    
    jet_phi = ak.Array([
        [0.5, 1.2, -0.8],
        [0.3, -1.1, 2.0, 0.9],
        [0.7, -0.5],
    ])
    
    jet_mass = ak.Array([
        [10.0, 8.0, 5.0],
        [15.0, 12.0, 10.0, 8.0],
        [12.0, 9.0],
    ])
    
    # 构建 Jet 对象（使用 RecordArray）
    Jet = ak.zip({
        'pt': jet_pt,
        'eta': jet_eta,
        'phi': jet_phi,
        'mass': jet_mass,
    })
    
    # 构建上下文
    context = {
        'met': met,
        'Jet': Jet,
    }
    
    return context


def test_basic_expressions():
    """测试基础表达式。"""
    print("=" * 60)
    print("测试 1: 基础表达式")
    print("=" * 60)
    
    engine = ExpressionEngine()
    context = create_test_data()
    
    # 测试 1: 直接访问变量
    result = engine.evaluate("met", context)
    print(f"met = {result}")
    assert len(result) == 3
    assert np.allclose(result, [50.0, 100.0, 75.0])
    
    # 测试 2: 属性访问
    result = engine.evaluate("Jet.pt", context)
    print(f"Jet.pt = {result}")
    print(f"  Event 0: {result[0]}")
    print(f"  Event 1: {result[1]}")
    print(f"  Event 2: {result[2]}")
    
    # 测试 3: 数学运算
    result = engine.evaluate("met * 2", context)
    print(f"met * 2 = {result}")
    assert np.allclose(result, [100.0, 200.0, 150.0])
    
    print("✓ 基础表达式测试通过\n")


def test_aggregation_functions():
    """测试聚合函数。"""
    print("=" * 60)
    print("测试 2: 聚合函数")
    print("=" * 60)
    
    engine = ExpressionEngine()
    context = create_test_data()
    
    # 测试 sum
    result = engine.evaluate("sum(Jet.pt)", context)
    print(f"sum(Jet.pt) = {result}")
    expected = np.array([75.0, 155.0, 80.0])  # 30+25+20, 50+40+35+30, 45+35
    assert np.allclose(result, expected)
    
    # 测试 mean
    result = engine.evaluate("mean(Jet.pt)", context)
    print(f"mean(Jet.pt) = {result}")
    expected = np.array([25.0, 38.75, 40.0])  # 75/3, 155/4, 80/2
    assert np.allclose(result, expected)
    
    # 测试 max
    result = engine.evaluate("max(Jet.pt)", context)
    print(f"max(Jet.pt) = {result}")
    expected = np.array([30.0, 50.0, 45.0])
    assert np.allclose(result, expected)
    
    # 测试 len
    result = engine.evaluate("len(Jet.pt)", context)
    print(f"len(Jet.pt) = {result}")
    expected = np.array([3, 4, 2])
    assert np.allclose(result, expected)
    
    print("✓ 聚合函数测试通过\n")


def test_math_functions():
    """测试数学函数。"""
    print("=" * 60)
    print("测试 3: 数学函数")
    print("=" * 60)
    
    engine = ExpressionEngine()
    context = create_test_data()
    
    # 测试 log1p
    result = engine.evaluate("log1p(met)", context)
    print(f"log1p(met) = {result}")
    expected = np.log1p([50.0, 100.0, 75.0])
    assert np.allclose(result, expected)
    
    # 测试 sqrt
    result = engine.evaluate("sqrt(met)", context)
    print(f"sqrt(met) = {result}")
    expected = np.sqrt([50.0, 100.0, 75.0])
    assert np.allclose(result, expected)
    
    # 测试复杂表达式
    result = engine.evaluate("sqrt(Jet.pt**2 + Jet.mass**2)", context)
    print(f"sqrt(Jet.pt**2 + Jet.mass**2) = {result}")
    print(f"  Event 0: {result[0]}")
    
    print("✓ 数学函数测试通过\n")


def test_custom_function():
    """测试自定义函数注册。"""
    print("=" * 60)
    print("测试 4: 自定义函数注册")
    print("=" * 60)
    
    engine = ExpressionEngine()
    context = create_test_data()
    
    # 注册自定义函数
    def custom_multiply(x, factor):
        """自定义乘法函数。"""
        if isinstance(x, ak.Array):
            return x * factor
        else:
            return np.asarray(x) * factor
    
    engine.register_function("multiply", custom_multiply)
    
    # 使用自定义函数
    result = engine.evaluate("multiply(met, 3)", context)
    print(f"multiply(met, 3) = {result}")
    expected = np.array([150.0, 300.0, 225.0])
    assert np.allclose(result, expected)
    
    print("✓ 自定义函数测试通过\n")


def test_dependency_extraction():
    """测试依赖提取。"""
    print("=" * 60)
    print("测试 5: 依赖提取")
    print("=" * 60)
    
    engine = ExpressionEngine()
    
    # 测试简单表达式
    deps = engine.get_dependencies("met")
    print(f"依赖 'met': {deps}")
    assert deps == {'met'}
    
    # 测试复杂表达式
    deps = engine.get_dependencies("met / (ht + 1e-6)")
    print(f"依赖 'met / (ht + 1e-6)': {deps}")
    assert deps == {'met', 'ht'}
    
    # 测试属性访问
    deps = engine.get_dependencies("sum(Jet.pt)")
    print(f"依赖 'sum(Jet.pt)': {deps}")
    assert deps == {'Jet'}
    
    print("✓ 依赖提取测试通过\n")


def test_feature_graph():
    """测试特征依赖图。"""
    print("=" * 60)
    print("测试 6: 特征依赖图")
    print("=" * 60)
    
    engine = ExpressionEngine()
    
    # 定义特征
    features = {
        'met': {
            'expr': 'met',
            'type': 'event',
        },
        'ht': {
            'expr': 'sum(Jet.pt)',
            'type': 'event',
            'dependencies': ['Jet'],
        },
        'met_ht_ratio': {
            'expr': 'met / (ht + 1e-6)',
            'type': 'event',
            'dependencies': ['met', 'ht'],
        },
        'jet_pt': {
            'expr': 'Jet.pt',
            'type': 'object',
        },
        'jet_pt_log': {
            'expr': 'log1p(Jet.pt)',
            'type': 'object',
            'dependencies': ['jet_pt'],
        },
    }
    
    # 构建图
    graph = FeatureGraph.from_feature_defs(features, engine)
    
    # 检查循环
    has_cycles = graph.has_cycles()
    print(f"是否有循环依赖: {has_cycles}")
    assert not has_cycles
    
    # 获取执行顺序
    execution_order = graph.get_execution_order()
    print(f"执行顺序: {execution_order}")
    
    # 验证顺序：met 和 jet_pt 应该在最前面（无依赖）
    assert 'met' in execution_order[:2] or 'jet_pt' in execution_order[:2]
    # met_ht_ratio 应该在 met 和 ht 之后
    assert execution_order.index('met_ht_ratio') > execution_order.index('met')
    assert execution_order.index('met_ht_ratio') > execution_order.index('ht')
    
    print("✓ 特征依赖图测试通过\n")


def test_feature_processor():
    """测试特征处理器。"""
    print("=" * 60)
    print("测试 7: 特征处理器")
    print("=" * 60)
    
    context = create_test_data()
    
    # 测试 event-level 特征处理
    feature_def = {
        'name': 'met_log',
        'type': 'event',
        'dtype': 'float32',
        'expr': 'log1p(met)',
        'normalize': {
            'method': 'auto',
        },
        'clip': {
            'min': 0.0,
            'max': 10.0,
        },
    }
    
    processor = FeatureProcessor(feature_def)
    
    # 计算原始值
    engine = ExpressionEngine()
    raw_value = engine.evaluate(feature_def['expr'], context)
    print(f"原始值: {raw_value}")
    
    # 处理
    processed_value = processor.process(raw_value, fit_normalizer=True)
    print(f"处理后: {processed_value}")
    print(f"  标准化中心: {processor.normalizer.center}")
    print(f"  标准化缩放: {processor.normalizer.scale}")
    
    # 测试 object-level 特征处理
    feature_def_obj = {
        'name': 'jet_pt',
        'type': 'object',
        'dtype': 'float32',
        'expr': 'Jet.pt',
        'normalize': {
            'method': 'auto',
        },
        'clip': {
            'min': 0.0,
            'max': 1000.0,
        },
        'padding': {
            'max_length': 5,
            'mode': 'constant',
            'value': 0.0,
        },
    }
    
    processor_obj = FeatureProcessor(feature_def_obj)
    raw_value_obj = engine.evaluate(feature_def_obj['expr'], context)
    print(f"\n原始值 (object-level):")
    print(f"  Event 0: {raw_value_obj[0]}")
    
    processed_value_obj = processor_obj.process(raw_value_obj, fit_normalizer=True)
    print(f"处理后 (object-level, padded to 5):")
    print(f"  Event 0: {processed_value_obj[0]}")
    assert processed_value_obj.shape == (3, 5)  # 3 个事件，每个填充到 5
    
    print("✓ 特征处理器测试通过\n")


def test_with_yaml_config():
    """测试使用 YAML 配置。"""
    print("=" * 60)
    print("测试 8: YAML 配置")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent / "test_features_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 合并所有特征
    all_features = {}
    for feature in config['features'].get('event_level', []):
        all_features[feature['name']] = feature
    for feature in config['features'].get('object_level', []):
        all_features[feature['name']] = feature
    
    print(f"加载了 {len(all_features)} 个特征")
    
    # 构建图
    engine = ExpressionEngine()
    graph = FeatureGraph.from_feature_defs(all_features, engine)
    
    # 获取执行顺序
    execution_order = graph.get_execution_order()
    print(f"执行顺序 (前 5 个): {execution_order[:5]}")
    
    # 测试处理几个特征
    context = create_test_data()
    
    # 处理 ht
    ht_def = all_features['ht']
    ht_processor = FeatureProcessor(ht_def)
    ht_raw = engine.evaluate(ht_def['expr'], context)
    ht_processed = ht_processor.process(ht_raw, fit_normalizer=True)
    print(f"\nht:")
    print(f"  原始值: {ht_raw}")
    print(f"  处理后: {ht_processed}")
    
    print("✓ YAML 配置测试通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("表达式引擎测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_basic_expressions()
        test_aggregation_functions()
        test_math_functions()
        test_custom_function()
        test_dependency_extraction()
        test_feature_graph()
        test_feature_processor()
        test_with_yaml_config()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

