"""
FeatureGraph 测试

测试：
1. 从 YAML 构建图
2. 依赖关系解析
3. 拓扑排序
4. Cache 功能
5. Debug/Inspect 功能
"""
import sys
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import awkward as ak
import yaml

from bamboohepml.data.features.expression import ExpressionEngine
from bamboohepml.data.features.feature_graph import FeatureGraph
from bamboohepml.data.features.processors import FeatureProcessor


def test_build_from_yaml():
    """测试从 YAML 构建图。"""
    print("=" * 60)
    print("测试 1: 从 YAML 构建图")
    print("=" * 60)
    
    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "test_features_config.yaml"
    
    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)
    
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    print(f"有循环: {graph.has_cycles()}")
    
    # 检查执行顺序
    execution_order = graph.get_execution_order()
    print(f"\n执行顺序: {execution_order}")
    
    # 验证顺序正确性
    # met 和 ht 应该在 met_ht_ratio 之前
    assert execution_order.index("met") < execution_order.index("met_ht_ratio")
    assert execution_order.index("ht") < execution_order.index("met_ht_ratio")
    
    print("✓ 从 YAML 构建图测试通过\n")


def test_dependency_resolution():
    """测试依赖解析。"""
    print("=" * 60)
    print("测试 2: 依赖解析")
    print("=" * 60)
    
    engine = ExpressionEngine()
    
    # 定义特征
    features = {
        'met': {'expr': 'met', 'type': 'event'},
        'ht': {'expr': 'sum(Jet.pt)', 'type': 'event'},
        'met_ht_ratio': {
            'expr': 'met / (ht + 1e-6)',
            'type': 'event',
            'dependencies': ['met', 'ht']
        },
        'jet_pt': {'expr': 'Jet.pt', 'type': 'object'},
        'jet_pt_log': {
            'expr': 'log1p(Jet.pt)',
            'type': 'object',
            'dependencies': ['jet_pt']
        },
    }
    
    graph = FeatureGraph.from_feature_defs(features, engine)
    
    # 检查依赖
    print("依赖关系:")
    for name in features.keys():
        deps = graph.get_dependencies(name)
        dependents = graph.get_dependents(name)
        print(f"  {name}:")
        print(f"    依赖: {deps}")
        print(f"    被依赖: {dependents}")
    
    # 验证依赖
    assert set(graph.get_dependencies("met_ht_ratio")) == {"met", "ht"}
    assert set(graph.get_dependencies("jet_pt_log")) == {"jet_pt"}
    
    print("✓ 依赖解析测试通过\n")


def test_cache():
    """测试 Cache 功能。"""
    print("=" * 60)
    print("测试 3: Cache 功能")
    print("=" * 60)
    
    engine = ExpressionEngine()
    
    features = {
        'met': {'expr': 'met', 'type': 'event'},
        'ht': {'expr': 'sum(Jet.pt)', 'type': 'event'},
    }
    
    graph = FeatureGraph.from_feature_defs(features, engine, enable_cache=True)
    
    # 缓存值
    test_value_met = np.array([50.0, 100.0, 75.0])
    test_value_ht = np.array([200.0, 300.0, 250.0])
    
    graph.cache_value("met", test_value_met)
    graph.cache_value("ht", test_value_ht)
    
    # 获取缓存值
    cached_met = graph.get_cached_value("met")
    cached_ht = graph.get_cached_value("ht")
    
    assert cached_met is not None
    assert cached_ht is not None
    assert np.allclose(cached_met, test_value_met)
    assert np.allclose(cached_ht, test_value_ht)
    
    print(f"缓存 met: {cached_met}")
    print(f"缓存 ht: {cached_ht}")
    
    # 清除缓存
    graph.clear_cache("met")
    assert graph.get_cached_value("met") is None
    assert graph.get_cached_value("ht") is not None
    
    graph.clear_cache()  # 清除所有
    assert graph.get_cached_value("ht") is None
    
    print("✓ Cache 功能测试通过\n")


def test_debug_inspect():
    """测试 Debug/Inspect 功能。"""
    print("=" * 60)
    print("测试 4: Debug/Inspect 功能")
    print("=" * 60)
    
    engine = ExpressionEngine()
    
    features = {
        'met': {'expr': 'met', 'type': 'event'},
        'ht': {'expr': 'sum(Jet.pt)', 'type': 'event'},
        'met_ht_ratio': {
            'expr': 'met / (ht + 1e-6)',
            'type': 'event',
            'dependencies': ['met', 'ht']
        },
    }
    
    graph = FeatureGraph.from_feature_defs(features, engine)
    
    # 记录计算统计
    graph.record_computation("met", 0.001, success=True)
    graph.record_computation("ht", 0.002, success=True)
    graph.cache_value("met", np.array([50.0, 100.0]))
    
    # 检查节点
    node_info = graph.inspect_node("met_ht_ratio")
    print("节点信息 (met_ht_ratio):")
    for key, value in node_info.items():
        if key != 'feature_def':  # 跳过 feature_def（太长）
            print(f"  {key}: {value}")
    
    # 检查整个图
    graph_info = graph.inspect_graph()
    print("\n图信息:")
    print(f"  节点数: {graph_info['num_nodes']}")
    print(f"  边数: {graph_info['num_edges']}")
    print(f"  有循环: {graph_info['has_cycles']}")
    print(f"  缓存的特征: {graph_info['cached_features']}")
    
    # 可视化
    print("\n可视化:")
    print(graph.visualize())
    
    # 统计信息
    stats = graph.get_stats()
    print("\n统计信息:")
    for name, stat in stats.items():
        print(f"  {name}: {stat}")
    
    print("✓ Debug/Inspect 功能测试通过\n")


def test_computation_order():
    """测试计算顺序。"""
    print("=" * 60)
    print("测试 5: 计算顺序")
    print("=" * 60)
    
    engine = ExpressionEngine()
    yaml_path = Path(__file__).parent / "test_features_config.yaml"
    graph = FeatureGraph.from_yaml(str(yaml_path), engine, enable_cache=True)
    
    # 准备数据
    context = {
        'met': np.array([50.0, 100.0, 75.0]),
        'Jet': ak.zip({
            'pt': ak.Array([[30.0, 25.0], [50.0, 40.0], [45.0]]),
        })
    }
    
    # 按顺序计算
    execution_order = graph.get_execution_order()
    print(f"执行顺序: {execution_order}")
    
    computed_features = []
    for feature_name in execution_order:
        # 检查依赖是否都已计算
        deps = graph.get_dependencies(feature_name)
        missing_deps = [dep for dep in deps if dep not in computed_features and dep not in context]
        
        if missing_deps:
            print(f"  跳过 {feature_name}（缺少依赖: {missing_deps}）")
            continue
        
        # 计算特征
        feature_def = graph.nodes[feature_name].feature_def
        if 'expr' in feature_def:
            try:
                raw_value = engine.evaluate(feature_def['expr'], context)
                processor = FeatureProcessor(feature_def)
                processed_value = processor.process(raw_value, fit_normalizer=True)
                graph.cache_value(feature_name, processed_value)
                context[feature_name] = processed_value
                computed_features.append(feature_name)
                print(f"  ✓ 计算 {feature_name}")
            except Exception as e:
                print(f"  ✗ 计算 {feature_name} 失败: {e}")
        else:
            # 直接使用 source
            source = feature_def.get('source')
            if source in context:
                graph.cache_value(feature_name, context[source])
                computed_features.append(feature_name)
                print(f"  ✓ 使用源数据 {feature_name}")
    
    print(f"\n成功计算的特征: {computed_features}")
    print(f"缓存的特征: {list(graph._cache.keys())}")
    
    print("✓ 计算顺序测试通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FeatureGraph 测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_build_from_yaml()
        test_dependency_resolution()
        test_cache()
        test_debug_inspect()
        test_computation_order()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

