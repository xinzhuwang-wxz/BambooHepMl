"""
特征依赖图（DAG）模块

提供：
- FeatureGraph: 特征依赖图（支持 cache 和 debug）
- FeatureNode: 图节点
- 拓扑排序和循环检测
- 从 YAML 配置自动构建
"""
import yaml
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..logger import _logger


@dataclass
class FeatureNode:
    """特征图节点。
    
    Attributes:
        name (str): 特征名
        feature_def (dict): 特征定义
        dependencies (List[str]): 依赖的特征列表
        dependents (List[str]): 依赖此特征的特征列表
        computed (bool): 是否已计算
        cached_value (Any): 缓存的计算结果
        computation_time (float): 计算耗时（秒）
    """
    name: str
    feature_def: dict
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    computed: bool = False
    cached_value: Any = None
    computation_time: float = 0.0


class FeatureGraph:
    """
    特征依赖图（有向无环图，DAG）
    
    支持：
    - 构建特征依赖图
    - 拓扑排序
    - 循环依赖检测
    - 中间变量缓存
    - Debug/Inspect 功能
    """
    
    def __init__(self, enable_cache: bool = True):
        """初始化特征图。
        
        Args:
            enable_cache (bool): 是否启用缓存。默认为 True。
        """
        self.nodes: Dict[str, FeatureNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (source, target) 表示 source 依赖 target
        self._execution_order: Optional[List[str]] = None
        self.enable_cache = enable_cache
        self._cache: Dict[str, Any] = {}  # 特征值缓存
        self._computation_stats: Dict[str, dict] = {}  # 计算统计信息
    
    def add_node(self, name: str, feature_def: dict):
        """添加节点。
        
        Args:
            name (str): 特征名
            feature_def (dict): 特征定义
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = FeatureNode(name=name, feature_def=feature_def)
        _logger.debug(f"Added node: {name}")
    
    def add_edge(self, source: str, target: str):
        """添加边：source -> target（source 依赖 target）。
        
        Args:
            source (str): 源节点（依赖 target）
            target (str): 目标节点（被 source 依赖）
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        if (source, target) not in self.edges:
            self.edges.append((source, target))
            self.nodes[source].dependencies.append(target)
            self.nodes[target].dependents.append(source)
            _logger.debug(f"Added edge: {source} -> {target}")
    
    def has_cycles(self) -> bool:
        """检查是否有循环依赖（使用 DFS）。
        
        Returns:
            bool: 是否有循环
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node_name: str) -> bool:
            """深度优先搜索检测循环。"""
            if node_name in rec_stack:
                return True  # 发现循环
            if node_name in visited:
                return False
            
            visited.add(node_name)
            rec_stack.add(node_name)
            
            # 检查所有依赖
            for dep in self.nodes[node_name].dependencies:
                if dfs(dep):
                    return True
            
            rec_stack.remove(node_name)
            return False
        
        # 检查所有节点
        for node_name in self.nodes:
            if node_name not in visited:
                if dfs(node_name):
                    return True
        
        return False
    
    def topological_sort(self) -> List[str]:
        """拓扑排序（Kahn 算法）。
        
        Returns:
            List[str]: 拓扑排序后的节点列表
            
        Raises:
            ValueError: 如果存在循环依赖
        """
        # 计算入度（有多少节点依赖此节点）
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        for source, target in self.edges:
            in_degree[source] += 1
        
        # 找到所有入度为 0 的节点（没有依赖的节点）
        queue: List[str] = [name for name, degree in in_degree.items() if degree == 0]
        result: List[str] = []
        
        while queue:
            node_name = queue.pop(0)
            result.append(node_name)
            
            # 减少依赖此节点的节点的入度
            for dependent in self.nodes[node_name].dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 检查是否所有节点都被处理（如果没有，说明有循环）
        if len(result) != len(self.nodes):
            raise ValueError("Circular dependency detected in feature graph!")
        
        return result
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序（缓存结果）。
        
        Returns:
            List[str]: 执行顺序列表
        """
        if self._execution_order is None:
            if self.has_cycles():
                raise ValueError("Circular dependency detected! Cannot determine execution order.")
            self._execution_order = self.topological_sort()
        return self._execution_order
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """获取特征的所有依赖。
        
        Args:
            feature_name (str): 特征名
            
        Returns:
            List[str]: 依赖列表
        """
        if feature_name not in self.nodes:
            raise ValueError(f"Feature '{feature_name}' not found")
        return self.nodes[feature_name].dependencies.copy()
    
    def get_dependents(self, feature_name: str) -> List[str]:
        """获取依赖此特征的所有特征。
        
        Args:
            feature_name (str): 特征名
            
        Returns:
            List[str]: 被依赖的特征列表
        """
        if feature_name not in self.nodes:
            raise ValueError(f"Feature '{feature_name}' not found")
        return self.nodes[feature_name].dependents.copy()
    
    def cache_value(self, feature_name: str, value: Any):
        """缓存特征值。
        
        Args:
            feature_name (str): 特征名
            value (Any): 特征值
        """
        if self.enable_cache:
            self._cache[feature_name] = value
            if feature_name in self.nodes:
                self.nodes[feature_name].cached_value = value
                self.nodes[feature_name].computed = True
    
    def get_cached_value(self, feature_name: str) -> Optional[Any]:
        """获取缓存的特征值。
        
        Args:
            feature_name (str): 特征名
            
        Returns:
            Any: 缓存的值，如果不存在返回 None
        """
        if self.enable_cache and feature_name in self._cache:
            return self._cache[feature_name]
        return None
    
    def clear_cache(self, feature_name: Optional[str] = None):
        """清除缓存。
        
        Args:
            feature_name (str, optional): 要清除的特征名。如果为 None，清除所有缓存。
        """
        if feature_name is None:
            self._cache.clear()
            for node in self.nodes.values():
                node.cached_value = None
                node.computed = False
        else:
            if feature_name in self._cache:
                del self._cache[feature_name]
            if feature_name in self.nodes:
                self.nodes[feature_name].cached_value = None
                self.nodes[feature_name].computed = False
    
    def record_computation(self, feature_name: str, time_taken: float, success: bool = True):
        """记录计算统计信息。
        
        Args:
            feature_name (str): 特征名
            time_taken (float): 计算耗时（秒）
            success (bool): 是否成功。默认为 True。
        """
        self._computation_stats[feature_name] = {
            'time_taken': time_taken,
            'success': success,
            'cached': feature_name in self._cache,
        }
        if feature_name in self.nodes:
            self.nodes[feature_name].computation_time = time_taken
    
    def get_stats(self) -> Dict[str, dict]:
        """获取计算统计信息。
        
        Returns:
            Dict[str, dict]: 统计信息字典
        """
        return self._computation_stats.copy()
    
    def inspect_node(self, feature_name: str) -> dict:
        """检查节点信息（用于 debug）。
        
        Args:
            feature_name (str): 特征名
            
        Returns:
            dict: 节点信息
        """
        if feature_name not in self.nodes:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        node = self.nodes[feature_name]
        return {
            'name': node.name,
            'feature_def': node.feature_def,
            'dependencies': node.dependencies,
            'dependents': node.dependents,
            'computed': node.computed,
            'has_cache': feature_name in self._cache,
            'computation_time': node.computation_time,
            'stats': self._computation_stats.get(feature_name, {}),
        }
    
    def inspect_graph(self) -> dict:
        """检查整个图的信息（用于 debug）。
        
        Returns:
            dict: 图信息
        """
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'has_cycles': self.has_cycles(),
            'execution_order': self.get_execution_order(),
            'cache_enabled': self.enable_cache,
            'cached_features': list(self._cache.keys()),
            'nodes': {
                name: {
                    'dependencies': node.dependencies,
                    'dependents': node.dependents,
                    'computed': node.computed,
                    'has_cache': name in self._cache,
                }
                for name, node in self.nodes.items()
            },
        }
    
    def visualize(self, max_depth: int = 3) -> str:
        """可视化依赖图（文本格式）。
        
        Args:
            max_depth (int): 最大深度。默认为 3。
            
        Returns:
            str: 可视化的字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Feature Dependency Graph")
        lines.append("=" * 60)
        lines.append(f"Total nodes: {len(self.nodes)}")
        lines.append(f"Total edges: {len(self.edges)}")
        lines.append(f"Has cycles: {self.has_cycles()}")
        lines.append("")
        
        # 按执行顺序显示
        execution_order = self.get_execution_order()
        lines.append("Execution Order:")
        for i, name in enumerate(execution_order, 1):
            node = self.nodes[name]
            cache_indicator = "✓" if name in self._cache else " "
            lines.append(f"  {i}. [{cache_indicator}] {name}")
            if node.dependencies:
                lines.append(f"      depends on: {', '.join(node.dependencies)}")
            if node.dependents:
                lines.append(f"      used by: {', '.join(node.dependents)}")
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @classmethod
    def from_feature_defs(cls, features: Dict[str, dict], expression_engine, enable_cache: bool = True) -> 'FeatureGraph':
        """从特征定义构建图。
        
        Args:
            features (Dict[str, dict]): 特征定义字典
            expression_engine: 表达式引擎（用于提取依赖）
            enable_cache (bool): 是否启用缓存。默认为 True。
            
        Returns:
            FeatureGraph: 构建的图
        """
        graph = cls(enable_cache=enable_cache)
        
        # 1. 创建所有节点
        for feature_name, feature_def in features.items():
            graph.add_node(feature_name, feature_def)
        
        # 2. 解析依赖并添加边
        for feature_name, feature_def in features.items():
            dependencies = set()
            
            # 从 expr 提取依赖
            if 'expr' in feature_def:
                expr = feature_def['expr']
                deps = expression_engine.get_dependencies(expr)
                dependencies.update(deps)
            
            # 从 source 提取依赖（如果是特征名）
            source = feature_def.get('source', [])
            if isinstance(source, str):
                if source in features:
                    dependencies.add(source)
            elif isinstance(source, list):
                for s in source:
                    if s in features:
                        dependencies.add(s)
            
            # 显式依赖
            if 'dependencies' in feature_def:
                dependencies.update(feature_def['dependencies'])
            
            # 添加边
            for dep in dependencies:
                if dep in graph.nodes:
                    graph.add_edge(feature_name, dep)
        
        return graph
    
    @classmethod
    def from_yaml(cls, yaml_path: str, expression_engine, enable_cache: bool = True) -> 'FeatureGraph':
        """从 YAML 配置文件构建图。
        
        Args:
            yaml_path (str): YAML 文件路径
            expression_engine: 表达式引擎（用于提取依赖）
            enable_cache (bool): 是否启用缓存。默认为 True。
            
        Returns:
            FeatureGraph: 构建的图
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        # 提取所有特征（合并 event_level 和 object_level）
        all_features = {}
        
        if 'features' in config:
            # 处理 event_level 特征
            if 'event_level' in config['features']:
                for feature in config['features']['event_level']:
                    if 'name' in feature:
                        all_features[feature['name']] = feature
            
            # 处理 object_level 特征
            if 'object_level' in config['features']:
                for feature in config['features']['object_level']:
                    if 'name' in feature:
                        all_features[feature['name']] = feature
        
        if not all_features:
            raise ValueError("No features found in YAML configuration")
        
        _logger.info(f"Loaded {len(all_features)} features from {yaml_path}")
        
        # 构建图
        return cls.from_feature_defs(all_features, expression_engine, enable_cache=enable_cache)
