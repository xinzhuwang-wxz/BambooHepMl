"""
特征依赖图（DAG）模块

提供：
- FeatureGraph: 特征依赖图（支持 cache 和 debug）
- FeatureNode: 图节点
- 拓扑排序和循环检测
- 从 YAML 配置自动构建
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import torch
import yaml

from ..logger import _logger
from .processors import FeatureProcessor


@dataclass
class FeatureNode:
    """特征图节点。

    Attributes:
        name (str): 特征名
        feature_def (dict): 特征定义
        dependencies (List[str]): 依赖的特征列表
        dependents (List[str]): 依赖此特征的特征列表
        computation_time (float): 计算耗时（秒）
    """

    name: str
    feature_def: dict
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    computation_time: float = 0.0


class FeatureGraph:
    """
    特征依赖图（有向无环图，DAG）

    支持：
    - 构建特征依赖图
    - 拓扑排序
    - 循环依赖检测
    - 预编译执行计划
    - 状态持久化（Normalizer参数）
    """

    def __init__(self, expression_engine=None, enable_cache: bool = True):
        """初始化特征图。

        Args:
            expression_engine: 表达式引擎实例（用于 build_batch）
            enable_cache (bool): 是否启用缓存。默认为 True。
        """
        self.nodes: dict[str, FeatureNode] = {}
        self.edges: list[tuple[str, str]] = []  # (source, target) 表示 source 依赖 target
        self._execution_order: list[str] | None = None
        self._compiled_plan: list[tuple[str, FeatureProcessor, Any]] | None = None  # 编译后的执行计划
        self.enable_cache = enable_cache
        self._cache: dict[str, Any] = {}  # 特征值缓存 (仅用于 debug 或单次计算)
        self._computation_stats: dict[str, dict] = {}  # 计算统计信息
        self.expression_engine = expression_engine  # 存储表达式引擎
        self.processors: dict[str, FeatureProcessor] = {}  # 特征处理器（包含状态）

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
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_name: str) -> bool:
            """深度优先搜索检测循环。"""
            if node_name in rec_stack:
                return True  # 发现循环
            if node_name in visited:
                return False

            visited.add(node_name)
            rec_stack.add(node_name)

            # 检查所有依赖（只检查在图中存在的节点）
            for dep in self.nodes[node_name].dependencies:
                if dep not in self.nodes:
                    # 依赖不在图中，跳过（可能是原始数据字段）
                    continue
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

    def topological_sort(self) -> list[str]:
        """拓扑排序（Kahn 算法）。

        Returns:
            List[str]: 拓扑排序后的节点列表

        Raises:
            ValueError: 如果存在循环依赖
        """
        # 计算入度（有多少个依赖必须在此节点之前处理）
        # 注意：edge (source, target) 表示 source 依赖 target
        # 对于拓扑排序，我们需要计算：每个节点有多少个依赖（dependencies）
        # 即：in_degree[node] = len(node.dependencies)
        in_degree: dict[str, int] = {}
        for name in self.nodes:
            # 入度 = 该节点有多少个依赖（dependencies）
            # 只计算在图中存在的依赖
            in_degree[name] = len([dep for dep in self.nodes[name].dependencies if dep in self.nodes])

        # 找到所有入度为 0 的节点（没有依赖的节点，可以立即处理）
        queue: list[str] = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)

            # 减少依赖此节点的节点的入度
            # 当 node_name 被处理后，所有依赖 node_name 的节点可以减少一个依赖
            for dependent in self.nodes[node_name].dependents:
                if dependent in in_degree:  # 确保 dependent 在图中
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # 检查是否所有节点都被处理（如果没有，说明有循环）
        if len(result) != len(self.nodes):
            # 找出未处理的节点（这些节点形成了循环）
            unprocessed = set(self.nodes.keys()) - set(result)
            raise ValueError(f"Circular dependency detected in feature graph! " f"Unprocessed nodes: {sorted(unprocessed)}")

        return result

    def get_execution_order(self) -> list[str]:
        """获取执行顺序（缓存结果）。

        Returns:
            List[str]: 执行顺序列表
        """
        if self._execution_order is None:
            if self.has_cycles():
                raise ValueError("Circular dependency detected! Cannot determine execution order.")
            self._execution_order = self.topological_sort()
        return self._execution_order

    def get_dependencies(self, feature_name: str) -> list[str]:
        """获取特征的所有依赖。

        Args:
            feature_name (str): 特征名

        Returns:
            List[str]: 依赖列表
        """
        if feature_name not in self.nodes:
            raise ValueError(f"Feature '{feature_name}' not found")
        return self.nodes[feature_name].dependencies.copy()

    def get_dependents(self, feature_name: str) -> list[str]:
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

    def get_cached_value(self, feature_name: str) -> Any | None:
        """获取缓存的特征值。

        Args:
            feature_name (str): 特征名

        Returns:
            Any: 缓存的值，如果不存在返回 None
        """
        if self.enable_cache and feature_name in self._cache:
            return self._cache[feature_name]
        return None

    def clear_cache(self, feature_name: str | None = None):
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
            "time_taken": time_taken,
            "success": success,
            "cached": feature_name in self._cache,
        }
        if feature_name in self.nodes:
            self.nodes[feature_name].computation_time = time_taken

    def get_stats(self) -> dict[str, dict]:
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
            "name": node.name,
            "feature_def": node.feature_def,
            "dependencies": node.dependencies,
            "dependents": node.dependents,
            "computed": node.computed,
            "has_cache": feature_name in self._cache,
            "computation_time": node.computation_time,
            "stats": self._computation_stats.get(feature_name, {}),
        }

    def inspect_graph(self) -> dict:
        """检查整个图的信息（用于 debug）。

        Returns:
            dict: 图信息
        """
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "has_cycles": self.has_cycles(),
            "execution_order": self.get_execution_order(),
            "cache_enabled": self.enable_cache,
            "cached_features": list(self._cache.keys()),
            "nodes": {
                name: {
                    "dependencies": node.dependencies,
                    "dependents": node.dependents,
                    "computed": node.computed,
                    "has_cache": name in self._cache,
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

    def output_spec(self) -> dict[str, dict]:
        """获取输出规范（用于模型输入维度推断和 metadata）。

        返回模型输入的规格说明，包括：
        - event: event-level 特征组（shape: [batch_size, feature_dim]）
        - object: object-level 特征组（shape: [batch_size, max_length, feature_dim]）
        - mask: attention mask（shape: [batch_size, max_length]）

        Returns:
            dict: 输出规范，格式为：
                {
                    "event": {
                        "features": ["met", "ht", ...],
                        "dim": 10,  # 特征数量
                    },
                    "object": {
                        "features": ["jet_pt", "jet_eta", ...],
                        "max_length": 128,
                        "dim": 4,  # 每个 object 的特征数量
                    },
                    "mask": {
                        "feature": "jet_mask",  # mask 特征名（如果有）
                        "max_length": 128,
                    },
                }
        """
        spec = {}

        # 收集 event-level 特征
        event_features = []
        for name, node in self.nodes.items():
            feature_type = node.feature_def.get("type", "event")
            # 跳过 mask 特征（单独处理）
            if name.endswith("_mask") or node.feature_def.get("dtype") == "bool":
                continue
            if feature_type == "event":
                event_features.append(name)

        if event_features:
            spec["event"] = {
                "features": event_features,
                "dim": len(event_features),
            }

        # 收集 object-level 特征
        object_features = []
        max_length = None
        mask_feature = None

        for name, node in self.nodes.items():
            feature_type = node.feature_def.get("type", "event")
            if feature_type == "object":
                # 检查是否是 mask
                if name.endswith("_mask") or node.feature_def.get("dtype") == "bool":
                    mask_feature = name
                    # 从 mask 的 padding 配置获取 max_length
                    if "padding" in node.feature_def:
                        max_length = node.feature_def["padding"].get("max_length")
                else:
                    object_features.append(name)
                    # 从 padding 配置获取 max_length
                    if "padding" in node.feature_def:
                        pad_max_length = node.feature_def["padding"].get("max_length")
                        if max_length is None:
                            max_length = pad_max_length
                        elif max_length != pad_max_length:
                            _logger.warning(f"Inconsistent max_length: {max_length} vs {pad_max_length}, using {max_length}")

        if object_features:
            if max_length is None:
                max_length = 128  # 默认值
                _logger.warning(f"No max_length specified for object features, using default: {max_length}")

            spec["object"] = {
                "features": object_features,
                "max_length": max_length,
                "dim": len(object_features),
            }

            # 如果有 mask，添加到 spec
            if mask_feature:
                spec["mask"] = {
                    "feature": mask_feature,
                    "max_length": max_length,
                }

        return spec

    def compile(self):
        """Compile the graph into an execution plan for efficient evaluation.

        The compiled plan is a list of tuples, one per feature in topological
        order.  Each tuple contains:
            (name, processor, expr_or_none, source_or_none, reduction_or_none)

        The ``reduction`` dict (from config) is kept so that ``fit`` and
        ``build_batch`` can apply the correct aggregation without falling
        back to an ``expr`` string.
        """
        if self._compiled_plan is not None:
            return

        execution_order = self.get_execution_order()
        plan = []
        for name in execution_order:
            node = self.nodes[name]
            # Get or create processor
            if name not in self.processors:
                self.processors[name] = FeatureProcessor(node.feature_def)
            processor = self.processors[name]

            # Determine source / expr / reduction
            source = None
            expr = None
            reduction = node.feature_def.get("reduction")

            if "expr" in node.feature_def:
                expr = node.feature_def["expr"]
            else:
                source = node.feature_def.get("source")
                if isinstance(source, list):
                    source = source[0]

            plan.append((name, processor, expr, source, reduction))

        self._compiled_plan = plan
        _logger.debug("FeatureGraph compiled execution plan.")

    def _apply_reduction(self, reduction: dict, source_value, context: dict):
        """Apply a structured reduction to produce an event-level scalar.

        Supported reduction types:
            safe_sum             — sum along axis=-1, returning 0 for empty.
            energy_weighted_mean — Σ(val·w)/Σ(w) along axis=-1.
            sum / mean / max / min / std — standard aggregations.

        Args:
            reduction: Dict with at least ``type`` key, optionally ``weight``.
            source_value: The raw (possibly jagged) array for the source branch.
            context: Current evaluation context (needed to look up weight branch).

        Returns:
            Event-level scalar array (numpy or awkward 1-D).
        """
        rtype = reduction["type"]

        if rtype == "safe_sum":
            return self.expression_engine.registry.get("safe_sum")(source_value)
        elif rtype == "energy_weighted_mean":
            weight_key = reduction["weight"]
            if weight_key not in context:
                raise ValueError(
                    f"Weight branch '{weight_key}' not found in context "
                    f"for energy_weighted_mean reduction."
                )
            weight_value = context[weight_key]
            return self.expression_engine.registry.get("energy_weighted_mean")(
                source_value, weight_value
            )
        elif rtype == "sum":
            return self.expression_engine.registry.get("sum")(source_value)
        elif rtype == "mean":
            return self.expression_engine.registry.get("mean")(source_value)
        elif rtype == "max":
            return self.expression_engine.registry.get("max")(source_value)
        elif rtype == "min":
            return self.expression_engine.registry.get("min")(source_value)
        elif rtype == "std":
            return self.expression_engine.registry.get("std")(source_value)
        else:
            raise ValueError(f"Unknown reduction type: '{rtype}'")

    def fit(self, table: ak.Array):
        """Fit all feature processors (compute normalisation statistics).

        Features are evaluated in topological order.  For each feature the
        framework first checks for a ``reduction`` block (R2 path), then
        falls back to ``expr`` (legacy path), then to a plain ``source``
        pass-through.

        Args:
            table: Training data table used to compute statistics.
        """
        self.compile()

        # Build a mutable evaluation context from the raw data fields.
        context = {k: table[k] for k in table.fields}

        for name, processor, expr, source, reduction in self._compiled_plan:
            try:
                # --- Compute the raw (unreduced) or reduced value ----------
                if reduction is not None:
                    # R2 path: structured reduction block
                    if source not in context:
                        raise ValueError(
                            f"Source '{source}' not found for feature '{name}'"
                        )
                    raw_value = self._apply_reduction(
                        reduction, context[source], context
                    )
                elif expr:
                    # Legacy path: evaluate string expression
                    if self.expression_engine is None:
                        raise ValueError("Expression engine not set.")
                    raw_value = self.expression_engine.evaluate(expr, context)
                else:
                    # Direct source pass-through
                    if source not in context:
                        raise ValueError(
                            f"Source '{source}' not found for feature '{name}'"
                        )
                    raw_value = context[source]

                # Fit normaliser / clipper and process
                processor.fit(raw_value)
                processed_value = processor.process(raw_value)
                context[name] = processed_value

            except Exception as e:
                _logger.error(f"Failed to fit feature '{name}': {e}")
                raise

        _logger.info("FeatureGraph fitted successfully.")

    def export_state(self) -> dict[str, Any]:
        """导出状态（Normalizer参数等）。"""
        state = {}
        for name, processor in self.processors.items():
            if processor.normalizer and processor.normalizer.method == "auto":
                state[name] = {
                    "center": processor.normalizer.center,
                    "scale": processor.normalizer.scale,
                }
        return state

    def load_state(self, state: dict[str, Any]):
        """加载状态。"""
        self.compile()  # Ensure processors exist
        for name, params in state.items():
            if name in self.processors:
                processor = self.processors[name]
                if processor.normalizer:
                    processor.normalizer.center = params.get("center")
                    processor.normalizer.scale = params.get("scale")
                    processor.normalizer._fitted = True
        _logger.info(f"Loaded FeatureGraph state for {len(state)} features.")

    def build_batch(self, table: ak.Array) -> dict[str, torch.Tensor]:
        """从原始数据表构建模型输入批次。

        Args:
            table: 原始数据表（awkward Array）

        Returns:
            dict: 模型输入批次
        """
        if self._compiled_plan is None:
            self.compile()

        # 构建上下文（包含原始数据字段）
        context = {k: table[k] for k in table.fields}

        # Compute all features in topological order.
        for name, processor, expr, source, reduction in self._compiled_plan:
            try:
                # --- Compute the raw value --------------------------------
                if reduction is not None:
                    # R2 path: structured reduction
                    src_val = context.get(source)
                    if src_val is None:
                        raise ValueError(
                            f"Source '{source}' not found in context "
                            f"for feature '{name}'"
                        )
                    raw_value = self._apply_reduction(
                        reduction, src_val, context
                    )
                elif expr:
                    if self.expression_engine is None:
                        raise ValueError("Expression engine not set.")
                    raw_value = self.expression_engine.evaluate(expr, context)
                else:
                    raw_value = context.get(source)
                    if raw_value is None:
                        raise ValueError(
                            f"Source '{source}' not found in context "
                            f"for feature '{name}'"
                        )

                # 处理特征（使用已拟合的参数）
                processed_value = processor.process(raw_value)
                context[name] = processed_value

            except Exception as e:
                _logger.warning(f"Failed to compute feature '{name}': {e}")
                raise

        # 获取输出规范
        output_spec = self.output_spec()

        # 构建模型输入
        batch = {}

        # Event-level 特征
        if "event" in output_spec:
            event_features = output_spec["event"]["features"]
            event_arrays = []
            for feat_name in event_features:
                if feat_name not in context:
                    raise ValueError(f"Feature '{feat_name}' not found in context")
                value = context[feat_name]

                # 转换为 numpy 数组
                if isinstance(value, ak.Array):
                    value = ak.to_numpy(value)
                elif isinstance(value, np.ndarray):
                    pass
                else:
                    value = np.array(value)

                # 确保是 1D 数组（每个事件一个值）
                if value.ndim == 0:
                    value = np.expand_dims(value, 0)
                elif value.ndim > 1:
                    # 如果是多维，展平（可能是有问题的配置）
                    value = value.flatten()

                event_arrays.append(value)

            if event_arrays:
                # 堆叠：shape [num_events, num_features]
                event_tensor = np.stack(event_arrays, axis=1)
                batch["event"] = torch.from_numpy(event_tensor.astype(np.float32))

        # Object-level 特征
        if "object" in output_spec:
            object_features = output_spec["object"]["features"]
            max_length = output_spec["object"]["max_length"]
            object_arrays = []

            for feat_name in object_features:
                if feat_name not in context:
                    raise ValueError(f"Feature '{feat_name}' not found in context")
                value = context[feat_name]

                # 转换为 numpy 数组
                if isinstance(value, ak.Array):
                    # 应该已经被 FeatureProcessor 处理成 padded 的 numpy array
                    value = ak.to_numpy(value) if isinstance(value, ak.Array) else value
                elif isinstance(value, np.ndarray):
                    pass
                else:
                    value = np.array(value)

                # 如果是 jagged array，需要 padding
                # 但通常在 FeatureProcessor 中已经处理过了
                if value.ndim == 1:
                    # 如果是 1D，应该是已经被 pad 的 [num_events * max_length]
                    value = value.reshape(-1, max_length)
                elif value.ndim == 2:
                    # 应该是 [num_events, max_length]
                    if value.shape[1] != max_length:
                        raise ValueError(f"Feature '{feat_name}' has shape {value.shape}, expected [num_events, {max_length}]")
                else:
                    raise ValueError(f"Unexpected shape for object feature '{feat_name}': {value.shape}")

                object_arrays.append(value)

            if object_arrays:
                # 堆叠：shape [num_events, max_length, num_features]
                object_tensor = np.stack(object_arrays, axis=2)  # [B, N, D]
                batch["object"] = torch.from_numpy(object_tensor.astype(np.float32))

                # Mask: 如果有 object 特征，总是需要创建 mask
                if "mask" in output_spec:
                    mask_feature = output_spec["mask"]["feature"]
                    if mask_feature in context:
                        mask_value = context[mask_feature]

                        # 转换为 numpy 数组
                        if isinstance(mask_value, ak.Array):
                            mask_value = ak.to_numpy(mask_value)
                        elif isinstance(mask_value, np.ndarray):
                            pass
                        else:
                            mask_value = np.array(mask_value, dtype=bool)

                        # 确保 shape 是 [num_events, max_length]
                        if mask_value.ndim == 1:
                            mask_value = mask_value.reshape(-1, max_length)
                        elif mask_value.ndim == 2:
                            if mask_value.shape[1] != max_length:
                                raise ValueError(f"Mask has shape {mask_value.shape}, expected [num_events, {max_length}]")
                        else:
                            raise ValueError(f"Unexpected shape for mask: {mask_value.shape}")

                        batch["mask"] = torch.from_numpy(mask_value.astype(bool))
                    else:
                        # 如果没有 mask 特征，生成默认 mask（所有位置都是 True）
                        num_events = len(table)
                        batch["mask"] = torch.ones((num_events, max_length), dtype=torch.bool)
                else:
                    # 如果没有显式定义 mask 特征，生成默认 mask（所有位置都是 True）
                    num_events = len(table)
                    batch["mask"] = torch.ones((num_events, max_length), dtype=torch.bool)

        # 如果没有 event 或 object，至少需要有一个输入
        if not batch:
            raise ValueError("No features found in output_spec. At least one of 'event' or 'object' must be specified.")

        return batch

    def get_source_branches(self) -> set[str]:
        """Return all ROOT-level branch paths required by features.

        Scans each feature definition for ``source`` and ``reduction.weight``
        to build the complete set of branches that must be loaded from ROOT
        files.  Only branches that are NOT themselves computed features are
        returned.

        Returns:
            Set of ROOT branch path strings.
        """
        branches: set[str] = set()
        for _name, node in self.nodes.items():
            fdef = node.feature_def
            src = fdef.get("source")
            if src and src not in self.nodes:
                branches.add(src)
            # Reduction weight branch
            reduction = fdef.get("reduction")
            if reduction:
                weight = reduction.get("weight")
                if weight and weight not in self.nodes:
                    branches.add(weight)
            # Legacy expr dependencies
            if "expr" in fdef and self.expression_engine:
                try:
                    deps = self.expression_engine.get_dependencies(fdef["expr"])
                    for dep in deps:
                        if dep not in self.nodes:
                            # External dependency — must be a raw branch.
                            branches.add(dep)
                        elif dep == _name and src is None:
                            # Self-referencing pass-through (e.g. feature
                            # "met" with expr "met"): the feature name IS
                            # the raw branch that needs loading.
                            branches.add(dep)
                except Exception:
                    pass
        return branches

    @classmethod
    def from_feature_defs(cls, features: dict[str, dict], expression_engine, enable_cache: bool = True) -> FeatureGraph:
        """Build a FeatureGraph from a dictionary of feature definitions.

        Args:
            features: Mapping of feature name → feature definition dict.
            expression_engine: ExpressionEngine instance (for expr evaluation
                and operator look-up).
            enable_cache: Whether to enable value caching.  Default ``True``.

        Returns:
            Fully-wired FeatureGraph ready for ``compile`` / ``fit``.
        """
        graph = cls(expression_engine=expression_engine, enable_cache=enable_cache)

        # 1. Create all nodes.
        for feature_name, feature_def in features.items():
            graph.add_node(feature_name, feature_def)

        # 2. Resolve inter-feature dependencies and add edges.
        for feature_name, feature_def in features.items():
            dependencies: set[str] = set()

            # From expr (legacy path)
            if "expr" in feature_def:
                expr = feature_def["expr"]
                try:
                    deps = expression_engine.get_dependencies(expr)
                    deps = {dep for dep in deps if dep in features}
                    dependencies.update(deps)
                except Exception:
                    pass

            # From source (if it is itself a computed feature)
            source = feature_def.get("source", [])
            if isinstance(source, str):
                if source in features:
                    dependencies.add(source)
            elif isinstance(source, list):
                for s in source:
                    if s in features:
                        dependencies.add(s)

            # From reduction.weight (if it is a computed feature)
            reduction = feature_def.get("reduction")
            if reduction:
                weight = reduction.get("weight")
                if weight and weight in features:
                    dependencies.add(weight)

            # Explicit dependencies
            if "dependencies" in feature_def:
                explicit_deps = feature_def["dependencies"]
                if isinstance(explicit_deps, list):
                    for dep in explicit_deps:
                        if dep in features:
                            dependencies.add(dep)
                elif isinstance(explicit_deps, str):
                    if explicit_deps in features:
                        dependencies.add(explicit_deps)

            # Final filter — only in-graph features
            dependencies = {dep for dep in dependencies if dep in features}

            for dep in dependencies:
                if dep in graph.nodes and feature_name != dep:
                    graph.add_edge(feature_name, dep)

        return graph

    @classmethod
    def from_yaml(cls, yaml_path: str, expression_engine, enable_cache: bool = True) -> FeatureGraph:
        """从 YAML 配置文件构建图。

        Args:
            yaml_path (str): YAML 文件路径
            expression_engine: 表达式引擎（用于提取依赖和 build_batch）
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

        if "features" in config:
            # 处理 event_level 特征
            if "event_level" in config["features"]:
                for feature in config["features"]["event_level"]:
                    if "name" in feature:
                        all_features[feature["name"]] = feature

            # 处理 object_level 特征
            if "object_level" in config["features"]:
                for feature in config["features"]["object_level"]:
                    if "name" in feature:
                        all_features[feature["name"]] = feature

        if not all_features:
            raise ValueError("No features found in YAML configuration")

        _logger.info(f"Loaded {len(all_features)} features from {yaml_path}")

        # 构建图
        return cls.from_feature_defs(all_features, expression_engine, enable_cache=enable_cache)
