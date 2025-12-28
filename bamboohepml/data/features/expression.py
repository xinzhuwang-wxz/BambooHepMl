"""
表达式引擎模块

提供：
- ExpressionEngine: 表达式求值引擎
- OperatorRegistry: 函数注册表（支持自定义函数）
- 向量化计算支持
- 表达式与数据源解耦
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable

import awkward as ak
import numpy as np

from ..logger import _logger


class OperatorRegistry:
    """
    函数注册表

    支持注册自定义函数，用于表达式求值。
    """

    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._register_builtin_functions()

    def register(self, name: str, func: Callable, override: bool = False):
        """注册函数。

        Args:
            name (str): 函数名
            func (Callable): 函数对象
            override (bool): 是否覆盖已存在的函数。默认为 False。
        """
        if name in self._functions and not override:
            raise ValueError(f"Function '{name}' already registered. Use override=True to replace it.")
        self._functions[name] = func
        _logger.debug(f"Registered function: {name}")

    def unregister(self, name: str):
        """注销函数。

        Args:
            name (str): 函数名
        """
        if name in self._functions:
            del self._functions[name]
            _logger.debug(f"Unregistered function: {name}")

    def get(self, name: str) -> Callable | None:
        """获取函数。

        Args:
            name (str): 函数名

        Returns:
            Callable: 函数对象，如果不存在返回 None
        """
        return self._functions.get(name)

    def has(self, name: str) -> bool:
        """检查函数是否存在。

        Args:
            name (str): 函数名

        Returns:
            bool: 是否存在
        """
        return name in self._functions

    def _register_builtin_functions(self):
        """注册内置函数。"""
        # 聚合函数（用于 object-level -> event-level）
        self.register("sum", lambda x: ak.sum(x, axis=-1) if isinstance(x, ak.Array) else np.sum(x))
        self.register("mean", lambda x: ak.mean(x, axis=-1) if isinstance(x, ak.Array) else np.mean(x))
        self.register("max", lambda x: ak.max(x, axis=-1) if isinstance(x, ak.Array) else np.max(x))
        self.register("min", lambda x: ak.min(x, axis=-1) if isinstance(x, ak.Array) else np.min(x))
        self.register("std", lambda x: ak.std(x, axis=-1) if isinstance(x, ak.Array) else np.std(x))
        self.register("len", lambda x: ak.num(x) if isinstance(x, ak.Array) else len(x))
        self.register("count", lambda x, condition: ak.sum(condition, axis=-1) if isinstance(x, ak.Array) else np.sum(condition))

        # 数学函数
        self.register("log", lambda x: np.log(x) if isinstance(x, np.ndarray) else ak.log(x))
        self.register("log1p", lambda x: np.log1p(x) if isinstance(x, np.ndarray) else ak.log1p(x))
        self.register("exp", lambda x: np.exp(x) if isinstance(x, np.ndarray) else ak.exp(x))

        # sqrt: 对于 jagged array，使用 ak 的逐元素操作
        def sqrt_func(x):
            if isinstance(x, ak.Array):
                # 对于 jagged array，使用 ak 的逐元素操作保持结构
                # 使用 ak.values_astype 和 ak.Array 来保持结构
                try:
                    # Try using ak's vectorized operations if available
                    if hasattr(ak, "sqrt"):
                        return ak.sqrt(x)
                    else:
                        # Fallback: use list comprehension for jagged arrays
                        # This preserves the jagged structure
                        return ak.Array([np.sqrt(item) if np.isscalar(item) else np.sqrt(item) for item in x])
                except Exception:
                    # Final fallback: flatten, compute, unflatten
                    flat = ak.flatten(x, axis=None)
                    sqrt_flat = np.sqrt(ak.to_numpy(flat))
                    counts = ak.num(x)
                    return ak.unflatten(ak.Array(sqrt_flat), counts)
            return np.sqrt(x)

        self.register("sqrt", sqrt_func)
        self.register("abs", lambda x: np.abs(ak.to_numpy(x)) if isinstance(x, ak.Array) else np.abs(x))
        self.register("sin", lambda x: np.sin(ak.to_numpy(x)) if isinstance(x, ak.Array) else np.sin(x))
        self.register("cos", lambda x: np.cos(ak.to_numpy(x)) if isinstance(x, ak.Array) else np.cos(x))

        # HEP 专用函数
        self.register("delta_r", self._delta_r)
        self.register("delta_phi", self._delta_phi)
        self.register("delta_eta", lambda eta1, eta2: eta1 - eta2)

    @staticmethod
    def _delta_r(eta1, phi1, eta2, phi2):
        """计算 ΔR = sqrt(Δη² + Δφ²)。"""
        deta = eta1 - eta2
        dphi = OperatorRegistry._delta_phi(phi1, phi2)
        return np.sqrt(deta**2 + dphi**2)

    @staticmethod
    def _delta_phi(phi1, phi2):
        """计算 Δφ（考虑周期性）。"""
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        return dphi


class ExpressionEngine:
    """
    表达式引擎

    支持：
    - 字符串表达式解析和求值
    - 向量化计算（numpy/awkward）
    - 表达式与数据源解耦
    - 函数注册机制
    """

    def __init__(self, registry: OperatorRegistry | None = None):
        """初始化表达式引擎。

        Args:
            registry (OperatorRegistry, optional): 函数注册表。如果为 None，创建默认注册表。
        """
        self.registry = registry if registry is not None else OperatorRegistry()
        self._exclude_names = {"awkward", "ak", "np", "numpy", "math", "len", "abs", "sum", "mean", "max", "min"}

    def get_dependencies(self, expr: str) -> set[str]:
        """从表达式提取变量名（依赖）。

        Args:
            expr (str): 表达式字符串

        Returns:
            Set[str]: 变量名集合
        """
        try:
            root = ast.parse(expr)
            variables = set()

            # 遍历 AST，提取所有变量名
            for node in ast.walk(root):
                if isinstance(node, ast.Name):
                    name = node.id
                    # 排除内置函数和模块名
                    if not name.startswith("_") and name not in self._exclude_names:
                        # 检查是否是函数调用
                        # 如果是 Call 节点的 func，则跳过（是函数名，不是变量）
                        is_function_call = False
                        for parent in ast.walk(root):
                            if isinstance(parent, ast.Call) and parent.func == node:
                                is_function_call = True
                                break

                        if not is_function_call:
                            variables.add(name)

                # 处理属性访问（如 Jet.pt）
                elif isinstance(node, ast.Attribute):
                    # 提取对象名（如 Jet.pt 中的 Jet）
                    if isinstance(node.value, ast.Name):
                        obj_name = node.value.id
                        if obj_name not in self._exclude_names:
                            variables.add(obj_name)

            return variables
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {expr}") from e

    def validate(self, expr: str) -> bool:
        """验证表达式语法。

        Args:
            expr (str): 表达式字符串

        Returns:
            bool: 是否有效
        """
        try:
            ast.parse(expr)
            return True
        except SyntaxError:
            return False

    def evaluate(self, expr: str, context: dict[str, Any]) -> Any:
        """求值表达式。

        Args:
            expr (str): 表达式字符串
            context (Dict[str, Any]): 上下文（变量字典）

        Returns:
            Any: 求值结果
        """
        if not self.validate(expr):
            raise ValueError(f"Invalid expression: {expr}")

        # 提取依赖
        dependencies = self.get_dependencies(expr)

        # 构建求值上下文
        eval_context = {}

        # 1. 添加变量（从 context 中获取）
        for var_name in dependencies:
            if var_name in context:
                eval_context[var_name] = context[var_name]
            else:
                # 尝试属性访问（如 Jet.pt）
                parts = var_name.split(".")
                if len(parts) == 2 and parts[0] in context:
                    obj = context[parts[0]]
                    if hasattr(obj, parts[1]):
                        eval_context[var_name] = getattr(obj, parts[1])
                    elif isinstance(obj, dict) and parts[1] in obj:
                        eval_context[var_name] = obj[parts[1]]
                    else:
                        raise ValueError(f"Variable '{var_name}' not found in context")
                else:
                    raise ValueError(f"Variable '{var_name}' not found in context")

        # 2. 添加内置模块
        eval_context.update(
            {
                "math": math,
                "np": np,
                "numpy": np,
                "ak": ak,
                "awkward": ak,
                "len": len,
            }
        )

        # 3. 添加注册的函数
        for func_name, func in self.registry._functions.items():
            eval_context[func_name] = func

        # 4. 支持属性访问（如 Jet.pt）
        # 在表达式中，Jet.pt 会被解析为属性访问
        # 我们需要在上下文中提供对象
        # 对于属性访问，我们需要在上下文中提供对象本身
        # 例如，如果表达式是 "Jet.pt"，我们需要在上下文中提供 "Jet" 对象

        # 添加对象到上下文（用于属性访问）
        for var_name in dependencies:
            if "." in var_name:
                obj_name = var_name.split(".")[0]
                if obj_name in context and obj_name not in eval_context:
                    eval_context[obj_name] = context[obj_name]

        try:
            # 使用 compile + eval 进行安全求值
            code = compile(expr, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, eval_context)
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expr}': {str(e)}") from e

    def register_function(self, name: str, func: Callable, override: bool = False):
        """注册自定义函数。

        Args:
            name (str): 函数名
            func (Callable): 函数对象
            override (bool): 是否覆盖已存在的函数。默认为 False。
        """
        self.registry.register(name, func, override=override)

    def unregister_function(self, name: str):
        """注销函数。

        Args:
            name (str): 函数名
        """
        self.registry.unregister(name)
