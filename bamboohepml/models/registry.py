"""
模型注册表

提供模型注册和发现机制，支持动态注册模型。
"""

from typing import Dict, Optional, Type

from .base import BaseModel


class ModelRegistry:
    """
    模型注册表（单例）

    用于注册和发现模型类。
    """

    _instance = None
    _registry: Dict[str, Type[BaseModel]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, model_class: Type[BaseModel]):
        """
        注册模型类。

        Args:
            name: 模型名称（用于查找）
            model_class: 模型类（必须是 BaseModel 的子类）
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must be a subclass of BaseModel, got {model_class}")
        self._registry[name] = model_class

    def get(self, name: str) -> Optional[Type[BaseModel]]:
        """
        获取模型类。

        Args:
            name: 模型名称

        Returns:
            模型类，如果不存在则返回 None
        """
        return self._registry.get(name)

    def list_models(self) -> list:
        """
        列出所有注册的模型名称。

        Returns:
            模型名称列表
        """
        return list(self._registry.keys())

    def unregister(self, name: str):
        """
        取消注册模型。

        Args:
            name: 模型名称
        """
        self._registry.pop(name, None)


# 全局注册表实例
_registry = ModelRegistry()


def register_model(name: str):
    """
    装饰器：注册模型类。

    Usage:
        @register_model('mlp_classifier')
        class MLPClassifier(BaseModel):
            ...
    """

    def decorator(model_class: Type[BaseModel]):
        _registry.register(name, model_class)
        return model_class

    return decorator


def get_model(model_name: str, task_type: str = "classification", **kwargs) -> BaseModel:
    """
    模型工厂函数：根据名称创建模型实例。

    Args:
        model_name: 模型名称（必须在注册表中）
        task_type: 任务类型（'classification', 'regression', 'multitask'）
        **kwargs: 传递给模型构造函数的参数

    Returns:
        模型实例

    Raises:
        ValueError: 如果模型名称不存在
    """
    model_class = _registry.get(model_name)
    if model_class is None:
        available_models = ", ".join(_registry.list_models())
        raise ValueError(f"Model '{model_name}' not found in registry. " f"Available models: {available_models}")

    # 添加任务类型到 kwargs
    kwargs["task_type"] = task_type

    return model_class(**kwargs)
