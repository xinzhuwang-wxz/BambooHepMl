"""
数据源基类

定义数据源的抽象接口，实现数据源与特征系统的解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import awkward as ak


@dataclass
class DataSourceConfig:
    """数据源配置。

    Attributes:
        file_paths: 文件路径列表或 glob 模式
        treename: 树名称（ROOT 文件）
        branch_magic: 分支名称映射（ROOT 文件）
        file_magic: 文件魔法变量（根据文件名模式设置变量值）— 已废弃
        load_range: 加载范围 (start, end)，范围在 [0, 1]
        class_labels: 文件路径 → 类别索引映射（由 classes 标签系统生成）
    """

    file_paths: str | list[str]
    treename: str | None = None
    branch_magic: dict[str, str] | None = None
    file_magic: dict[str, dict[str, Any]] | None = None
    load_range: tuple | None = None
    class_labels: dict[str, int] | None = None


class DataSource(ABC):
    """
    数据源抽象基类

    定义数据源的统一接口，实现数据源与特征系统的解耦。
    """

    def __init__(self, config: DataSourceConfig):
        """初始化数据源。

        Args:
            config (DataSourceConfig): 数据源配置
        """
        self.config = config
        self._file_paths = self._resolve_file_paths(config.file_paths)

    @staticmethod
    def _resolve_file_paths(file_paths: str | list[str]) -> list[str]:
        """解析文件路径。

        Args:
            file_paths: 文件路径（字符串、列表或 glob 模式）

        Returns:
            List[str]: 文件路径列表
        """
        import glob

        if isinstance(file_paths, str):
            # 可能是 glob 模式
            resolved = glob.glob(file_paths)
            if not resolved:
                # 如果不是 glob，可能是单个文件
                resolved = [file_paths]
            return resolved
        elif isinstance(file_paths, (list, tuple)):
            return list(file_paths)
        else:
            raise ValueError(f"Invalid file_paths type: {type(file_paths)}")

    @abstractmethod
    def load_branches(self, branches: list[str]) -> ak.Array:
        """加载指定的分支。

        Args:
            branches (List[str]): 要加载的分支列表

        Returns:
            ak.Array: 加载的数据（awkward Array）
        """
        pass

    @abstractmethod
    def get_available_branches(self) -> list[str]:
        """获取可用的分支列表。

        Returns:
            List[str]: 可用分支列表
        """
        pass

    def get_file_paths(self) -> list[str]:
        """获取文件路径列表。

        Returns:
            List[str]: 文件路径列表
        """
        return self._file_paths.copy()

    def get_num_events(self) -> int | None:
        """获取事件数量（如果可用）。

        Returns:
            int: 事件数量，如果无法确定返回 None
        """
        return None
