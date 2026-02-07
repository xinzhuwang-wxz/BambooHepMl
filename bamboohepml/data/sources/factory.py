"""
数据源工厂

根据文件类型自动创建相应的数据源。
"""

import glob
import os

from .base import DataSource, DataSourceConfig
from .hdf5_source import HDF5DataSource
from .parquet_source import ParquetDataSource
from .root_source import ROOTDataSource


class DataSourceFactory:
    """数据源工厂类。"""

    @staticmethod
    def create(file_paths: str | list[str], **kwargs) -> DataSource:
        """根据文件类型创建数据源。

        Args:
            file_paths: 文件路径（字符串、列表或 glob 模式）
            **kwargs: 其他配置参数（treename, branch_magic, file_magic, load_range）

        Returns:
            DataSource: 数据源实例
        """
        # 解析文件路径
        if isinstance(file_paths, str):
            resolved = glob.glob(file_paths)
            if not resolved:
                resolved = [file_paths]
        else:
            resolved = list(file_paths)

        if len(resolved) == 0:
            raise ValueError("No files found")

        # 根据文件扩展名确定数据源类型
        first_file = resolved[0]
        ext = os.path.splitext(first_file)[1].lower()

        # 创建配置
        config = DataSourceConfig(
            file_paths=file_paths,
            treename=kwargs.get("treename"),
            branch_magic=kwargs.get("branch_magic"),
            file_magic=kwargs.get("file_magic"),
            load_range=kwargs.get("load_range"),
            class_labels=kwargs.get("class_labels"),
        )

        # 创建相应的数据源
        if ext == ".root":
            return ROOTDataSource(config)
        elif ext == ".parquet":
            return ParquetDataSource(config)
        elif ext in (".h5", ".hdf5"):
            return HDF5DataSource(config)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported types: .root, .parquet, .h5, .hdf5")

    @staticmethod
    def from_config(config: DataSourceConfig) -> DataSource:
        """
        从配置创建数据源。

        Args:
            config: 数据源配置

        Returns:
            DataSource: 数据源实例
        """
        if len(config.file_paths) == 0:
            raise ValueError("No files in config")

        first_file = config.file_paths[0]
        ext = os.path.splitext(first_file)[1].lower()

        if ext == ".root":
            return ROOTDataSource(config)
        elif ext == ".parquet":
            return ParquetDataSource(config)
        elif ext in (".h5", ".hdf5"):
            return HDF5DataSource(config)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported types: .root, .parquet, .h5, .hdf5")
