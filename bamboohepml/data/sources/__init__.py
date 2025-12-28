"""
数据源模块

提供：
- 数据源抽象接口
- 具体数据源实现（ROOT, Parquet, HDF5）
- 数据源与特征系统解耦
"""

from .base import DataSource, DataSourceConfig
from .root_source import ROOTDataSource
from .parquet_source import ParquetDataSource
from .hdf5_source import HDF5DataSource
from .factory import DataSourceFactory

__all__ = [
    "DataSource",
    "DataSourceConfig",
    "ROOTDataSource",
    "ParquetDataSource",
    "HDF5DataSource",
    "DataSourceFactory",
]

