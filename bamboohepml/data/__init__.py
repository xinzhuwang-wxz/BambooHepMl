"""
数据模块

提供：
- 数据源（ROOT/Parquet/HDF5）
- Dataset 类（支持 jagged array, padding, mask）
- 数据配置
- 特征工程集成
"""

from .config import DataConfig
from .dataset import HEPDataset, TransformerDataset
from .fileio import read_files
from .preprocess import AutoStandardizer, WeightMaker
from .sources import DataSource, DataSourceFactory, HDF5DataSource, ParquetDataSource, ROOTDataSource

__all__ = [
    "DataConfig",
    "HEPDataset",
    "TransformerDataset",
    "DataSource",
    "DataSourceFactory",
    "ROOTDataSource",
    "ParquetDataSource",
    "HDF5DataSource",
    "AutoStandardizer",
    "WeightMaker",
    "read_files",
]
