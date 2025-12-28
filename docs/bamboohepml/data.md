# Data Module

数据与特征系统模块，借鉴 weaver-core 的实现。

## 核心组件

### DataConfig

数据配置类，管理数据加载和特征配置。

::: bamboohepml.data.config.DataConfig
    options:
      show_source: true
      heading_level: 3

### HEPDataset

高能物理数据集类，支持：
- Jagged arrays（变长数组）
- Padding 和 mask
- Transformer 输入格式

::: bamboohepml.data.dataset.HEPDataset
    options:
      show_source: true
      heading_level: 3

### TransformerDataset

Transformer 数据集类，提供 transformer 模型所需的输入格式。

::: bamboohepml.data.dataset.TransformerDataset
    options:
      show_source: true
      heading_level: 3

### DataSource

数据源抽象基类，支持多种数据格式。

::: bamboohepml.data.sources.base.DataSource
    options:
      show_source: true
      heading_level: 3

### DataSourceFactory

数据源工厂，根据文件扩展名自动创建相应的数据源。

::: bamboohepml.data.sources.factory.DataSourceFactory
    options:
      show_source: true
      heading_level: 3

### ROOTDataSource

ROOT 文件数据源，使用 uproot 读取 ROOT 文件。

::: bamboohepml.data.sources.root_source.ROOTDataSource
    options:
      show_source: true
      heading_level: 3

### ParquetDataSource

Parquet 文件数据源，使用 awkward 读取 Parquet 文件。

::: bamboohepml.data.sources.parquet_source.ParquetDataSource
    options:
      show_source: true
      heading_level: 3

### HDF5DataSource

HDF5 文件数据源，使用 tables 读取 HDF5 文件。

::: bamboohepml.data.sources.hdf5_source.HDF5DataSource
    options:
      show_source: true
      heading_level: 3

### FeatureGraph

特征依赖图，自动构建特征依赖关系。

::: bamboohepml.data.features.feature_graph.FeatureGraph
    options:
      show_source: true
      heading_level: 3

### ExpressionEngine

表达式引擎，支持向量化计算和自定义函数。

::: bamboohepml.data.features.expression.ExpressionEngine
    options:
      show_source: true
      heading_level: 3

### OperatorRegistry

函数注册表，支持注册自定义函数用于表达式求值。

::: bamboohepml.data.features.expression.OperatorRegistry
    options:
      show_source: true
      heading_level: 3

### AutoStandardizer

自动标准化器，自动计算标准化参数。

::: bamboohepml.data.preprocess.AutoStandardizer
    options:
      show_source: true
      heading_level: 3

### WeightMaker

权重生成器，生成重加权直方图。

::: bamboohepml.data.preprocess.WeightMaker
    options:
      show_source: true
      heading_level: 3
