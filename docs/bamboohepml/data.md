# Data Module

数据与特征系统模块，借鉴 weaver-core 的实现。

## 核心组件

### DataConfig

数据配置类，管理数据加载和特征配置。

### HEPDataset

高能物理数据集类，支持：
- Jagged arrays（变长数组）
- Padding 和 mask
- Transformer 输入格式

### FeatureGraph

特征依赖图，自动构建特征依赖关系。

### ExpressionEngine

表达式引擎，支持向量化计算和自定义函数。

