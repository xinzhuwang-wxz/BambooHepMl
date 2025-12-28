# Models Module

模型定义模块。

## 核心组件

### BaseModel

基础模型类，提供通用接口。

::: bamboohepml.models.base.BaseModel
    options:
      show_source: true
      heading_level: 3

### ClassificationModel

分类模型基类。

::: bamboohepml.models.base.ClassificationModel
    options:
      show_source: true
      heading_level: 3

### RegressionModel

回归模型基类。

::: bamboohepml.models.base.RegressionModel
    options:
      show_source: true
      heading_level: 3

### MultitaskModel

多任务模型基类。

::: bamboohepml.models.base.MultitaskModel
    options:
      show_source: true
      heading_level: 3

### ModelRegistry

模型注册表，支持动态模型加载。

::: bamboohepml.models.registry.ModelRegistry
    options:
      show_source: true
      heading_level: 3

### get_model

模型工厂函数，根据名称获取模型实例。

::: bamboohepml.models.registry.get_model
    options:
      show_source: true
      heading_level: 3

### register_model

模型注册装饰器，用于注册模型类。

::: bamboohepml.models.registry.register_model
    options:
      show_source: true
      heading_level: 3

### MLPClassifier

MLP 分类器实现。

::: bamboohepml.models.common.mlp.MLPClassifier
    options:
      show_source: true
      heading_level: 3

### MLPRegressor

MLP 回归器实现。

::: bamboohepml.models.common.mlp.MLPRegressor
    options:
      show_source: true
      heading_level: 3
