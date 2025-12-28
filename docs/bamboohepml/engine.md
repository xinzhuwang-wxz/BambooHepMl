# Engine Module

训练引擎模块。

## 核心组件

### Trainer

统一训练器，支持：
- 监督 / 半监督 / 无监督学习
- 多任务损失
- Callback 系统

::: bamboohepml.engine.trainer.Trainer
    options:
      show_source: true
      heading_level: 3

### Evaluator

评估器，计算各种指标。

::: bamboohepml.engine.evaluator.Evaluator
    options:
      show_source: true
      heading_level: 3

### Predictor

预测器，支持批量预测和概率输出。

::: bamboohepml.engine.predictor.Predictor
    options:
      show_source: true
      heading_level: 3

### Callback

回调基类，所有回调都应该继承此类。

::: bamboohepml.engine.callbacks.Callback
    options:
      show_source: true
      heading_level: 3

### LoggingCallback

日志回调，记录训练过程中的信息。

::: bamboohepml.engine.callbacks.LoggingCallback
    options:
      show_source: true
      heading_level: 3

### EarlyStoppingCallback

早停回调，根据验证指标自动停止训练。

::: bamboohepml.engine.callbacks.EarlyStoppingCallback
    options:
      show_source: true
      heading_level: 3

### MLflowCallback

MLflow 回调，自动记录实验到 MLflow。

::: bamboohepml.engine.callbacks.MLflowCallback
    options:
      show_source: true
      heading_level: 3

### TensorBoardCallback

TensorBoard 回调，自动记录实验到 TensorBoard。

::: bamboohepml.engine.callbacks.TensorBoardCallback
    options:
      show_source: true
      heading_level: 3
