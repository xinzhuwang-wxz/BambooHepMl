# Engine Module

训练引擎模块。

## 核心组件

### Trainer

统一训练器，支持：
- 监督 / 半监督 / 无监督学习
- 多任务损失
- Callback 系统

### Evaluator

评估器，计算各种指标。

### Predictor

预测器，支持批量预测和概率输出。

### Callbacks

回调系统：
- LoggingCallback
- EarlyStoppingCallback
- MLflowCallback
- TensorBoardCallback

