# Tasks Module

任务子系统模块。

## 核心任务

### train_task

训练任务，支持本地和 Ray 分布式训练。

::: bamboohepml.tasks.train.train_task
    options:
      show_source: true
      heading_level: 3

### predict_task

预测任务，支持批量预测。

::: bamboohepml.tasks.predict.predict_task
    options:
      show_source: true
      heading_level: 3

### export_task

导出任务，支持 ONNX 格式导出。

::: bamboohepml.tasks.export.export_task
    options:
      show_source: true
      heading_level: 3

### inspect_task

检查任务，检查数据和特征。

::: bamboohepml.tasks.inspect.inspect_task
    options:
      show_source: true
      heading_level: 3
