# CLI Module

命令行接口模块。

## 命令

### train

训练模型命令。

::: bamboohepml.cli.train
    options:
      show_source: true
      heading_level: 3

### predict

预测命令，使用模型进行预测。

::: bamboohepml.cli.predict
    options:
      show_source: true
      heading_level: 3

### export

导出命令，导出模型为 ONNX 格式。

::: bamboohepml.cli.export
    options:
      show_source: true
      heading_level: 3

### inspect

检查命令，检查数据和特征。

::: bamboohepml.cli.inspect
    options:
      show_source: true
      heading_level: 3

### serve

服务命令，启动推理服务（FastAPI 或 Ray Serve）。

::: bamboohepml.serve.cli.serve_app
    options:
      show_source: true
      heading_level: 3
