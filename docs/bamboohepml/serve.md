# Serve Module

服务部署模块。

## 核心组件

### FastAPI Server

FastAPI 推理服务，提供 RESTful API。

::: bamboohepml.serve.fastapi_server.create_app
    options:
      show_source: true
      heading_level: 3

### ONNXPredictor

ONNX 推理接口，支持 CPU/GPU 推理。

::: bamboohepml.serve.onnx_predictor.ONNXPredictor
    options:
      show_source: true
      heading_level: 3

### RayServeDeployment

Ray Serve 部署类，使用 Ray Serve 部署模型服务。

::: bamboohepml.serve.ray_serve.RayServeDeployment
    options:
      show_source: true
      heading_level: 3
