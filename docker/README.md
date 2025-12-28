# Docker 支持

BambooHepMl 提供两种 Docker 镜像：

## 1. CPU 版本

使用标准 Dockerfile：

```bash
docker build -t bamboohepml:latest .
```

## 2. GPU 版本

使用 GPU Dockerfile（需要 NVIDIA Docker runtime）：

```bash
docker build -f docker/Dockerfile.gpu -t bamboohepml:gpu .
```

### 运行 GPU 容器

```bash
docker run --gpus all -it bamboohepml:gpu python -m bamboohepml.cli train -c configs/pipeline.yaml
```

## 使用示例

### 训练

```bash
# CPU
docker run -v $(pwd)/configs:/app/configs -v $(pwd)/data:/app/data bamboohepml:latest \
    python -m bamboohepml.cli train -c configs/pipeline.yaml

# GPU
docker run --gpus all -v $(pwd)/configs:/app/configs -v $(pwd)/data:/app/data bamboohepml:gpu \
    python -m bamboohepml.cli train -c configs/pipeline.yaml
```

### 推理服务

```bash
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs bamboohepml:latest \
    python -m bamboohepml.serve.fastapi_server serve_fastapi \
    --model-path outputs/model.pt --metadata-path outputs/metadata.json
```

## 要求

- Docker 20.10+
- 对于 GPU 版本：NVIDIA Docker runtime (nvidia-docker2)
