# BambooHepMl

> 一个面向高能物理的现代机器学习框架，结合了强大的特征工程能力和完整的 ML 工程实践。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 特性

- 🎯 **配置驱动**：通过 YAML 配置完成所有特征工程，无需硬编码
- 🧩 **模块化设计**：清晰的模块职责，易于扩展和维护
- 🚀 **生产就绪**：完整的 ML pipeline（data → model → train → eval → export → serve）
- ⚛️ **高能物理优化**：专为 HEP 数据格式和任务设计
- 🔄 **灵活学习范式**：支持监督、半监督、无监督学习
- 📦 **开箱即用**：集成 Docker、ONNX、FastAPI、Ray Serve 等现代工具

## 🏗️ 架构

```
BambooHepMl/
├── bamboohepml/          # 核心包
│   ├── data/             # 数据与特征系统
│   ├── models/           # 模型定义
│   ├── engine/           # 训练引擎
│   ├── tasks/            # 任务子系统
│   ├── pipeline/         # Pipeline 编排
│   ├── scheduler/        # 调度系统
│   ├── serve/            # 服务部署
│   └── experiment/       # 实验跟踪
├── tests/                # 测试套件
├── configs/              # 配置示例
└── docs/                 # 文档
```

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 基本使用

```bash
# 训练模型
bamboohepml train -c configs/pipeline.yaml --experiment-name my_exp

# 预测
bamboohepml predict -c configs/pipeline.yaml -m outputs/model.pt -o predictions.root

# 导出 ONNX
bamboohepml export -c configs/pipeline.yaml -m outputs/model.pt -o model.onnx

# 启动推理服务
bamboohepml serve fastapi -m outputs/model.pt -c configs/pipeline.yaml
```

## 📖 配置指南

### Pipeline 配置 (`pipeline.yaml`)

```yaml
data:
  config_path: "configs/data.yaml"
  source_path: "data/train.root"
  treename: "Events"
  val_split: 0.1

features:
  config_path: "configs/features.yaml"

model:
  name: "ParticleTransformer"
  params:
    num_classes: 2
    hidden_dim: 128

train:
  num_epochs: 20
  batch_size: 128
  learning_rate: 0.001
  task_type: "classification"
  learning_paradigm: "supervised"
```

### 特征配置 (`features.yaml`)

```yaml
features:
  event_level:                      # Event 级别特征
    - name: "met"
      source: "MET"
      dtype: "float32"
      normalize:
        method: "auto"

    - name: "ht"
      expr: "sum(Jet_pt)"           # 表达式特征
      dtype: "float32"
      normalize:
        method: "manual"
        center: 100.0
        scale: 0.01

  object_level:                     # Object 级别特征（变长序列）
    - name: "jet_pt"
      source: "Jet_pt"
      dtype: "float32"
      normalize:
        method: "auto"
      clip:
        min: 0.0
        max: 500.0
      padding:
        max_length: 128
        mode: "constant"
        value: 0.0
```

### 数据配置 (`data.yaml`)

#### 分类任务

**方式 1: 字典方式（推荐）**

```bash
# 命令行配置
data_train="B:/path/to/bb/*.root Bbar:/path/to/bbbar/*.root C:/path/to/cc/*.root"
```

系统自动生成标签配置并推断类别数。

**方式 2: 手动配置**

```yaml
train_load_branches:
  - "MET"
  - "Jet_pt"
  - "is_B"
  - "is_Bbar"
  - "is_C"

test_load_branches:
  - "MET"
  - "Jet_pt"

labels:
  type: "simple"
  value:
    - "is_B"
    - "is_Bbar"
    - "is_C"
```

#### 回归任务

```yaml
train_load_branches:
  - "MET"
  - "Jet_pt"
  - "target_value"

test_load_branches:
  - "MET"
  - "Jet_pt"

labels:
  type: "complex"
  value:
    "_label_": "target_value"
```

## 🎓 学习范式

### 有监督学习（默认）

```yaml
train:
  learning_paradigm: "supervised"
  task_type: "classification"
```

### 半监督学习

```yaml
train:
  learning_paradigm: "semi-supervised"
  task_type: "classification"
  paradigm_config:
    strategy: "self-training"        # 或 "consistency", "pseudo-labeling"
    unsupervised_weight: 0.1
    confidence_threshold: 0.9
```

**标签约定**：
- 有标签样本：`label >= 0`
- 无标签样本：`label == -1`

### 无监督学习

```yaml
train:
  learning_paradigm: "unsupervised"
  paradigm_config:
    method: "autoencoder"            # 或 "vae", "contrastive"
    reconstruction_weight: 1.0
    kl_weight: 0.001
```

## 💾 模型保存与推理

### 模型保存

训练完成后会生成以下文件：

| 文件 | 说明 | 用途 |
|------|------|------|
| `best_model.pt` | 验证损失最小的模型 | ✅ 推荐用于推理 |
| `final_model.pt` | 最后一个 epoch 的模型 | 训练完成时的状态 |
| `model.pt` | `best_model.pt` 的副本 | ✅ 推荐用于推理 |

**保存机制**：
- 监控指标：`val_loss`（越小越好）
- 自动保存：当 `val_loss` 改善时自动保存最佳模型
- 保存格式：仅保存模型权重（`state_dict`），体积小，加载快

### 预测

#### 分类任务输出

```python
# ROOT 文件包含：
{
    "is_B": [True, False, ...],      # one-hot 标签
    "score_B": [0.95, 0.05, ...],    # 类别分数
    "prediction": [0, 1, ...],       # 预测类别
    "_label_": [0, 1, ...],          # 真实标签
    "met": [50.2, 45.8, ...],        # 观察变量
}
```

#### 回归任务输出

```python
{
    "prediction": [1.23, 2.45, ...], # 预测值
    "_label_": [1.25, 2.50, ...],    # 真实标签
    "met": [50.2, 45.8, ...],        # 观察变量
}
```

#### 使用新数据推理

```bash
# 分类模型
bamboohepml predict \
  -c configs/pipeline.yaml \
  -m outputs/model.pt \
  -o predictions.root \
  --probabilities

# 回归模型
bamboohepml predict \
  -c configs/pipeline.yaml \
  -m outputs/model.pt \
  -o predictions.root
```

**关键点**：
- 推理时不需要标签字段
- 只需在 `test_load_branches` 中包含特征字段
- 标签字段为可选，如果存在会被保存到输出文件

## 🐳 Docker 支持

### CPU 版本

```bash
docker build -t bamboohepml:latest .
docker run -v $(pwd)/configs:/app/configs -v $(pwd)/data:/app/data \
    bamboohepml:latest python -m bamboohepml.cli train -c configs/pipeline.yaml
```

### GPU 版本

```bash
docker build -f docker/Dockerfile.gpu -t bamboohepml:gpu .
docker run --gpus all -v $(pwd)/configs:/app/configs -v $(pwd)/data:/app/data \
    bamboohepml:gpu python -m bamboohepml.cli train -c configs/pipeline.yaml
```

### 推理服务

```bash
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs bamboohepml:latest \
    python -m bamboohepml.serve.fastapi_server serve_fastapi \
    --model-path outputs/model.pt --metadata-path outputs/metadata.json
```

## 🧪 开发与测试

### 代码风格

```bash
make style      # 格式化代码
make clean      # 清理临时文件
make test       # 运行测试
make test-cov   # 测试覆盖率
```

### Pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

### 测试

```bash
# 运行所有新架构测试
pytest tests/integration/test_new_architecture.py -v -s

# 运行特定测试
pytest tests/integration/test_new_architecture.py::test_only_event_features -v -s
```

**测试覆盖**：
- ✅ Event-only 特征
- ✅ Object-only 特征
- ✅ Event + Object 特征组合
- ✅ PipelineOrchestrator 自动维度推断
- ✅ 回归任务
- ✅ 真实 ROOT 文件测试

## 📚 文档

```bash
# 安装文档依赖
pip install mkdocs mkdocstrings[python]

# 本地预览
mkdocs serve

# 构建文档
mkdocs build
```

## 📄 许可证

MIT License

## 🙏 致谢

BambooHepMl 的开发受到了以下项目的启发和支持：

- **[weaver-core](https://github.com/colizz/weaver-core)**
- **[Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)**
