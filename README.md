# BambooHepMl

一个面向高能物理的机器学习框架，结合了 weaver-core 的强大特征工程能力和 Made-With-ML 的现代 ML 工程实践。

## 设计理念

- **配置驱动**：所有特征工程通过 YAML 配置完成，无需硬编码
- **模块化设计**：清晰的模块职责，易于扩展和维护
- **生产就绪**：完整的 ML pipeline（data → model → train → eval → export → serve）
- **高能物理优化**：专为 HEP 数据格式和任务设计

## 核心架构

### FeatureGraph：唯一特征源

FeatureGraph 是框架的核心，作为**唯一可信的特征事实源（Single Source of Truth）**：

- **Config-driven 特征定义**：所有特征在 YAML 中定义，支持表达式（expr）、依赖解析、DAG 管理
- **自动预处理**：normalize、clip、padding、mask 完全由配置控制
- **状态持久化**：Normalizer 参数自动拟合和保存，确保训练/验证/测试一致性
- **输出规范**：自动生成模型输入格式（event/object/mask），与模型维度对齐

### Metadata-Driven 架构

训练/导出/服务完全解耦：

- **训练阶段**：自动保存 `metadata.json`（包含 feature_spec、model_config、normalizer 参数等）
- **导出阶段**：仅依赖模型权重和 metadata，无需重现训练 Dataset
- **服务阶段**：FastAPI/ONNX 服务仅依赖模型和 metadata，完全独立于训练流程

### LearningParadigm 系统

统一的学习范式接口，支持：

- **监督学习**：标准分类/回归任务
- **半监督学习**：self-training、consistency regularization、pseudo-labeling
- **无监督学习**：autoencoder、VAE、contrastive learning

## 核心特性

### 1. 数据与特征系统（weaver-core 级别）

- ✅ Config-driven 特征定义（YAML）
- ✅ Expression 表达式引擎（支持 numpy/awkward 运算）
- ✅ 自动依赖解析和 DAG 构建
- ✅ 完整的预处理：normalize / clip / padding / mask
- ✅ 支持 sequence / transformer 输入
- ✅ 零硬编码：所有特征工程在配置中完成

### 2. ML Pipeline（Made-With-ML 级别）

- ✅ 清晰的模块边界（config / data / model / engine / tasks / serve）
- ✅ 完整 ML pipeline：data → feature → model → train → eval → export → serve
- ✅ Metadata-driven 架构（训练/导出/服务解耦）
- ✅ 支持测试、CI/CD、Docker
- ✅ MLflow / TensorBoard 实验跟踪
- ✅ 面向实验 + 生产

### 3. 调度系统

- ✅ 本地执行（local）
- ✅ SLURM 集群提交（sbatch）

### 4. 任务支持

- ✅ 分类 / 回归 / 多任务
- ✅ 监督 / 半监督 / 无监督
- ✅ 微调（finetune）与预训练模型

### 5. 工具集成

- ✅ CLI 子系统（train / predict / export / inspect）
- ✅ ONNX 导出
- ✅ TensorBoard + MLflow 监控
- ✅ FastAPI + Ray Serve 推理服务

## 项目结构

```
BambooHepMl/
├── bamboohepml/          # 主包
│   ├── data/             # 数据与特征系统
│   │   ├── features/     # FeatureGraph、ExpressionEngine、FeatureProcessor
│   │   ├── sources/      # DataSource（ROOT/Parquet/HDF5）
│   │   └── dataset.py    # HEPDataset
│   ├── models/           # 模型定义
│   ├── engine/           # 训练引擎
│   │   ├── trainer.py    # Trainer
│   │   ├── paradigms/    # LearningParadigm 系统
│   │   └── evaluator.py  # Evaluator
│   ├── tasks/            # 任务子系统
│   │   ├── train.py      # train_task（LocalBackend / RayBackend）
│   │   ├── predict.py    # predict_task
│   │   └── export.py     # export_task
│   ├── pipeline/         # Pipeline 编排
│   │   ├── orchestrator.py  # PipelineOrchestrator
│   │   └── state.py         # PipelineState
│   ├── scheduler/        # 调度系统
│   ├── serve/            # 服务部署
│   ├── experiment/       # 实验跟踪
│   └── cli.py            # CLI 入口
├── tests/                # 测试
│   └── integration/      # 集成测试（完整 pipeline）
├── configs/              # 配置示例
└── README.md
```

## 快速开始

### 配置文件 Schema

BambooHepMl 使用 YAML 进行全流程配置。主要配置文件包括 `pipeline.yaml` 以及其引用的 `data.yaml` 和 `features.yaml`。

#### Pipeline 配置 (`pipeline.yaml`)

```yaml
data:
  config_path: "configs/data.yaml"  # 数据集配置文件路径
  source_path: "data/train.root"    # 数据源路径
  treename: "Events"                # ROOT Tree 名称 (可选)
  load_range: [0, 10000]            # 加载范围 (可选)
  val_split: 0.1                    # 验证集比例 (可选，0-1)

features:
  config_path: "configs/features.yaml" # 特征配置文件路径

model:
  name: "ParticleTransformer"       # 模型名称 (注册的模型名)
  params:                           # 模型参数
    num_classes: 2
    hidden_dim: 128

train:
  num_epochs: 20
  batch_size: 128
  learning_rate: 0.001
  task_type: "classification"       # classification, regression
  learning_paradigm: "supervised"   # supervised, semi-supervised, unsupervised
  paradigm_config:                  # 范式特定配置
    loss_weight: 0.5
```

#### 特征配置 (`features.yaml`)

```yaml
features:
  event_level:                      # Event 级别特征
    - name: "met"
      source: "MET"                 # 原始字段名
      dtype: "float32"
      normalize:                    # 标准化配置
        method: "auto"              # auto, manual, none

    - name: "ht"
      expr: "sum(Jet_pt)"           # 表达式特征 (支持 numpy/awkward 运算)
      dtype: "float32"
      normalize:
        method: "manual"
        center: 100.0
        scale: 0.01

  object_level:                     # Object 级别特征 (变长序列)
    - name: "jet_pt"
      source: "Jet_pt"
      dtype: "float32"
      normalize:
        method: "auto"
      clip:                         # 裁剪配置
        min: 0.0
        max: 500.0
      padding:                      # Padding 配置
        max_length: 128
        mode: "constant"
        value: 0.0
```

#### 数据配置 (`data.yaml`)

```yaml
train_load_branches:                # 训练时加载的分支列表
  - "MET"
  - "Jet_pt"
  - "Jet_eta"
  - "label"

test_load_branches:                 # 测试时加载的分支列表
  - "MET"
  - "Jet_pt"
  - "Jet_eta"

label: "label"                      # 标签字段名
weight: "weight"                    # 权重字段名 (可选)
```

### 半监督学习约定

在半监督学习（`learning_paradigm: "semi-supervised"`）中，数据标签遵循以下约定：

- **有标签数据**：标签为正常的类别索引（如 0, 1, 2...）或回归值（需非负）。
- **无标签数据**：标签值应设为 `-1`。

系统会自动识别无标签数据（`labels < 0`），并在计算有监督 Loss 时将其排除，同时这些数据将参与无监督 Loss（如一致性正则化、伪标签）的计算。

### 安装

```bash
pip install -e .
```

### 使用示例

```bash
# 训练模型
bamboohepml train -c configs/pipeline.yaml --experiment-name my_exp

# 预测
bamboohepml predict -c configs/pipeline.yaml -m outputs/model.pt -o predictions.json

# 导出 ONNX
bamboohepml export -m outputs/model.pt -o model.onnx --metadata-path outputs/metadata.json

# 启动服务
bamboohepml serve fastapi -m outputs/model.pt --metadata-path outputs/metadata.json

# SLURM 提交
bamboohepml train -c configs/pipeline.yaml --scheduler slurm
```

## 完整 Pipeline 流程

BambooHepMl 实现了完整的 ML pipeline，从数据加载到模型服务的全流程：

1. **Data**：从 ROOT/Parquet/HDF5 加载数据，支持 jagged array
2. **Feature**：FeatureGraph 从 YAML 配置构建特征，自动预处理和规范化
3. **Model**：根据配置创建模型，输入维度自动从 FeatureGraph 推断
4. **Train**：支持本地和 Ray 分布式训练，完整的学习范式支持
5. **Eval**：评估器自动计算任务相关的指标（accuracy、AUC、MSE 等）
6. **Export**：导出 ONNX 模型，仅依赖模型和 metadata
7. **Serve**：FastAPI/ONNX 服务，完全独立于训练流程

所有阶段通过 metadata 解耦，确保生产环境的一致性。

## 文档

使用 mkdocs 生成文档：

```bash
# 安装文档依赖
pip install mkdocs mkdocstrings[python]

# 本地预览
mkdocs serve

# 构建文档
mkdocs build
```

## 开发

### 代码风格

```bash
# 格式化代码
make style

# 清理临时文件
make clean

# 运行测试
make test

# 运行测试并生成覆盖率报告
make test-cov
```

### Pre-commit hooks

```bash
# 安装 pre-commit hooks
pre-commit install

# 手动运行
pre-commit run --all-files
```

### 运行完整 Pipeline 测试

```bash
# 运行完整 pipeline 集成测试
pytest tests/integration/test_full_pipeline.py -v

# 运行特定测试
pytest tests/integration/test_full_pipeline.py::test_full_pipeline_flow -v
```

## 架构亮点

### 1. FeatureGraph 作为唯一特征源

- 彻底消除了特征系统的"双重性"问题
- Config-driven，真正达到 DSL 级别
- 自动依赖解析、DAG 管理、循环检测
- 状态持久化（Normalizer 参数）

### 2. Metadata-Driven 架构

- 训练/导出/服务完全解耦
- 生产环境不需要重现训练时的 Dataset
- 符合 ML 工程最佳实践

### 3. LearningParadigm 系统

- 统一的学习范式接口，扩展性强
- 半监督和无监督学习完整实现
- 支持多种策略和方法

## 许可证

MIT License
