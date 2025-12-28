# BambooHepMl

一个面向高能物理的机器学习框架，结合了 weaver-core 的强大特征工程能力和 Made-With-ML 的现代 ML 工程实践。

## 设计理念

- **配置驱动**：所有特征工程通过 YAML 配置完成，无需硬编码
- **模块化设计**：清晰的模块职责，易于扩展和维护
- **生产就绪**：完整的 ML pipeline（data → model → train → eval → export → serve）
- **高能物理优化**：专为 HEP 数据格式和任务设计

## 核心特性

### 1. 数据与特征系统（借鉴 weaver-core）
- Config-driven 特征定义
- expr 表达式生成新变量
- 自动变量计算、标准化、裁剪、padding
- 支持 sequence / mask / transformer 输入
- 用户通过 YAML config 即可完成特征工程

### 2. ML Pipeline（借鉴 Made-With-ML）
- 清晰的模块职责
- 完整 ML pipeline（data → model → train → eval → export → serve）
- 支持测试、CI/CD、MLflow、TensorBoard、Ray、FastAPI
- 从实验到生产的现代 ML 系统

### 3. 调度系统
- 本地执行（local）
- SLURM 集群提交（sbatch）

### 4. 任务支持
- 分类 / 回归 / 多任务
- 监督 / 半监督 / 无监督
- 微调（finetune）与预训练模型

### 5. 工具集成
- CLI 子系统（train / predict / export / inspect）
- ONNX 导出
- TensorBoard + MLflow 监控
- FastAPI + Ray Serve 推理服务

## 项目结构

```
BambooHepMl/
├── bamboohepml/          # 主包
│   ├── data/             # 数据与特征系统
│   ├── models/           # 模型定义
│   ├── engine/           # 训练引擎
│   ├── tasks/            # 任务子系统
│   ├── pipeline/         # Pipeline 编排
│   ├── scheduler/        # 调度系统
│   ├── serve/            # 服务部署
│   ├── experiment/       # 实验跟踪
│   └── cli.py            # CLI 入口
├── tests/                # 测试
├── configs/              # 配置示例
├── docs/                 # 文档（mkdocs）
└── README.md
```

## 快速开始

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
bamboohepml export -c configs/pipeline.yaml -m outputs/model.pt -o model.onnx

# 启动服务
bamboohepml serve fastapi -m outputs/model.pt -c configs/pipeline.yaml

# SLURM 提交
bamboohepml train -c configs/pipeline.yaml --scheduler slurm
```

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

## 许可证

MIT License
