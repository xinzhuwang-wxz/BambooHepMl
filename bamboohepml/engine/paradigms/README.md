# 学习范式模块

## 概述

学习范式模块提供了统一的学习范式接口，支持：
- **有监督学习** (Supervised)
- **半监督学习** (Semi-supervised) - 支持多种策略
- **无监督学习** (Unsupervised) - 支持多种方法

## 架构设计

### 基类：LearningParadigm

所有学习范式都继承自 `LearningParadigm` 基类，实现以下接口：

```python
class LearningParadigm(ABC):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor | None,
        outputs: torch.Tensor,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """计算损失"""
        pass

    def prepare_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """准备批次数据"""
        pass
```

## 使用方式

### 1. 有监督学习（默认）

```python
from bamboohepml.engine import Trainer
from bamboohepml.engine.paradigms import SupervisedParadigm

# 方式 1: 自动创建（默认）
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="classification",  # 或 "regression"
    learning_paradigm="supervised",  # 可选，默认值
)

# 方式 2: 显式创建 paradigm
paradigm = SupervisedParadigm(config={"task_type": "classification"})
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="classification",
    learning_paradigm=paradigm,
)
```

### 2. 半监督学习

```python
from bamboohepml.engine import Trainer

# Self-training 策略（默认）
trainer = Trainer(
    model=model,
    train_loader=train_loader,  # 数据中可以包含无标签样本（label=-1）
    task_type="classification",
    learning_paradigm="semi-supervised",
    paradigm_config={
        "strategy": "self-training",  # 或 "consistency", "pseudo-labeling"
        "unsupervised_weight": 0.1,  # 无监督损失权重
    },
)

# Consistency regularization 策略
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="classification",
    learning_paradigm="semi-supervised",
    paradigm_config={
        "strategy": "consistency",
        "unsupervised_weight": 0.5,
    },
)

# Pseudo-labeling 策略
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="classification",
    learning_paradigm="semi-supervised",
    paradigm_config={
        "strategy": "pseudo-labeling",
        "confidence_threshold": 0.9,  # 只对高置信度样本使用伪标签
        "unsupervised_weight": 0.1,
    },
)
```

### 3. 无监督学习

```python
from bamboohepml.engine import Trainer

# Autoencoder（默认）
trainer = Trainer(
    model=autoencoder_model,  # 应该是自编码器模型
    train_loader=train_loader,
    task_type="classification",  # 用于评估器，不影响训练
    learning_paradigm="unsupervised",
    paradigm_config={
        "method": "autoencoder",
        "reconstruction_weight": 1.0,
    },
)

# Variational Autoencoder (VAE)
trainer = Trainer(
    model=vae_model,  # 需要支持 get_latent_params() 方法
    train_loader=train_loader,
    learning_paradigm="unsupervised",
    paradigm_config={
        "method": "vae",
        "reconstruction_weight": 1.0,
        "kl_weight": 0.001,  # KL 散度权重
    },
)

# Contrastive Learning
trainer = Trainer(
    model=contrastive_model,
    train_loader=train_loader,
    learning_paradigm="unsupervised",
    paradigm_config={
        "method": "contrastive",
    },
)
```

## 半监督学习策略详解

### Self-training
- 使用模型预测的 argmax 作为伪标签
- 简单有效，适合大多数场景
- 配置：`{"strategy": "self-training", "unsupervised_weight": 0.1}`

### Consistency Regularization
- 对无标签数据添加噪声，要求输出一致
- 适合需要数据增强的场景
- 配置：`{"strategy": "consistency", "unsupervised_weight": 0.5}`

### Pseudo-labeling
- 只对高置信度的预测使用伪标签
- 更保守，适合噪声敏感的场景
- 配置：`{"strategy": "pseudo-labeling", "confidence_threshold": 0.9}`

## 无监督学习方法详解

### Autoencoder
- 标准的自编码器重构损失
- 适合特征学习
- 配置：`{"method": "autoencoder", "reconstruction_weight": 1.0}`

### Variational Autoencoder (VAE)
- 重构损失 + KL 散度
- 需要模型支持 `get_latent_params()` 方法返回 (mu, logvar)
- 配置：`{"method": "vae", "reconstruction_weight": 1.0, "kl_weight": 0.001}`

### Contrastive Learning
- SimCLR 风格的对比学习
- 需要模型输出特征表示
- 配置：`{"method": "contrastive"}`

## 数据格式要求

### 有监督学习
- 标准格式：`{"event": features, "_label_": labels}`
- labels 必需

### 半监督学习
- **标签约定（重要）**：
  - 有标签样本：`label >= 0`（例如：0, 1, 2, ... 用于分类；实际值用于回归）
  - 无标签样本：`label == -1`
  - 这是标准的半监督学习约定，与 scikit-learn 的 LabelSpreading 等保持一致
- 支持混合数据：batch 中可以同时包含有标签和无标签样本
- 示例：`{"event": features, "_label_": torch.tensor([0, 1, -1, 0, -1])}`
  - 前两个样本（label 0 和 1）和最后一个样本（label 0）是有标签的
  - 中间两个样本（label -1）是无标签的

### 无监督学习
- 不需要 labels：`{"event": features}` 或 `{"event": features, "_label_": None}`

## 向后兼容

为了保持向后兼容，旧的代码仍然可以工作：

```python
# 旧方式（仍然支持，但会发出警告）
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="supervised",  # 旧参数
)

# 新方式（推荐）
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    task_type="classification",  # 用于评估器
    learning_paradigm="supervised",  # 明确指定学习范式
)
```

## 扩展

要添加新的学习范式，继承 `LearningParadigm` 并实现必要的方法：

```python
from bamboohepml.engine.paradigms import LearningParadigm

class MyParadigm(LearningParadigm):
    def compute_loss(self, model, inputs, labels, outputs, loss_fn=None):
        # 实现损失计算逻辑
        return loss
```
