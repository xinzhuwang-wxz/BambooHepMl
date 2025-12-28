"""
Pipeline Orchestrator

统一入口，协调整个 ML pipeline：
- 加载配置（data / feature / model / train）
- 构建 dataset
- 构建 model
- 调用 trainer
"""

from .orchestrator import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"]
