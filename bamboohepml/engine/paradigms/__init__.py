"""
学习范式模块

支持不同的学习范式：
- Supervised: 有监督学习
- Semi-supervised: 半监督学习（支持多种策略）
- Unsupervised: 无监督学习（支持多种方法）
"""

from .base import LearningParadigm
from .semi_supervised import SemiSupervisedParadigm
from .supervised import SupervisedParadigm
from .unsupervised import UnsupervisedParadigm

__all__ = [
    "LearningParadigm",
    "SupervisedParadigm",
    "SemiSupervisedParadigm",
    "UnsupervisedParadigm",
]
