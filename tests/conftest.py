"""
pytest 配置和共享 fixtures
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径（以便 conftest 可以导入项目模块）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from bamboohepml.engine import Predictor  # noqa: E402
from bamboohepml.models import get_model  # noqa: E402


@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_model():
    """创建示例模型（使用新架构，只有 event 特征）。"""
    model = get_model(
        "mlp_classifier",
        event_input_dim=10,
        object_input_dim=None,
        embed_dim=64,
        hidden_dims=[64, 32],
        num_classes=2,
    )
    return model


@pytest.fixture
def sample_predictor(sample_model):
    """创建示例预测器。"""
    return Predictor(sample_model)


@pytest.fixture
def sample_features():
    """创建示例特征。"""
    return np.random.randn(32, 10).astype(np.float32)


@pytest.fixture
def sample_labels():
    """创建示例标签。"""
    return np.random.randint(0, 2, size=32)


@pytest.fixture
def sample_data(sample_features, sample_labels):
    """创建示例数据。"""
    from torch.utils.data import DataLoader, TensorDataset

    features_tensor = torch.tensor(sample_features)
    labels_tensor = torch.tensor(sample_labels)
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=8)
    return dataloader
