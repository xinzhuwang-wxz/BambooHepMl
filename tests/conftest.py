"""
pytest 配置和共享 fixtures
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from bamboohepml.engine import Predictor
from bamboohepml.models import get_model


@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_model():
    """创建示例模型。"""
    model = get_model("mlp_classifier", input_dim=10, hidden_dims=[64, 32], num_classes=2)
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
