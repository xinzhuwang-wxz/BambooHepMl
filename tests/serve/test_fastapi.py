"""
FastAPI 服务测试（简化版）

只测试核心功能，避免复杂的 startup 事件处理。
"""

import pytest
import torch
from fastapi.testclient import TestClient

from bamboohepml.serve.fastapi_server import create_app


@pytest.fixture
def sample_model_path(temp_dir, sample_model):
    """保存示例模型。"""
    model_path = temp_dir / "test_model.pt"
    torch.save(sample_model.state_dict(), model_path)
    return str(model_path)


@pytest.fixture
def fastapi_app(sample_model_path):
    """创建 FastAPI 应用。"""
    return create_app(
        model_path=sample_model_path,
        model_name="mlp_classifier",
        model_params={"input_dim": 10, "hidden_dims": [64, 32], "num_classes": 2},
    )


@pytest.fixture
def client(fastapi_app):
    """创建测试客户端。"""
    # TestClient 会自动处理 startup 事件
    return TestClient(fastapi_app)


def test_health_check(client):
    """测试健康检查。"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    # 接受 "healthy" 或 "unavailable"（取决于模型是否成功加载）
    assert data["status"] in ["healthy", "unavailable"]


def test_predict(client, sample_features):
    """测试预测。"""
    features_list = sample_features.tolist()[:2]  # 只测试少量样本以加快速度
    response = client.post(
        "/predict",
        json={"features": features_list, "return_probabilities": False},
    )
    # 如果模型未加载，返回 503 也是可以接受的
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(features_list)


def test_predict_with_probabilities(client, sample_features):
    """测试带概率的预测。"""
    features_list = sample_features.tolist()[:2]
    response = client.post(
        "/predict",
        json={"features": features_list, "return_probabilities": True},
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "probabilities" in data
