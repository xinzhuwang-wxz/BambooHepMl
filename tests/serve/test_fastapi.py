"""
FastAPI 服务测试
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from bamboohepml.serve import create_app


@pytest.fixture
def sample_model_path(temp_dir, sample_model):
    """保存示例模型。"""
    model_path = temp_dir / "test_model.pt"
    torch.save(sample_model.state_dict(), model_path)
    return str(model_path)


@pytest.fixture
def fastapi_app(sample_model_path):
    """创建 FastAPI 应用。"""
    app = create_app(
        model_path=sample_model_path,
        model_name='mlp_classifier',
        model_params={'input_dim': 10, 'hidden_dims': [64, 32], 'num_classes': 2},
    )
    return app


@pytest.fixture
def client(fastapi_app):
    """创建测试客户端。"""
    return TestClient(fastapi_app)


def test_health_check(client):
    """测试健康检查。"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_info" in data


def test_predict(client, sample_features):
    """测试预测。"""
    features_list = sample_features.tolist()
    response = client.post(
        "/predict",
        json={
            "features": features_list,
            "return_probabilities": False,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(features_list)


def test_predict_with_probabilities(client, sample_features):
    """测试带概率的预测。"""
    features_list = sample_features.tolist()
    response = client.post(
        "/predict",
        json={
            "features": features_list,
            "return_probabilities": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert len(data["predictions"]) == len(features_list)
    assert len(data["probabilities"]) == len(features_list)


def test_batch_predict(client, sample_features):
    """测试批量预测。"""
    features_list = sample_features.tolist()
    samples = [{"features": feat} for feat in features_list]
    
    response = client.post(
        "/predict/batch",
        json={
            "samples": samples,
            "return_probabilities": False,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(samples)

