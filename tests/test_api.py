"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "xgboost_model" in data["loaded_artifacts"]


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["calibrated_threshold"] > 0.0
    assert len(data["feature_names"]) > 20


def test_predict_single_endpoint(client):
    payload = {
        "transaction_id": "TX_TEST_001",
        "Time": 5000.0,
        "Amount": 125.50,
        "V1": -1.2,
        "V4": 1.5,
        "V14": -2.3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["action"] in ["ALLOW", "FLAG_CHALLENGE", "BLOCK_AND_INVESTIGATE"]
    assert len(data["primary_reasons"]) > 0
    assert data["latency_ms"] >= 0.0


def test_predict_invalid_amount_rejected(client):
    payload = {
        "transaction_id": "TX_INVALID",
        "Time": 5000.0,
        "Amount": -100.0,  # Negative amount must be rejected by Pydantic validation
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_batch_predict_endpoint(client):
    payload = {
        "transactions": [
            {"transaction_id": "TX_BATCH_1", "Time": 1000.0, "Amount": 15.0, "V4": -0.5},
            {"transaction_id": "TX_BATCH_2", "Time": 2000.0, "Amount": 2500.0, "V4": 3.5, "V14": -5.0},
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert data["processing_time_ms"] >= 0.0
