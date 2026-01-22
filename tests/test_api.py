import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import project99.api as api


@pytest.fixture(autouse=True)
def mock_model():
    # Create a fake model object
    fake_model = MagicMock()
    fake_model.predict.return_value = [1]
    fake_model.predict_proba.return_value = [[0.1, 0.9]]
    api.model = fake_model
    yield fake_model


client = TestClient(api.app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Project 99 API is running"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] in ["healthy", "unhealthy"]


def test_model_info_endpoint():
    response = client.get("/model/info")

    if response.status_code == 200:
        data = response.json()
        assert data["model_type"] == "XGBoost"
        assert "feature_count" in data
        assert "feature_names" in data
    else:
        assert response.status_code == 503


def test_predict_endpoint_valid_input():
    valid_input = {
        "SetNo": 1,
        "GameNo": 3,
        "PointNumber": 15,
        "PointServer": 1,
        "ServeIndicator": 1,
        "P1GamesWon": 2,
        "P1SetsWon": 0,
        "P1Score": "30",
        "P1PointsWon": 12,
        "P1Momentum": 2,
        "P2GamesWon": 1,
        "P2SetsWon": 0,
        "P2Score": "15",
        "P2PointsWon": 10,
        "P2Momentum": -1,
    }

    # Vertex AI format
    request_body = {"instances": [valid_input]}
    response = client.post("/predict", json=request_body)

    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert "prediction" in pred
        assert "probability" in pred
        assert pred["prediction"] in [0, 1]
        assert 0 <= pred["probability"] <= 1
    else:
        assert response.status_code == 503


def test_predict_endpoint_invalid_input():
    invalid_input = {
        "SetNo": "invalid",
        "GameNo": 3,
        "PointNumber": 15,
        "PointServer": 1,
        "ServeIndicator": 1,
        "P1GamesWon": 2,
        "P1SetsWon": 0,
        "P1Score": "30",
        "P1PointsWon": 12,
        "P1Momentum": 2,
        "P2GamesWon": 1,
        "P2SetsWon": 0,
        "P2Score": "15",
        "P2PointsWon": 10,
        "P2Momentum": -1,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    incomplete_input = {
        "SetNo": 1,
        "GameNo": 3,
    }

    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422


def test_batch_predict_valid_csv():
    header = (
        "PointServer,P1Score,P2Score,P1GamesWon,P2GamesWon,P1PointsWon,"
        "P2PointsWon,P1SetsWon,P2SetsWon,SetNo,GameNo,PointNumber,"
        "ServeIndicator,P1Momentum,P2Momentum"
    )
    csv_content = f"""{header}
1,30,15,2,1,12,10,0,0,1,3,15,1,2,-1
2,40,30,3,2,25,20,1,0,2,6,45,1,5,3"""

    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    response = client.post("/predict/batch", files=files)

    if response.status_code == 200:
        data = response.json()
        assert "total_predictions" in data
        assert "csv_with_predictions" in data
        assert data["total_predictions"] == 2
    else:
        assert response.status_code == 503


def test_batch_predict_invalid_file_type():
    files = {"file": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "File must be a CSV" in response.json()["detail"]


def test_batch_predict_missing_columns():
    """Test batch prediction with CSV missing required columns."""
    csv_content = """SetNo,GameNo
1,3
2,6"""

    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]
