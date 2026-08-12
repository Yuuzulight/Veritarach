from fastapi.testclient import TestClient

from veritarach.service.app import app

client = TestClient(app)


def test_health_returns_200_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_valid_body_returns_501_model_not_loaded():
    response = client.post("/predict", json={"text": "some text to classify"})

    assert response.status_code == 501
    assert response.json() == {"detail": "model not loaded"}


def test_predict_with_missing_text_field_returns_422():
    response = client.post("/predict", json={})

    assert response.status_code == 422
