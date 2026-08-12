from unittest.mock import MagicMock

import torch
from fastapi.testclient import TestClient

import veritarach.service.app as app_module
from veritarach.service.app import app

client = TestClient(app)


def test_health_returns_200_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_missing_text_field_returns_422():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_returns_501_when_no_checkpoint_exists(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_tokenizer", None)
    monkeypatch.setattr(app_module, "_load_model", lambda: None)  # simulates no checkpoint found

    response = client.post("/predict", json={"text": "some text to classify"})

    assert response.status_code == 501
    assert response.json() == {"detail": "model not loaded"}


def test_load_model_leaves_model_none_when_checkpoint_dir_missing(monkeypatch, test_settings):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_tokenizer", None)
    # test_settings.data_dir is a fresh tmp_path -- no model/final subdirectory under it.
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    app_module._load_model()

    assert app_module._model is None


def test_predict_returns_real_prediction_when_model_is_loaded(monkeypatch, test_settings):
    mock_tokenizer = MagicMock(
        return_value={"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    )
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.1, 5.0]])  # strongly favors class 1 -> ai_generated
    mock_model.return_value = mock_output

    monkeypatch.setattr(app_module, "_model", mock_model)
    monkeypatch.setattr(app_module, "_tokenizer", mock_tokenizer)
    monkeypatch.setattr(app_module, "_load_model", lambda: None)  # "already loaded", no-op
    # predict() also reads get_settings().training_max_length -- must be mocked too, or this
    # only passes locally where a real .env happens to exist, and fails in CI (no .env there,
    # correctly gitignored). Same CI-parity trap as issue #2's config tests.
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    response = client.post("/predict", json={"text": "some AI-sounding text"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "ai_generated"
    assert 0.9 < body["confidence"] <= 1.0


def test_predict_returns_human_written_when_that_class_wins(monkeypatch, test_settings):
    mock_tokenizer = MagicMock(
        return_value={"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    )
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[5.0, 0.1]])  # strongly favors class 0 -> human_written
    mock_model.return_value = mock_output

    monkeypatch.setattr(app_module, "_model", mock_model)
    monkeypatch.setattr(app_module, "_tokenizer", mock_tokenizer)
    monkeypatch.setattr(app_module, "_load_model", lambda: None)
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    response = client.post("/predict", json={"text": "some human-sounding text"})

    assert response.status_code == 200
    assert response.json()["label"] == "human_written"
