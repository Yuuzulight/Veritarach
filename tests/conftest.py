import pytest

from veritarach.config import Settings


@pytest.fixture
def test_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return Settings(data_dir=tmp_path)
