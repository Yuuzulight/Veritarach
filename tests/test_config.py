from pathlib import Path

import pytest
from pydantic import ValidationError

from veritarach.config import Settings


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # _env_file=None: this repo has a real .env with real keys, so the default env_file
    # lookup would mask a missing var. Disabling it isolates the test to process env only.
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults_applied_when_only_keys_are_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.data_dir == Path("data")
    assert settings.generation_budget_usd == 10.0
    assert settings.claude_sample_count == 75
    assert settings.gpt4o_sample_count == 75
    assert settings.gemini_sample_count == 150
    assert settings.paired_ratio == 0.7
    assert settings.claude_model == "claude-sonnet-5"
    assert settings.openai_model == "gpt-4o"
    assert settings.gemini_model == "gemini-flash-latest"
    assert settings.hc3_sample_limit is None
    assert settings.wikipedia_sample_count == 500


def test_data_dir_accepts_path_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert settings.data_dir == tmp_path


def test_test_settings_fixture_uses_fake_keys(test_settings):
    assert test_settings.anthropic_api_key == "test-key"
    assert test_settings.openai_api_key == "test-key"
    assert test_settings.gemini_api_key == "test-key"
