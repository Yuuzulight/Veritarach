import pytest

from veritarach.config import Settings
from veritarach.data_pipeline.providers.base import ProviderError


@pytest.fixture
def test_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return Settings(data_dir=tmp_path)


class FakeProvider:
    """Test double implementing the LLMProvider protocol -- returns a canned response,
    or raises ProviderError on the Nth call if configured to simulate exhaustion."""

    def __init__(self, name: str, responses: list[str] | None = None, fail_after: int | None = None):
        self.name = name
        self._responses = responses or []
        self._fail_after = fail_after
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        self._call_count += 1
        if self._fail_after is not None and self._call_count > self._fail_after:
            raise ProviderError(f"{self.name}: simulated exhaustion")
        index = min(self._call_count - 1, len(self._responses) - 1)
        return self._responses[index]


@pytest.fixture
def fake_claude_provider():
    return FakeProvider(name="claude", responses=["mock claude response"])


@pytest.fixture
def fake_gpt4o_provider():
    return FakeProvider(name="gpt4o", responses=["mock gpt-4o response"])


@pytest.fixture
def fake_gemini_provider():
    return FakeProvider(name="gemini", responses=["mock gemini response"])
