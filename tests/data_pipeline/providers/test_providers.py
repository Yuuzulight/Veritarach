from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
import pytest
from google.api_core.exceptions import ResourceExhausted

from tests.conftest import FakeProvider
from veritarach.data_pipeline.providers.base import ProviderError
from veritarach.data_pipeline.providers.claude import ClaudeProvider
from veritarach.data_pipeline.providers.gemini import GeminiProvider
from veritarach.data_pipeline.providers.openai_gpt import OpenAIProvider


def _anthropic_rate_limit_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _openai_rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _anthropic_overloaded_error():
    # Mirrors the real 529 "Overloaded" error Anthropic's own docs say is retry-worthy --
    # confirmed as a real failure mode in production (2026-08-12), not hypothetical.
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(529, request=request)
    return anthropic.InternalServerError("overloaded", response=response, body=None)


def _openai_server_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return openai.InternalServerError("internal server error", response=response, body=None)


def _gemini_rate_limit_error():
    return ResourceExhausted("rate limited")


class TestClaudeProvider:
    @patch("veritarach.data_pipeline.providers.claude.anthropic.Anthropic")
    def test_generate_returns_response_text(self, mock_anthropic_cls, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello from claude")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = ClaudeProvider(test_settings).generate("say hi")

        assert result == "hello from claude"

    @patch("veritarach.data_pipeline.providers.claude.time.sleep")
    @patch("veritarach.data_pipeline.providers.claude.anthropic.Anthropic")
    def test_generate_retries_on_rate_limit_then_succeeds(self, mock_anthropic_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="succeeded on retry")]
        mock_client.messages.create.side_effect = [_anthropic_rate_limit_error(), mock_response]
        mock_anthropic_cls.return_value = mock_client

        result = ClaudeProvider(test_settings).generate("say hi")

        assert result == "succeeded on retry"
        assert mock_client.messages.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("veritarach.data_pipeline.providers.claude.time.sleep")
    @patch("veritarach.data_pipeline.providers.claude.anthropic.Anthropic")
    def test_generate_raises_provider_error_after_3_failures(self, mock_anthropic_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _anthropic_rate_limit_error()
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(ProviderError):
            ClaudeProvider(test_settings).generate("say hi")

        assert mock_client.messages.create.call_count == 3

    @patch("veritarach.data_pipeline.providers.claude.time.sleep")
    @patch("veritarach.data_pipeline.providers.claude.anthropic.Anthropic")
    def test_generate_retries_on_overloaded_error_then_succeeds(self, mock_anthropic_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="succeeded on retry")]
        mock_client.messages.create.side_effect = [_anthropic_overloaded_error(), mock_response]
        mock_anthropic_cls.return_value = mock_client

        result = ClaudeProvider(test_settings).generate("say hi")

        assert result == "succeeded on retry"
        assert mock_client.messages.create.call_count == 2

    @patch("veritarach.data_pipeline.providers.claude.anthropic.Anthropic")
    def test_generate_does_not_retry_non_rate_limit_errors(self, mock_anthropic_cls, test_settings):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ValueError("some other failure")
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(ValueError):
            ClaudeProvider(test_settings).generate("say hi")

        assert mock_client.messages.create.call_count == 1


class TestOpenAIProvider:
    @patch("veritarach.data_pipeline.providers.openai_gpt.openai.OpenAI")
    def test_generate_returns_response_text(self, mock_openai_cls, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello from gpt-4o"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        result = OpenAIProvider(test_settings).generate("say hi")

        assert result == "hello from gpt-4o"

    @patch("veritarach.data_pipeline.providers.openai_gpt.time.sleep")
    @patch("veritarach.data_pipeline.providers.openai_gpt.openai.OpenAI")
    def test_generate_retries_on_rate_limit_then_succeeds(self, mock_openai_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="succeeded on retry"))]
        mock_client.chat.completions.create.side_effect = [_openai_rate_limit_error(), mock_response]
        mock_openai_cls.return_value = mock_client

        result = OpenAIProvider(test_settings).generate("say hi")

        assert result == "succeeded on retry"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("veritarach.data_pipeline.providers.openai_gpt.time.sleep")
    @patch("veritarach.data_pipeline.providers.openai_gpt.openai.OpenAI")
    def test_generate_raises_provider_error_after_3_failures(self, mock_openai_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai_rate_limit_error()
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ProviderError):
            OpenAIProvider(test_settings).generate("say hi")

        assert mock_client.chat.completions.create.call_count == 3

    @patch("veritarach.data_pipeline.providers.openai_gpt.time.sleep")
    @patch("veritarach.data_pipeline.providers.openai_gpt.openai.OpenAI")
    def test_generate_retries_on_server_error_then_succeeds(self, mock_openai_cls, mock_sleep, test_settings):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="succeeded on retry"))]
        mock_client.chat.completions.create.side_effect = [_openai_server_error(), mock_response]
        mock_openai_cls.return_value = mock_client

        result = OpenAIProvider(test_settings).generate("say hi")

        assert result == "succeeded on retry"
        assert mock_client.chat.completions.create.call_count == 2


class TestGeminiProvider:
    @patch("veritarach.data_pipeline.providers.gemini.time.sleep")
    @patch("veritarach.data_pipeline.providers.gemini.genai.GenerativeModel")
    @patch("veritarach.data_pipeline.providers.gemini.genai.configure")
    def test_generate_returns_response_text(self, mock_configure, mock_model_cls, mock_sleep, test_settings):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="hello from gemini")
        mock_model_cls.return_value = mock_model

        result = GeminiProvider(test_settings).generate("say hi")

        assert result == "hello from gemini"
        # Pacing must fire even on a clean success -- that's the whole point of the fix.
        mock_sleep.assert_called_once_with(test_settings.gemini_request_delay_seconds)

    @patch("veritarach.data_pipeline.providers.gemini.time.sleep")
    @patch("veritarach.data_pipeline.providers.gemini.genai.GenerativeModel")
    @patch("veritarach.data_pipeline.providers.gemini.genai.configure")
    def test_generate_retries_on_rate_limit_then_succeeds(
        self, mock_configure, mock_model_cls, mock_sleep, test_settings
    ):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            _gemini_rate_limit_error(),
            MagicMock(text="succeeded on retry"),
        ]
        mock_model_cls.return_value = mock_model

        result = GeminiProvider(test_settings).generate("say hi")

        assert result == "succeeded on retry"
        assert mock_model.generate_content.call_count == 2
        # 1 backoff sleep (failed attempt) + 1 pacing sleep (after the call finishes) = 2.
        assert mock_sleep.call_count == 2

    @patch("veritarach.data_pipeline.providers.gemini.time.sleep")
    @patch("veritarach.data_pipeline.providers.gemini.genai.GenerativeModel")
    @patch("veritarach.data_pipeline.providers.gemini.genai.configure")
    def test_generate_raises_provider_error_after_3_failures(
        self, mock_configure, mock_model_cls, mock_sleep, test_settings
    ):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = _gemini_rate_limit_error()
        mock_model_cls.return_value = mock_model

        with pytest.raises(ProviderError):
            GeminiProvider(test_settings).generate("say hi")

        assert mock_model.generate_content.call_count == 3
        # Pacing must still fire on total failure -- otherwise the *next* sample in the
        # batch fires immediately into the same still-active rate limit, which is exactly
        # what happened in production before this fix (every sample failed identically
        # back-to-back once the quota was hit).
        mock_sleep.assert_called_with(test_settings.gemini_request_delay_seconds)


class TestFakeProvider:
    def test_matches_llm_provider_protocol_shape(self, fake_claude_provider, fake_gpt4o_provider, fake_gemini_provider):
        for provider in (fake_claude_provider, fake_gpt4o_provider, fake_gemini_provider):
            assert isinstance(provider.name, str)
            assert callable(provider.generate)

    def test_returns_configured_response(self, fake_claude_provider):
        assert fake_claude_provider.generate("anything") == "mock claude response"

    def test_can_simulate_exhaustion_after_n_calls(self):
        provider = FakeProvider(name="claude", responses=["ok", "ok"], fail_after=2)
        provider.generate("first")
        provider.generate("second")
        with pytest.raises(ProviderError):
            provider.generate("third")
