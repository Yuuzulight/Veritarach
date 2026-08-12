import time

import anthropic

from veritarach.config import Settings

from .base import ProviderError

MAX_ATTEMPTS = 3


class ClaudeProvider:
    name = "claude"

    def __init__(self, settings: Settings):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    def generate(self, prompt: str) -> str:
        """Calls the provider SDK. On a rate-limit error, retries up to MAX_ATTEMPTS
        times with manual exponential backoff. Raises ProviderError if every attempt
        fails."""
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2**attempt)
        raise ProviderError(f"{self.name}: exhausted {MAX_ATTEMPTS} attempts") from last_error
