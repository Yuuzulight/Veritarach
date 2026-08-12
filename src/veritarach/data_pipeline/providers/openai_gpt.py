import time

import openai

from veritarach.config import Settings

from .base import ProviderError

MAX_ATTEMPTS = 3


class OpenAIProvider:
    name = "gpt4o"

    def __init__(self, settings: Settings):
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, prompt: str) -> str:
        """Calls the provider SDK. On a rate-limit or transient server error (5xx),
        retries up to MAX_ATTEMPTS times with manual exponential backoff. Raises
        ProviderError if every attempt fails."""
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except (openai.RateLimitError, openai.InternalServerError) as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2**attempt)
        raise ProviderError(f"{self.name}: exhausted {MAX_ATTEMPTS} attempts") from last_error
