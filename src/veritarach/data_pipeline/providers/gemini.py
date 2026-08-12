import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from veritarach.config import Settings

from .base import ProviderError

MAX_ATTEMPTS = 3


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        self._request_delay_seconds = settings.gemini_request_delay_seconds

    def generate(self, prompt: str) -> str:
        """Calls the provider SDK. On a rate-limit error, retries up to MAX_ATTEMPTS
        times with manual exponential backoff. Raises ProviderError if every attempt
        fails. Paces itself afterward (success or failure) so consecutive calls from
        the orchestrator don't fire back-to-back and blow through the free tier's
        requests-per-minute cap -- see gemini_request_delay_seconds in config.py."""
        try:
            return self._generate_with_retry(prompt)
        finally:
            time.sleep(self._request_delay_seconds)

    def _generate_with_retry(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except ResourceExhausted as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2**attempt)
        raise ProviderError(f"{self.name}: exhausted {MAX_ATTEMPTS} attempts") from last_error
