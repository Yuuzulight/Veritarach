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

    def generate(self, prompt: str) -> str:
        """Calls the provider SDK. On a rate-limit error, retries up to MAX_ATTEMPTS
        times with manual exponential backoff. Raises ProviderError if every attempt
        fails."""
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
