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
        """Calls the provider SDK. On a rate-limit or transient server error (5xx,
        including the "Overloaded" 529 Anthropic explicitly documents as retry-worthy),
        retries up to MAX_ATTEMPTS times with manual exponential backoff. Raises
        ProviderError if every attempt fails."""
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                # claude-sonnet-5 runs adaptive thinking by default even with no `thinking`
                # param set, so content[0] is sometimes a ThinkingBlock, not the text --
                # pick the actual text block instead of assuming position 0.
                return next(block.text for block in response.content if block.type == "text")
            except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2**attempt)
        raise ProviderError(f"{self.name}: exhausted {MAX_ATTEMPTS} attempts") from last_error
