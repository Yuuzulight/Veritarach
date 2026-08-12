from typing import Protocol


class ProviderError(Exception):
    """Raised when a provider exhausts its retries without a successful response."""


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...
