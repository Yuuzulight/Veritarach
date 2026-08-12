from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    openai_api_key: str
    gemini_api_key: str

    data_dir: Path = Path("data")

    generation_budget_usd: float = 10.0
    claude_sample_count: int = 75
    gpt4o_sample_count: int = 75
    gemini_sample_count: int = 150
    paired_ratio: float = 0.7  # fraction of each provider's samples using the paired-HC3 strategy

    claude_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o"  # confirmed still current: $2.50/$10 per MTok in/out (2026-08-12)
    gemini_model: str = "gemini-flash-latest"  # gemini-2.0-flash was shut down 2026-06-01;
    # gemini-2.5-flash (its replacement) turned out to be listed but blocked for new API keys
    # ("no longer available to new users", confirmed via a real 404 on this exact key,
    # 2026-08-12) -- using Google's own "latest" alias instead of a pinned version avoids
    # hardcoding into the next one of these that turns up.

    hc3_sample_limit: int | None = None  # cap for fast local dev; None = full dataset
    wikipedia_sample_count: int = 500  # oversample; build_dataset trims to match generated-sample count

    gemini_request_delay_seconds: float = 4.0  # ~15 req/min pacing -- a conservative assumption,
    # not a verified number. Free-tier RPM is account-specific and only shown in the user's own
    # AI Studio dashboard (ai.google.dev/gemini-api/docs/rate-limits confirms this isn't publicly
    # documented). Firing requests back-to-back with no pacing burned through the real quota in
    # 18 calls (confirmed 2026-08-12); raise this if that still trips, or check the dashboard.

    base_model: str = "microsoft/deberta-v3-base"
    training_max_length: int = 256  # covers the bulk of HC3/generated text without excess padding
    training_num_epochs: int = 3
    training_batch_size: int = 16
    training_learning_rate: float = 2e-5


def get_settings() -> Settings:
    return Settings()
