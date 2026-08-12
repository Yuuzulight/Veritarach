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
    gemini_model: str = "gemini-2.5-flash"  # gemini-2.0-flash was shut down 2026-06-01; this is
    # its confirmed current replacement, with the same free tier (2026-08-12)

    hc3_sample_limit: int | None = None  # cap for fast local dev; None = full dataset
    wikipedia_sample_count: int = 500  # oversample; build_dataset trims to match generated-sample count


def get_settings() -> Settings:
    return Settings()
