"""Runs the full data pipeline end-to-end: fetch HC3 + Wikipedia, generate self-samples
from each provider, then assemble the final balanced train/val/test dataset.

Usage: uv run python scripts/run_pipeline.py [provider ...]
  With no arguments, targets all three providers (claude, gpt4o, gemini).
  With arguments, only those providers generate new samples this run -- e.g.
  `uv run python scripts/run_pipeline.py claude` runs just Claude, skipping the others
  entirely (useful when another provider is still rate/quota-limited). The final dataset
  is always rebuilt from every *_samples.jsonl on disk, not just the ones this run
  targeted, so it reflects everything generated so far regardless.
"""

import json
import sys
from pathlib import Path

from veritarach.config import get_settings
from veritarach.data_pipeline.build_dataset import build_dataset
from veritarach.data_pipeline.fetch_hc3 import fetch_hc3
from veritarach.data_pipeline.fetch_wikipedia import fetch_wikipedia
from veritarach.data_pipeline.generate_llm_samples import GeneratedSample, generate_llm_samples
from veritarach.data_pipeline.providers.claude import ClaudeProvider
from veritarach.data_pipeline.providers.gemini import GeminiProvider
from veritarach.data_pipeline.providers.openai_gpt import OpenAIProvider


def _load_jsonl(path: Path, cls):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [cls(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw"
    generated_dir = settings.data_dir / "generated"

    requested_providers = set(sys.argv[1:]) or {"claude", "gpt4o", "gemini"}

    print(f"==> Fetching HC3 (limit={settings.hc3_sample_limit})...")
    hc3_pairs = fetch_hc3(raw_dir, limit=settings.hc3_sample_limit)
    print(f"    {len(hc3_pairs)} HC3 pairs")

    print(f"==> Fetching Wikipedia ({settings.wikipedia_sample_count} target samples)...")
    wikipedia_samples = fetch_wikipedia(raw_dir, sample_count=settings.wikipedia_sample_count)
    print(f"    {len(wikipedia_samples)} Wikipedia samples")

    # Gemini is free tier -- no budget pressure. Claude and GPT-4o split the configured
    # budget evenly; at these sample counts actual spend is expected to land well under
    # $1 total per provider, so this split is generous headroom, not a tight constraint.
    provider_plan = [
        (ClaudeProvider(settings), settings.claude_sample_count, settings.generation_budget_usd / 2),
        (OpenAIProvider(settings), settings.gpt4o_sample_count, settings.generation_budget_usd / 2),
        (GeminiProvider(settings), settings.gemini_sample_count, 1000.0),
    ]
    provider_plan = [p for p in provider_plan if p[0].name in requested_providers]

    for provider, sample_count, budget in provider_plan:
        print(f"==> Generating {sample_count} samples from {provider.name} (budget=${budget:.2f})...")
        try:
            samples = generate_llm_samples(
                provider=provider,
                hc3_pairs=hc3_pairs,
                sample_count=sample_count,
                paired_ratio=settings.paired_ratio,
                manifest_path=generated_dir / f"{provider.name}_manifest.json",
                output_path=generated_dir / f"{provider.name}_samples.jsonl",
                budget_usd=budget,
            )
        except Exception as exc:
            # A provider-level failure (bad API key, insufficient credit, account issue)
            # shouldn't take down the other providers, which are independent accounts.
            print(f"    {provider.name} failed, skipping: {exc}")
            continue
        print(f"    {len(samples)} samples generated from {provider.name}")

    # Reload every provider's samples from disk -- not just the ones this run targeted --
    # so the final dataset always reflects everything generated so far.
    generated_samples = []
    for provider_name in ("claude", "gpt4o", "gemini"):
        provider_samples = _load_jsonl(generated_dir / f"{provider_name}_samples.jsonl", GeneratedSample)
        generated_samples.extend(provider_samples)

    print(f"==> Building final dataset ({len(generated_samples)} total generated samples)...")
    result = build_dataset(
        hc3_pairs=hc3_pairs,
        wikipedia_samples=wikipedia_samples,
        generated_samples=generated_samples,
        output_dir=generated_dir,
    )
    for split_name, rows in result.items():
        print(f"    {split_name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
