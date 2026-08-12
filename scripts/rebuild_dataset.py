"""Rebuilds the final train/val/test dataset from already-fetched/generated data on
disk, without re-running any fetching or generation. Useful after a generation run was
interrupted or a provider hit its limit -- combines whatever *_samples.jsonl files exist
without re-attempting any provider calls.

Usage: uv run python scripts/rebuild_dataset.py
"""

import json
from pathlib import Path

from veritarach.config import get_settings
from veritarach.data_pipeline.build_dataset import build_dataset
from veritarach.data_pipeline.fetch_hc3 import HC3Pair
from veritarach.data_pipeline.fetch_wikipedia import HumanTextSample
from veritarach.data_pipeline.generate_llm_samples import GeneratedSample


def _load_jsonl(path: Path, cls):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [cls(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw"
    generated_dir = settings.data_dir / "generated"

    hc3_pairs = _load_jsonl(raw_dir / "hc3_pairs.jsonl", HC3Pair)
    wikipedia_samples = _load_jsonl(raw_dir / "wikipedia_samples.jsonl", HumanTextSample)
    print(f"hc3: {len(hc3_pairs)} pairs, wikipedia: {len(wikipedia_samples)} samples")

    generated_samples = []
    for provider_name in ("claude", "gpt4o", "gemini"):
        samples = _load_jsonl(generated_dir / f"{provider_name}_samples.jsonl", GeneratedSample)
        print(f"{provider_name}: {len(samples)} samples on disk")
        generated_samples.extend(samples)

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
