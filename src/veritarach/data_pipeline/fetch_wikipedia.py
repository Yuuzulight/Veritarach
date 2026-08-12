import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

MIN_PARAGRAPH_CHARS = 200
MAX_PARAGRAPH_CHARS = 2000


@dataclass
class HumanTextSample:
    source: str
    text: str


def fetch_wikipedia(output_dir: Path, sample_count: int = 500, seed: int = 42) -> list[HumanTextSample]:
    """Streams wikimedia/wikipedia (20231101.en config), keeps paragraphs between
    MIN_PARAGRAPH_CHARS and MAX_PARAGRAPH_CHARS characters, randomly samples
    `sample_count` of them (seeded), writes output_dir/wikipedia_samples.jsonl, and
    returns the samples."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    # Stop pulling from the stream once there's a comfortable oversample to pick from --
    # iterating the entire multi-million-article stream just to sample a few hundred
    # paragraphs would defeat the point of streaming in the first place.
    candidate_pool_size = max(sample_count * 10, 2000)
    candidates: list[str] = []
    for row in dataset:
        for paragraph in row["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if MIN_PARAGRAPH_CHARS <= len(paragraph) <= MAX_PARAGRAPH_CHARS:
                candidates.append(paragraph)
                if len(candidates) >= candidate_pool_size:
                    break
        if len(candidates) >= candidate_pool_size:
            break

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(sample_count, len(candidates)))
    samples = [HumanTextSample(source="wikipedia", text=text) for text in chosen]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wikipedia_samples.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + "\n")

    return samples
