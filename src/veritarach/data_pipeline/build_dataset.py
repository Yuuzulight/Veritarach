import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from veritarach.data_pipeline.fetch_hc3 import HC3Pair
from veritarach.data_pipeline.fetch_wikipedia import HumanTextSample
from veritarach.data_pipeline.generate_llm_samples import GeneratedSample


@dataclass
class DatasetRow:
    text: str
    label: str
    source: str
    generation_strategy: str | None
    pair_id: str | None
    split: str


def build_dataset(
    hc3_pairs: list[HC3Pair],
    wikipedia_samples: list[HumanTextSample],
    generated_samples: list[GeneratedSample],
    output_dir: Path,
    seed: int = 42,
) -> dict[str, list[DatasetRow]]:
    """Returns {"train": [...], "val": [...], "test": [...]} and writes each list to
    output_dir/{split}.jsonl."""
    hc3_rows = _expand_hc3(hc3_pairs)
    generated_rows = _expand_generated(generated_samples)

    # HC3 contributes equally to both classes by construction (one human + one AI row per
    # pair), so the only imbalance comes from generated_samples (AI-only) vs. Wikipedia
    # (human-only). Wikipedia is trimmed to match exactly -- never HC3.
    wikipedia_needed = len(generated_samples)
    if len(wikipedia_samples) < wikipedia_needed:
        raise ValueError(
            f"build_dataset: need {wikipedia_needed} Wikipedia rows to balance classes, "
            f"only {len(wikipedia_samples)} available -- re-run fetch_wikipedia with a "
            f"higher sample_count"
        )
    rng = random.Random(seed)
    wikipedia_rows = _expand_wikipedia(rng.sample(wikipedia_samples, wikipedia_needed))

    draft_rows = hc3_rows + wikipedia_rows + generated_rows
    groups = [
        row.pair_id if row.pair_id is not None else f"__solo__{i}" for i, row in enumerate(draft_rows)
    ]

    train_idx, rest_idx = _group_split(len(draft_rows), groups, train_size=0.8, seed=seed)
    rest_groups = [groups[i] for i in rest_idx]
    val_in_rest, test_in_rest = _group_split(len(rest_idx), rest_groups, train_size=0.5, seed=seed)
    val_idx = [rest_idx[i] for i in val_in_rest]
    test_idx = [rest_idx[i] for i in test_in_rest]

    result = {
        "train": [replace(draft_rows[i], split="train") for i in train_idx],
        "val": [replace(draft_rows[i], split="val") for i in val_idx],
        "test": [replace(draft_rows[i], split="test") for i in test_idx],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in result.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(asdict(row)) + "\n")

    return result


def _expand_hc3(hc3_pairs: list[HC3Pair]) -> list[DatasetRow]:
    rows = []
    for pair in hc3_pairs:
        rows.append(
            DatasetRow(
                text=pair.human_answer,
                label="human_written",
                source="hc3",
                generation_strategy=None,
                pair_id=pair.pair_id,
                split="",
            )
        )
        rows.append(
            DatasetRow(
                text=pair.ai_answer,
                label="ai_generated",
                source="hc3",
                generation_strategy=None,
                pair_id=pair.pair_id,
                split="",
            )
        )
    return rows


def _expand_wikipedia(samples: list[HumanTextSample]) -> list[DatasetRow]:
    return [
        DatasetRow(
            text=sample.text,
            label="human_written",
            source="wikipedia",
            generation_strategy=None,
            pair_id=None,
            split="",
        )
        for sample in samples
    ]


def _expand_generated(samples: list[GeneratedSample]) -> list[DatasetRow]:
    return [
        DatasetRow(
            text=sample.text,
            label="ai_generated",
            source=sample.source,
            generation_strategy=sample.generation_strategy,
            pair_id=sample.pair_id,
            split="",
        )
        for sample in samples
    ]


def _group_split(n: int, groups: list[str], train_size: float, seed: int) -> tuple[list[int], list[int]]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, rest_idx = next(splitter.split(X=list(range(n)), groups=groups))
    return list(train_idx), list(rest_idx)
