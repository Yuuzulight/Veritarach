import json
from pathlib import Path

from datasets import Dataset

LABEL_TO_ID = {"human_written": 0, "ai_generated": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_split(path: Path) -> Dataset:
    """Loads a JSONL dataset split (as produced by build_dataset.py) into a HF Dataset
    with just `text` and `label` columns, label mapped to 0/1 via LABEL_TO_ID."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append({"text": row["text"], "label": LABEL_TO_ID[row["label"]]})
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Tokenizes the `text` column, keeping `label`. Drops `text` afterward since the
    Trainer only needs the tokenized fields."""

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])
